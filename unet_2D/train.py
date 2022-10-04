import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import albumentations as A
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import skimage
import cv2
from torch import Tensor
from torch.utils.data.dataloader import Dataset, DataLoader
from torchinfo import summary
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from patchify import patchify
from datetime import datetime
from unet.model import UNET, ResNetUNet
from unet.dataset import COCODataset, InferenceDataset, get_class_weights
from unet.metrics import UNetMetrics
from tools.utils import (create_directory, get_COCO_transforms,
                         get_inference_loader, get_COCO_loaders,
                         get_CoNSeP_loaders, get_CoNSeP_transforms)
from tools.checks import _check_save_labels_as_rgb_input

# Hyperparameters
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if torch.cuda.is_available() else False
BATCH_SIZE = 5
#NUM_CLASSES = 80 + 1 # COCO + Background class
#NUM_CLASSES = 3 + 1 # CoNSeP + Background class
NUM_CLASSES = 1 # CoNSeP binary + Background class
NUM_EPOCHS = 50
NUM_WORKERS = 0 * max(1, torch.cuda.device_count())
IMAGE_SIZE = 256
CHECKPOINT_DIR = "/home/dkermany/BoneSegmentation/checkpoints/{}".format(
    datetime.today().strftime("%m%d%Y")
)

class UNetRunner:
    """
    Performs training and transfer learning using UNet

    Default behavior is to train learnable parameters from scratch,
    however passing a path to a *.pth.tar checkpoint in the checkpoint_dir
    parameter will:
        1. Freeze all layers


    Usage
        > trainer = UNetRunner(**args)
        > trainer.run()

    Arguments:
        - checkpoint_dir (str): path to .pth.tar file containing UNet
                                 parameters to reinitialize
        - freeze (bool): If False (default), all learnable parameters are
                         left unfrozen. If True, all weights except for
                         the final convolutional layer will be frozen for
                         retraining of final layer, followed by unfreezing
                         the layers and finetuning at a lower learning
                         rate.
    """
    def __init__(
        self,
        image_size: int,
        num_classes: int,
        batch_size: int = 1,
        num_epochs: int = 1,
        learning_rate: float = 3e-4,
        checkpoint_dir: str = "",
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last_layer: bool = False,
    ):
        self.image_size = image_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last_layer = drop_last_layer

        self.step = 0
        self.best_val_loss = 9999.
        self.freeze = False

        #if self.num_classes < 2:
        #    raise ValueError(f"Class number must be at least 2 or greater.\
        #                     Received: ({self.num_classes})")

        # Initialize UNet model
        # self.model = UNET(in_channels=3, out_channels=self.num_classes)
        self.model = ResNetUNet(self.num_classes)

        # Initialize optimizer
        # passing only those parameters that explicitly require grad
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
        )

        # If before DataParallel, keys will not have module. prefix. Otherwise,
        # keys in state_dict will have module. prefix
        if self.checkpoint_dir != "":
            self.load_checkpoint()

        # Casts model to selected device (CPU vs GPU)
        self.model.to(device=DEVICE)

        # Parallelizes model across multiple GPUs
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs...")
            self.model = nn.DataParallel(self.model)

        self.scaler = torch.cuda.amp.GradScaler()
        create_directory(CHECKPOINT_DIR)

        # Create UNetMetrics instance
        self.metrics = UNetMetrics(self.num_classes, device=DEVICE)

    def freeze_model(self):
        # Don't load weights for final_layer since we will be retraining this
        self.freeze = True
        for name, param in self.model.named_parameters():
            name = name.removeprefix("module.")
            if name.startswith("final_conv") or\
               name.startswith("decoder.7") or\
               name.startswith("decoder.6"):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_model(self):
        self.freeze = False
        for name, param in self.model.named_parameters():
            param.requires_grad = True

    def load_checkpoint(self):
        """
        """
        if not self.checkpoint_dir.endswith(".pth.tar"):
            err = f"""Checkpoint {self.checkpoint_dir} not valid checkpoint file
                      type (expected: .pth.tar)"""
            raise ValueError(err)

        print(f"==> Loading checkpoint: {self.checkpoint_dir}")
        # loads checkpoint into model and optimizer using state dicts
        checkpoint = torch.load(self.checkpoint_dir)

        # Convert dataparallel weights
        for i in ["state_dict", "optimizer"]:
            checkpoint[i] = {
                k.removeprefix("module."): v
                for k, v in checkpoint[i].items()
            }

        # Model state_dict
        model_dict = self.model.state_dict()

        # Do not load final layer
        if self.drop_last_layer:
            checkpoint["state_dict"] = {
                k: v for k, v in checkpoint["state_dict"].items()
                if not k.startswith("final_conv") and\
                   not k.startswith("decoder.7") and\
                   not k.startswith("decoder.6")
            }
            not_loaded = [k for k, v in checkpoint["state_dict"].items()
                          if k not in model_dict]
            print("NOT LOADED: ", not_loaded)

        assert len(checkpoint["state_dict"].keys()) > 1

        # Load the new state dict
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        #self.optimizer.load_state_dict(checkpoint["optimizer"], strict=False)

    def save_checkpoint(self, state: dict[str, object], filename: str) -> None:
        print(f"=> Saving checkpoint: {filename}")
        torch.save(state, filename)


    def one_hot_encoding(self, label: torch.Tensor) -> torch.Tensor:
        """
        One-Hot Encoding for segmentation masks
        Example: Converts (batch, 256, 256) => (batch, num_classes, 256, 256)
        with convention of (B, C, H, W)
        """
        # nn.function.one_hot returns in CHANNEL_LAST formatting
        # permute needed to convert to CHANNEL_FIRST
        #label[label < 0] = 0
        one_hot = nn.functional.one_hot(
            label.long(),
            num_classes=self.num_classes
        )
        return one_hot.permute(0,3,1,2)

    def inference(self, image_path: str, inference_ext: str):
        """
        Runs inference on InferenceDataset
        """
        self.model.eval()

        # Load transforms
        transforms = get_COCO_transforms(self.image_size)

        # Get Loaders
        inference_ds, inference_loader = get_inference_loader(
            image_path=image_path,
            image_ext=inference_ext,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            transform=transforms["inference"],
        )

        # Loop over inference DataLoader
        for batch_idx, (data, filenames) in enumerate(tqdm(inference_loader)):
            data.float().to(device=DEVICE)

            # Enables automatic casting between F16 & F32
            with torch.cuda.amp.autocast():
                # Disable gradient calculations
                with torch.no_grad():
                    # forward
                    logits = self.model(data)

            predictions = torch.argmax(logits, dim=1).detach().cpu().numpy()
            self._save_labels_as_rgb(predictions, filenames, image_path)

        self.model.train()

    def _save_labels_as_rgb(
        self,
        labels: np.ndarray,
        filenames: str,
        image_path: str
    ):
        """
        Expects label as a batch of numpy arrays in shape (N, size, size),
        where N is the batch size. Creates a folder at the same directory level
        as image_path named "{image_path}_predictions where labels are
        saved"
        """
        _check_save_labels_as_rgb_input(labels)

        # Set output directory
        output_dir = os.path.join(
            os.path.dirname(image_path),
            f"{os.path.basename(image_path)}_predictions",
        )
        create_directory(output_dir)

        # Initialize color palette with self.num_classes different colors
        palette = sns.color_palette("Set1", self.num_classes)

        # using builtin list() function to only convert the outer-most
        # dimension to a list, where as the np.ndarray.tolist() function
        # recursively converts all np.ndarrays into builtin lists
        # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tolist.html
        for label, filename in zip(list(labels), filenames):
            rgb_label = skimage.color.label2rgb(
                label,
                bg_label=0,
                colors=palette,
            )

            # Scale predictions from range (0, 1) to (0, 255)
            rgb_label = (rgb_label * 255).astype("uint8")

            # Convert from RGB formatting to BGR to use cv2 functions
            bgr_label = cv2.cvtColor(rgb_label, cv2.COLOR_RGB2BGR)

            # Save label
            output_filename = os.path.join(output_dir, filename + ".png")
            cv2.imwrite(output_filename, bgr_label)

    def _get_patches(self, data: Tensor, targets: Tensor) -> tuple[Tensor]:
        assert data.shape[2] % self.image_size == 0
        assert targets.shape[2] % self.image_size == 0
        np_data = data.numpy()
        np_targets = targets.numpy()

        # Patches per image
        ppi = int(data.shape[2] / self.image_size) ** 2

        final_data = np.zeros((
            ppi * self.batch_size,
            3,
            self.image_size,
            self.image_size,
        ))
        final_targets = np.zeros((
            ppi * self.batch_size,
            self.image_size,
            self.image_size,
        ))
        for i, image in enumerate(list(np_data)):
            data_patches = patchify(
                image,
                (3, self.image_size, self.image_size),
                step=self.image_size,
            ).reshape(-1, 3, self.image_size, self.image_size)
            final_data[i*ppi: (i+1)*ppi] = data_patches

        for i, target in enumerate(list(np_targets)):
            targets_patches = patchify(
                target,
                (self.image_size, self.image_size),
                step=self.image_size,
            ).reshape(-1, self.image_size, self.image_size)
            final_targets[i*ppi: (i+1)*ppi] = targets_patches

        return Tensor(final_data).int(), Tensor(final_targets).int()

    def _run_epoch(
        self,
        loader: DataLoader[object],
        epoch: int,
        train: bool,
        create_patches: bool = False,
    ) -> dict[str, float]:
        """
        Performs one epoch over provided `loader` (DataLoader)

        Arguments:
            - loader (DataLoader): training or validation pytorch dataloader
            - epoch (int): current epoch number
            - train (bool): Whether or not to enable gradients
        """
        if train:
            self.model.train()
            self.metrics.train()
        else:
            self.model.eval()
            self.metrics.eval()

        # Running loss variable
        epoch_loss = 0.0

        loop = tqdm(loader)
        for i, (data, targets) in enumerate(loop):
            if create_patches:
                data, targets = self._get_patches(data, targets)

            # Get batch data and load to GPU
            data = data.float().to(device=DEVICE)
            if self.num_classes > 2:
                targets = self.one_hot_encoding(targets)
            targets = targets.float().to(device=DEVICE)

            # forward
            with torch.cuda.amp.autocast():
                with torch.set_grad_enabled(train):
                    logits = self.model(data)
                    loss = self.loss_fn(torch.squeeze(logits), targets)

            # backward
            if train:
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            #targets = torch.argmax(targets, axis=1).int()
            targets = targets.int()
            predictions = torch.argmax(logits, dim=1)

            # Update metrics
            r = self.metrics.update_metrics(predictions, targets)
            acc, dice = r["acc"], r["dice"]

            # Updating running loss
            epoch_loss += loss.item()

            # Training plot is updated every batch and the step is incremented
            # for each training batch
            if train:
                self.metrics.update_plot(self.step, loss.item(), acc, dice)

                # Increment plot step (x-axis)
                self.step += 1

            # update tqdm loop
            stage = "Train" if train else "Validation"
            desc = (
                f"{stage} Epoch: {epoch+1}/{self.num_epochs} - Loss: "
                f"{loss.item():.4f}, Acc: {acc:.4f}, Dice: {dice:.4f}"
            )
            loop.set_description(desc)

        # Calculate average metrics
        results = self.metrics.compute()

        # Validation plot is updated only once every training epoch and the
        # step is NOT incremented
        if not train:
            self.metrics.update_plot(
                self.step,
                epoch_loss,
                results["acc"],
                results["dice"],
            )

        # Average loss over epoch and add it to results dict
        epoch_loss /= len(loader)
        results["loss"] = epoch_loss

        return results

    def run_CoNSeP(self, train_path: str, val_path: str):
        """
        Runs training and validation on consep. Calculates validation scores every epoch.
        Training metrics are not averages together until the end of training.
        Begins from loaded checkpoint
        """
        # Load transforms
        transforms = get_CoNSeP_transforms(self.image_size)

        # Get Loaders
        train_loader, val_loader = get_CoNSeP_loaders(
            train_path=train_path,
            val_path=val_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            transforms=transforms,
        )

        # class_weights = get_class_weights(
        #     val_loader,
        #     self.num_classes
        # ).to(device=DEVICE)

        # Initialize loss function
        if self.num_classes <= 2:
            self.loss_fn = nn.BCEWithLogitsLoss()
        if self.num_classes > 2:
            self.loss_fn = nn.CrossEntropyLoss()

        # Freeze all learnable parameters except the final layer
        # print("==> Freezing model")
        # self.freeze_model()

        #for i, stage in enumerate(["feature_extracter", "fine_tune"]):
        #    if i+1 == 2:
        #        # print("==> Unfreezing model")
        #        # self.unfreeze_model()
        #        for g in self.optimizer.param_groups:
        #            g["lr"] = self.learning_rate * 0.5

        for epoch in range(self.num_epochs):
            # Training epoch
            train_results = self._run_epoch(train_loader, epoch, train=True)

            # Validation epoch
            val_results = self._run_epoch(
                val_loader,
                epoch,
                train=False,
                create_patches=True,
            )

            print("-" * 75)
            print(f"Epoch {epoch+1} Average Metrics")
            print(f"""Train\n\tLoss: {train_results["loss"]:.4f},\
                  \n\tAcc: {train_results["acc"]:.4f},\
                  \n\tDice: {train_results["dice"]:.4f}""")
            print(f"""Validation\n\tLoss: {val_results["loss"]:.4f},\
                  \n\tAcc: {val_results["acc"]:.4f},\
                  \n\tDice: {val_results["dice"]:.4f}""")
            print("-" * 75)

            # If this is the best performance so far, save checkpoint
            if val_results["loss"] < self.best_val_loss:
                self.best_val_loss = val_results["loss"]

                # Save model
                checkpoint = {
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                self.save_checkpoint(
                    checkpoint,
                    filename=os.path.join(CHECKPOINT_DIR,
                                  f"resnetunet1.0-consep-binary-epoch{epoch+1}.pth.tar")
                )

            # Save metrics to CSV file. Runs after every epoch to update the file
            # as training progresses, rather than waiting until the end. This
            # ensures data is not loss in the case of script interruption
            self.metrics.write_to_file(CHECKPOINT_DIR)


        # Save training and validation plot as image (.PNG)
        self.metrics.save_plots(CHECKPOINT_DIR)

    def run_COCO(self, image_path: str, mask_path: str):
        """
        Runs training and validation. Calculates validation scores every epoch.
        Training metrics are not averages together until the end of training.
        """
        # Load transforms
        transforms = get_COCO_transforms(self.image_size)

        # Get Loaders
        train_loader, val_loader = get_COCO_loaders(
            image_path=image_path,
            mask_path=mask_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            transforms=transforms,
        )

        class_weights = get_class_weights(
            val_loader,
            self.num_classes
        ).to(device=DEVICE)

        # Initialize loss function
        if self.num_classes <= 2:
            self.loss_fn = nn.BCEWithLogitsLoss(weight=class_weights)
        if self.num_classes > 2:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        for epoch in range(self.num_epochs):
            # Training epoch
            train_results = self._run_epoch(train_loader, epoch, train=True)

            # Validation epoch
            val_results = self._run_epoch(val_loader, epoch, train=False)

            print("-" * 75)
            print(f"Epoch {epoch+1} Average Metrics")
            print(f"""Train\n\tLoss: {train_results["loss"]:.4f},\
                  \n\tAcc: {train_results["acc"]:.4f},\
                  \n\tDice: {train_results["dice"]:.4f}""")
            print(f"""Validation\n\tLoss: {val_results["loss"]:.4f},\
                  \n\tAcc: {val_results["acc"]:.4f},\
                  \n\tDice: {val_results["dice"]:.4f}""")
            print("-" * 75)

            # If this is the best performance so far, save checkpoint
            if val_results["loss"] < self.best_val_loss:
                self.best_val_loss = val_results["loss"]

                # Save model
                checkpoint = {
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                self.save_checkpoint(
                    checkpoint,
                    filename=os.path.join(CHECKPOINT_DIR,
                                  f"unet1.0-coco-epoch{epoch+1}.pth.tar")
                )

            # Save metrics to CSV file. Runs after every epoch to update the file
            # as training progresses, rather than waiting until the end. This
            # ensures data is not loss in the case of script interruption
            self.metrics.write_to_file(CHECKPOINT_DIR)

        # Save training and validation plot as image (.PNG)
        self.metrics.save_plots(CHECKPOINT_DIR)


def main():
    # Initialize runner
    runner = UNetRunner(
        image_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        checkpoint_dir=FLAGS.checkpoint,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last_layer=FLAGS.freeze,
    )

    # Run inference
    if FLAGS.masks == "":
        print("==> Running UNet in INFERENCE mode")
        runner.inference(os.path.normpath(FLAGS.images), FLAGS.inference_ext)
        return

    # Run training from scratch
    if FLAGS.checkpoint == "":
        print("==> Running UNet in TRAINING mode")
        # runner.run_COCO(
        #     image_path=FLAGS.images,
        #     mask_path=FLAGS.masks,
        # )
        runner.run_CoNSeP(
            train_path=FLAGS.images,
            val_path=FLAGS.masks,
        )
        return

    # Run retraining with new final layer from checkpoint
    if FLAGS.freeze:
        print("==> Running UNet in TRAINING mode on the CoNSeP Dataset")
        runner.run_CoNSeP(
            train_path=FLAGS.images,
            val_path=FLAGS.masks,
        )
        return

    # Continue training from checkpoint
    raise NotImplementedError("Need to resolve metrics and plotting issues")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images",
        required=True,
        type=str,
        help="Path to images or train directory"
    )
    parser.add_argument(
        "--masks",
        default="",
        type=str,
        help="Path to masks or val directory. If empty, runs inference\
              on images"
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        type=str,
        help="Path to checkpoint (.pth.tar) from which to load"
    )
    parser.add_argument(
        "--freeze",
        default=False,
        type=bool,
        help="Whether to perform training or inference on validation"
    )
    parser.add_argument(
        "--inference_ext",
        default="jpg",
        type=str,
        help="Image extension for inference files"
    )

    FLAGS, _ = parser.parse_known_args()

    # Check FLAGS
    err = None
    if FLAGS.freeze and FLAGS.checkpoint == "":
        err = "Must specify a checkpoint file (.pth.tar) for freeze"
    if FLAGS.masks == "" and FLAGS.checkpoint == "":
        err = "Must specify a checkpoint file (.pth.tar) for inference"
    if not os.path.exists(FLAGS.images):
        err = "Invalid path given for --images"
    if err:
        raise ValueError(err)

    main()
