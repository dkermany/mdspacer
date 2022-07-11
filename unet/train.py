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
from datetime import datetime
from unet.model import UNET
from unet.dataset import COCODataset, InferenceDataset
from unet.metrics import UNetMetrics
from tools.utils import (create_directory, get_COCO_transforms,
                         get_inference_loader, get_COCO_loaders,
                         get_CoNSeP_loaders)
from tools.checks import _check_save_labels_as_rgb_input

# Hyperparameters
LEARNING_RATE = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if torch.cuda.is_available() else False
BATCH_SIZE = 100
NUM_CLASSES = 80 + 1 # Background class
NUM_EPOCHS = 5
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
        image_path: str,
        mask_path: str,
        image_size: int,
        num_classes: int,
        batch_size: int = 1,
        num_epochs: int = 1,
        learning_rate: float = 3e-4,
        checkpoint_dir: str = "",
        freeze: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        self.image_path = os.path.normpath(image_path)
        self.mask_path = os.path.normpath(mask_path)
        self.image_size = image_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.freeze = freeze
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.step = 0
        self.best_val_loss = 9999.

        # Initialize UNet model
        self.model = UNET(in_channels=3, out_channels=self.num_classes)

        # Initialize loss function
        # TODO: Add class_weights
        if self.num_classes == 2:
            self.loss_fn = nn.BCEWithLogitsLoss()
        if self.num_classes > 2:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Class number must be at least 2 or greater.\
                             Received: ({self.num_classes})")

        # Initialize optimizer
        # passing only those parameters that explicitly require grad
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
        )

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

    def freeze(self):

        # Don't load weights for final_layer since we will be retraining this
        #if freeze:
        #    checkpoint_state_dict = for key in checkpoint["state_dict"]

        # Unfrozen, to continue or evaluate
        if not freeze:
            for param in self.model.parameters():
                param.requires_grad = True
        # Freeze all layers, except final layer
        else:
            for name, param in self.model.named_parameters():
                if name.startswith("final_conv"):
                    param.requires_grad = True
                else:
                    param.requires_grad = False


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

        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


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
        one_hot = nn.functional.one_hot(
            label.long(),
            num_classes=self.num_classes
        )
        return one_hot.permute(0,3,1,2)

    def inference(self, inference_ext: str):
        """
        Runs inference on InferenceDataset
        """
        self.model.eval()

        # Load transforms
        transforms = get_COCO_transforms(self.image_size)

        # Get Loaders
        inference_ds, inference_loader = get_inference_loader(
            image_path=self.image_path,
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
            self._save_labels_as_rgb(predictions, filenames)

        self.model.train()

    def _save_labels_as_rgb(self, labels: np.ndarray, filenames: str):
        """
        Expects label as a batch of numpy arrays in shape (N, size, size),
        where N is the batch size. Creates a folder at the same directory level
        as self.image_path named "{self.image_path}_predictions where labels are
        saved"
        """
        _check_save_labels_as_rgb_input(labels)

        # Set output directory
        output_dir = os.path.join(
            os.path.dirname(self.image_path),
            f"{os.path.basename(self.image_path)}_predictions",
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

    def _run_epoch(
        self,
        loader: DataLoader[object],
        epoch: int,
        train: bool,
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
            # Get batch data and load to GPU
            data = data.float().to(device=DEVICE)
            targets = self.one_hot_encoding(targets).float().to(device=DEVICE)

            # forward
            with torch.cuda.amp.autocast():
                with torch.set_grad_enabled(train):
                    logits = self.model(data)
                    loss = self.loss_fn(logits, targets)

            # backward
            if train:
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            targets = torch.argmax(targets, axis=1).int()
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

    def run_consep(self):
        """
        Runs training and validation on consep. Calculates validation scores every epoch.
        Training metrics are not averages together until the end of training.
        Begins from loaded checkpoint
        """
        # Load transforms
        transforms = self.get_transforms()

        # Get Loaders
        train_loader, val_loader = get_consep_loaders(
            image_path=self.image_path,
            mask_path=self.mask_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            transforms=transforms,
        )

    def run_coco(self):
        """
        Runs training and validation. Calculates validation scores every epoch.
        Training metrics are not averages together until the end of training.
        """
        # Load transforms
        transforms = get_COCO_transforms(self.image_size)

        # Get Loaders
        train_loader, val_loader = get_COCO_loaders(
            image_path=self.image_path,
            mask_path=self.mask_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            transforms=transforms,
        )

        for epoch in range(self.num_epochs):
            # Training epoch
            train_results = self._run_epoch(train_loader, epoch, train=True)
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
        image_path=FLAGS.images,
        mask_path=FLAGS.masks,
        image_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        checkpoint_dir=FLAGS.checkpoint,
        freeze=FLAGS.freeze,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # Run inference
    if FLAGS.masks == "":
        print("==> Running UNet in INFERENCE mode")
        runner.inference(FLAGS.inference_ext)
        return

    # Run training from scratch
    if FLAGS.checkpoint != "":
        print("==> Running UNet in TRAINING mode on the COCO Dataset")
        runner.run_coco()
        return

    # Run retraining with new final layer from checkpoint
    if FLAGS.freeze:
        print("==> Running UNet in TRAINING mode on the CoNSeP Dataset")
        runner.run_consep()
        return

    # Continue training from checkpoint
    raise NotImplementedError("Need to resolve metrics and plotting issues")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images",
        required=True,
        type=str,
        help="Path to train/val/test images directory"
    )
    parser.add_argument(
        "--masks",
        default="",
        type=str,
        help="Path to train/val/test masks directory. If empty, runs inference\
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
