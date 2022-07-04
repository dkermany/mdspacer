import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import albumentations as A
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from torchmetrics import JaccardIndex, Accuracy
from torchinfo import summary
from albumentations.pytorch import ToTensorV2
from os.path import abspath, join, dirname, normpath
from tqdm import tqdm
from model import UNET
from dataset import COCODataset

sys.path.append(abspath(join(dirname(__file__), "..", "tools")))
from utils import (
    load_checkpoint,
    save_checkpoint,
    create_directory,
)

# Hyperparameters
LEARNING_RATE = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if torch.cuda.is_available() else False
BATCH_SIZE = 100
NUM_CLASSES = 80 + 1 # Background class
NUM_EPOCHS = 80
NUM_WORKERS = 4
IMAGE_SIZE = 256
TENSORBOARD_PATH = "/home/dkermany/BoneSegmentation/tensorboard/COCO"

CHECKPOINT_PATH = "/home/dkermany/BoneSegmentation/checkpoints/{}".format(
    datetime.today().strftime("%m%d%Y")
)

class UNetTrainer:
    """
    Performs training and transfer learning using UNet

    Default behavior is to train learnable parameters from scratch,
    however passing a path to a *.pth.tar checkpoint in the load_path
    parameter will:
        1. Freeze all layers


    Usage
        > trainer = UNetTrainer(**params)
        > trainer.run()
    """
    def __init__(
        self,
        image_size: int,
        num_classes: int,
        batch_size: int = 1,
        num_epochs: int = 1,
        learning_rate: float = 3e-4,
        num_workers: int = 1,
        pin_memory: bool = False,
        load_path: str = "",
        tensorboard_path: str = "./",
    ):
        self.image_size = image_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.load_path = load_path
        self.tensorboard_path = tensorboard_path

        # Initialize UNet model
        self.model = UNET(in_channels=3, out_channels=self.num_classes)

        # Initialize loss function
        # TODO: Add class_weights after implementing
        self.loss_fn = nn.CrossEntropyLoss()

        # Initialize optimizer
        # passing only those paramters that explicitly require grad
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
        )

        # Casts model to selected device (CPU vs GPU)
        self.model.to(device=DEVICE)

        # Parallelizes model across multiple GPUs
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs...")
            self.model = nn.DataParallel(self.model)

        self.scaler = torch.cuda.amp.GradScaler()
        create_directory(CHECKPOINT_PATH)

        self.batch_idx = 0
        self.metrics = {"loss": [], "acc": []}

    def get_transforms(self) -> dict[str, A.Compose]:
        train_transform = A.Compose(
            [
                A.Resize(
                    height=int(self.image_size*1.13),
                    width=int(self.image_size*1.13),
                ),
                A.RandomCrop(height=self.image_size, width=self.image_size),
                A.Rotate(limit=90, p=0.9),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.CLAHE(clip_limit=4.0, p=0.35),
                A.ColorJitter(p=0.15),
                A.GaussNoise(p=0.15),
                A.Normalize(
                    mean=[0.4690, 0.4456, 0.4062],
                    std=[0.2752, 0.2701, 0.2847],
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ]
        )
        val_transform = A.Compose(
            [
                A.Resize(height=self.image_size, width=self.image_size),
                A.Normalize(
                    mean=[0.4690, 0.4456, 0.4062],
                    std=[0.2752, 0.2701, 0.2847],
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ]
        )
        return {"train": train_transform, "val": val_transform}

    def get_coco_loaders(
        self,
        transforms: dict[str, A.Compose]
    ) -> tuple[DataLoader[Any]]:

        train_ds = COCODataset(
            join(FLAGS.images, "train2017"),
            join(FLAGS.masks, "train2017"),
            transform=transforms["train"]
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        val_ds = COCODataset(
            join(FLAGS.images, "val2017"),
            join(FLAGS.masks, "val2017"),
            transform=transforms["val"]
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        return train_loader, val_loader

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

    def train_fn(
        self,
        loader: DataLoader[Any],
        epoch: int,
    ):
        loop = tqdm(loader)
        for i, (data, targets) in enumerate(loop):
            data = data.float().to(device=DEVICE)
            targets = self.one_hot_encoding(targets).float().to(device=DEVICE)

            # forward
            with torch.cuda.amp.autocast():
                logits = self.model(data)
                loss = self.loss_fn(logits, targets)

            # backward
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Updates JaccardIndex with batch
            targets = torch.argmax(targets, axis=1).int()
            predictions = torch.argmax(logits, dim=1)

            # Pixel-wise accuracy
            accuracy = (predictions == targets).sum() / torch.numel(targets)

            # Add metrics
            self.batch_idx = epoch * len(loader) + i
            self.metrics["loss"].append((self.batch_idx, loss.item(), "train"))
            self.metrics["acc"].append((self.batch_idx, accuracy.item(), "train"))

            # update tqdm loop
            loop.set_description("Epoch: {}/{} - Loss: {:.2f}, Acc: {:.2%}".format(
                epoch+1,
                self.num_epochs,
                loss.item(),
                accuracy.item()
            ))

    def validate_fn(self, loader: DataLoader[Any]) -> dict:
        """
        Performs inference on the entire validation loader and calculates
        loss and mean jaccard index. Meant to be used at the end of training
        since the function takes about 40minutes to run
        """
        # TODO: Class-Weighted Pixel-Wise Accuracy Metric
        # TODO: Class-Weighted Multiclass Dice Coefficient
        # Set model into evaluation mode
        self.model.eval()
        with torch.no_grad():

            # Class-Weighted mIoU Score
            jaccard = JaccardIndex(
                num_classes=self.num_classes,
                average="micro",
                ignore_index=0,
                mdmc_average="global",
            ).to(device=DEVICE)

            for data, targets in tqdm(loader):
                data = data.float().to(device=DEVICE)
                targets = self.one_hot_encoding(targets).float().to(device=DEVICE)
                with torch.cuda.amp.autocast():
                    logits = torch.softmax(self.model(data), dim=1)
                    loss = self.loss_fn(logits, targets)

                # Change tensor shape 
                # (batch, n_classes, size, size) -> (batch, size, size)
                targets = torch.argmax(targets, axis=1).int()
                predictions = torch.argmax(logits, dim=1)

                # Pixel-wise accuracy
                accuracy += (predictions == targets).sum() / torch.numel(targets)

                # Updates JaccardIndex with batch
                j = jaccard(predictions, targets)

            mean_accuracy = accuracy / len(loader)

            # Average over all batches uisng mdmc_average method
            total_jaccard = jaccard.compute()

        # Return model to train mode
        self.model.train()

        return {
            "loss": loss,
            "jaccard": total_jaccard,
            "accuracy": mean_accuracy,
        }

    def validate_one_batch(self, loader: DataLoader[Any]) -> dict:
        """
        Performs inference only on the first batch of the validation set and
        calculates loss and the mean jaccard index. Meant to be used after
        each epoch to track the model's performance over the epochs
        """
        # Set model into evaluation mode
        self.model.eval()
        with torch.no_grad():

            # Get only the same first batch of images
            # Make sure loader shuffle is set to False to ensure geting the
            # same batch each time
            data, targets = next(iter(loader))

            data = data.float().to(device=DEVICE)
            targets = self.one_hot_encoding(targets).float().to(device=DEVICE)
            with torch.cuda.amp.autocast():
                logits = torch.softmax(self.model(data), dim=1)
                loss = self.loss_fn(logits, targets)

            # Change tensor shape 
            # (batch, n_classes, size, size) -> (batch, size, size)
            targets = torch.argmax(targets, axis=1).int()
            predictions = torch.argmax(logits, dim=1)

            # Pixel-wise accuracy
            accuracy = (predictions == targets).sum() / torch.numel(targets)

            # Add metrics
            self.metrics["loss"].append((self.batch_idx, loss.item(), "val"))
            self.metrics["acc"].append((self.batch_idx, accuracy.item(), "val"))

        # Return model to train mode
        self.model.train()

        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
        }

    def run(self):
        # Load transforms
        transforms = self.get_transforms()

        # Load Datasets
        train_loader, val_loader = self.get_coco_loaders(transforms)

        if self.load_path != "":
            print(f"Loading checkpoint at {self.load_path}")
            # Updates model and optimizer state dicts
            # load_checkpoint(self.load_path, self.model, self.optimizer)

            # if finetune:
            #     for param in model.parameters():
            #         param.requires_grad = True
            # else:
            #     for name, param in model.named_parameters():
            #         if name.startswith("final_conv"):
            #             param.requires_grad = True
            #         else:
            #             param.requires_grad = False

        best_loss = 9999.
        for epoch in range(self.num_epochs):
            self.train_fn(train_loader, epoch)

            # Check accuracy
            results = self.validate_one_batch(val_loader)

            print("Epoch {} Validation - Loss: {:.2f}, Acc: {:.2%}".format(
                epoch+1,
                results["loss"],
                results["accuracy"],
            ))

            # If this is the best performance so far, save checkpoint
            if results["loss"] < best_loss:
                best_loss = results["loss"]

                # Save model
                checkpoint = {
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                save_checkpoint(
                    checkpoint,
                    filename=join(CHECKPOINT_PATH,
                                  f"unet1.0-coco-epoch{epoch}.pth.tar")
                )

        # Evaluate model on entire validation set
        results = self.validate_fn(val_loader)
        print("Final Validation - Loss: {}, mIoU: {}, Acc: {}".format(
                epoch,
                results["loss"],
                results["jaccard"],
                results["accuracy"],
        ))

        # Save metrics to files
        print("Saving metrics to files...")
        pd.DataFrame(
            self.metrics["loss"], 
            columns=["Batch Number", "Loss", "Mode"],
        ).to_csv(join(CHECKPOINT_PATH, "loss.csv"))
        pd.DataFrame(
            self.metrics["acc"], 
            columns=["Batch Number", "Accuracy", "Mode"],
        ).to_csv(join(CHECKPOINT_PATH, "accuracy.csv"))


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
        required=True,
        type=str,
        help="Path to train/val/test masks directory"
    )
    parser.add_argument(
        "--load",
        default="",
        type=str,
        help="Path to checkpoint (.pth.tar) if loading"
    )
    FLAGS, _ = parser.parse_known_args()

    trainer = UNetTrainer(
        image_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        load_path=CHECKPOINT_PATH,
        tensorboard_path=TENSORBOARD_PATH,
    )
    trainer.run()
