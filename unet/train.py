import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import albumentations as A
import argparse
from time import time
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import JaccardIndex, Accuracy
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
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if torch.cuda.is_available() else False
BATCH_SIZE = 100
N_CLASSES = 80 + 1 # Background class
NUM_EPOCHS = 25 
NUM_WORKERS = 4
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
LOAD_MODEL = False
CHECKPOINT_PATH = "/home/dkermany/BoneSegmentation/checkpoints"
TENSORBOARD_PATH = "/home/dkermany/BoneSegmentation/tensorboard/COCO"
create_directory(CHECKPOINT_PATH)

def get_transforms():
    train_transform = A.Compose(
        [
            A.Resize(height=int(IMAGE_HEIGHT*1.13),
                     width=int(IMAGE_WIDTH*1.13)),
            A.RandomCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
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
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.4690, 0.4456, 0.4062],
                std=[0.2752, 0.2701, 0.2847],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )
    return {"train": train_transform, "val": val_transform}

def load_coco(transforms, params):
    train_ds = COCODataset(
        join(FLAGS.images, "train2017"),
        join(FLAGS.masks, "train2017"),
        transform=transforms["train"]
    )
    train_loader = torch.utils.data.DataLoader(train_ds, **params["train"])

    val_ds = COCODataset(
        join(FLAGS.images, "val2017"),
        join(FLAGS.masks, "val2017"),
        transform=transforms["val"]
    )
    val_loader = torch.utils.data.DataLoader(val_ds, **params["val"])

    return train_loader, val_loader

def one_hot_encoding(label) -> torch.Tensor:
    """
    One-Hot Encoding for segmentation masks
    Example: Converts (batch, 256, 256) => (batch, n_classes, 256, 256)
    with convention of (B, C, H, W)
    """
    # nn.function.one_hot returns in CHANNEL_LAST formatting
    # permute needed to convert to CHANNEL_FIRST
    one_hot = nn.functional.one_hot(label.long(), num_classes=N_CLASSES)
    return one_hot.permute(0,3,1,2)

def reverse_one_hot_encoding(label, axis=1) -> torch.Tensor:
    """
    Reverses One-Hot Encoding for segmentation masks
    Example: Converts (16, n_classes, 256, 256) => (16, 1, 256, 256)
    with convention of (BATCH_SIZE, C, H, W)
    """
    return torch.argmax(label, axis=axis)

def train_fn(loader, model, optimizer, loss_fn, scaler, writer, epoch, step):
    eval = False
    loop = tqdm(loader)
    for data, targets in loop:
        data = data.float().to(device=DEVICE)
        targets = one_hot_encoding(targets).float().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            logits = model(data)
            loss = loss_fn(logits, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Updates JaccardIndex with batch
        targets = torch.argmax(targets, axis=1).int()
        predictions = torch.argmax(logits, dim=1)

        if eval:
            j = torchmetrics.functional.jaccard_index(
                predictions,
                targets,
                num_classes=N_CLASSES,
                average="micro",
                ignore_index=0,
            )
            writer.add_scalar("Training IoU", j.item(), global_step=step)

        # # Added metrics to Tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

        # update tqdm loop
        loop.set_description(f"Epoch: {epoch}/{NUM_EPOCHS} - Loss: {loss.item()}")

def validate_fn(loader, model, loss_fn, writer, step):
    # TODO: Class-Weighted Pixel-Wise Accuracy Metric
    # TODO: Class-Weighted Multiclass Dice Coefficient
    # TODO: Add Jaccard to Tensorboard (error when using jaccard(x,y) compared
    # to when using jaccard.update(x,y)
    model.eval()
    with torch.no_grad():

        # Class-Weighted mIoU Score
        jaccard = JaccardIndex(
            num_classes=N_CLASSES,
            average="micro",
            ignore_index=0,
            mdmc_average="global",
        ).to(device=DEVICE)

        for data, targets in tqdm(loader):
            data = data.float().to(device=DEVICE)
            targets = one_hot_encoding(targets).float().to(device=DEVICE)
            with torch.cuda.amp.autocast():
                logits = torch.softmax(model(data), dim=1)
                loss = loss_fn(logits, targets)

            # Updates JaccardIndex with batch
            targets = torch.argmax(targets, axis=1).int()
            predictions = torch.argmax(logits, dim=1)
            j = jaccard(predictions, targets)

            # Added metrics to Tensorboard
            writer.add_scalar("Validation loss", loss, global_step=step)
            writer.add_scalar("Validation IoU", j.item(), global_step=step)


        # Average over all batches uisng mdmc_average method
        total_jaccard = jaccard.compute()

    model.train()
    return {
        "jaccard": total_jaccard,
    }

def main():
    params = {
        "train": {
            "batch_size": BATCH_SIZE,
            "shuffle": True,
            "num_workers": NUM_WORKERS,
            "pin_memory": PIN_MEMORY,
        },
        "val": {
            "batch_size": BATCH_SIZE,
            "shuffle": False,
            "num_workers": NUM_WORKERS,
            "pin_memory": PIN_MEMORY,
        }
    }

    # Load transforms
    transforms = get_transforms()

    # Load Datasets
    train_loader, val_loader = load_coco(transforms, params)


    model = UNET(in_channels=3, out_channels=N_CLASSES)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs...")
        model = nn.DataParallel(model)


    model.to(device=DEVICE)
    loss_fn = nn.CrossEntropyLoss() # Add class_weights after implementing
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(TENSORBOARD_PATH)
    step = 0
    best_jaccard = 0.
    for epoch in range(NUM_EPOCHS):
        train_fn(
            train_loader,
            model,
            optimizer,
            loss_fn,
            scaler,
            writer,
            epoch,
            step,
        )


        # Check accuracy
        results = validate_fn(val_loader, model, loss_fn, writer, step)
        print(f"Epoch{epoch} Validation Jaccard: {results['jaccard']}")

        # If this is the best performance so far, save checkpoint
        if results["jaccard"].item() > best_jaccard:
            best_jaccard = results["jaccard"].item()

            # Save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(
                checkpoint,
                filename=join(CHECKPOINT_PATH, f"unet1.0-coco-epoch{epoch}.pth.tar")
            )

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
    FLAGS, _ = parser.parse_known_args()
    main()
