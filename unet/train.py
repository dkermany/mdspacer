import sys
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
import argparse
from albumentations.pytorch import ToTensorV2
from os.path import abspath, join, dirname
from tqdm import tqdm
from model import UNET
from dataset import COCODataset

sys.path.append(abspath(join(dirname(__file__), "..", "tools")))
from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
)

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
N_CLASSES = 80
NUM_EPOCHS = 10
NUM_WORKERS = 0
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False

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

def one_hot_encoding(label, n_classes=80) -> torch.Tensor:
    """
    One-Hot Encoding for segmentation masks
    Example: Converts (batch, 256, 256) => (batch, n_classes, 256, 256)
    with convention of (B, C, H, W)
    """
    # nn.function.one_hot returns in CHANNEL_LAST formatting
    # permute needed to convert to CHANNEL_FIRST
    one_hot = nn.functional.one_hot(label.long(), num_classes=n_classes)
    return one_hot.permute(0,3,1,2)

def reverse_one_hot_encoding(label, axis=1) -> torch.Tensor:
    """
    Reverses One-Hot Encoding for segmentation masks
    Example: Converts (16, n_classes, 256, 256) => (16, 1, 256, 256)
    with convention of (BATCH_SIZE, C, H, W)
    """
    return torch.argmax(label, axis=axis)

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.float().to(device=DEVICE)
        targets = one_hot_encoding(targets, n_classes=N_CLASSES).float().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            print(predictions.shape, targets.shape)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    params = {
        "train": {
            "batch_size": BATCH_SIZE,
            "shuffle": True,
            "num_workers": 0
        },
        "val": {
            "batch_size": BATCH_SIZE,
            "shuffle": False,
            "num_workers": 0
        }
    }

    # Load transforms
    transforms = get_transforms()

    # Load Datasets
    train_loader, val_loader = load_coco(transforms, params)


    model = UNET(in_channels=3, out_channels=N_CLASSES).to(device=DEVICE)
    loss_fn = nn.CrossEntropyLoss() # Add class_weights after implementing
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    scaler = torch.cuda.amp.GradScaler()

    #data, targets = next(iter(train_loader))
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # Save model
        #checkpoint = {
        #    "state_dict": model.state_dict(),
        #    "optimizer": optimizer.state_dict(),
        #}
        #save_checkpoint(checkpoint)

        # Check accuracy
        #check_accuracy(val_loader, model, device=DEVICE)




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
