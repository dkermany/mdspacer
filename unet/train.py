import sys
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
import argparse
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from os.path import abspath, join, dirname
from tqdm import tqdm
from model import UNET
from dataset import COCODataset

sys.path.append(abspath(join(dirname(__file__), "..", "tools")))
# from utils import (
#     load_checkpoint,
#     save_checkpoint,
#     get_loaders,
#     check_accuracy,
#     save_predictions_as_imgs
# )

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False

def get_transforms():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT*1.13, width=IMAGE_WIDTH*1.13),
            A.RandomCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
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
    return train_transform, val_transform

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
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
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "num_workers": 0
    }

    train_transform, val_transform = get_transforms()
    train_cocodata = COCODataset(join(FLAGS.images, "train2017"),
                                 join(FLAGS.mask, "train2017"),
                                 transform=train_transform)
    train_cocoloader = DataLoader(train_cocodata, **params)

    val_cocodata = COCODataset(join(FLAGS.images, "val2017"),
                               join(FLAGS.mask, "val2017"),
                               transform=val_transform)
    val_cocoloader = DataLoader(val_cocodata, **params)

    model = UNET(in_channels=3, out_channels=80)
    loss_fn = nn.CrossEntropyLoss() # Add class_weights after implementing
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # Save model
        # Check accuracy




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
