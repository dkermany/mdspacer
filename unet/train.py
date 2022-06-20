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

def train(loader, model, optimizer, loss_fn, scaler):
    for batch_idx, (data, targets) in enumerate(tqdm(loader)):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)


def main():
    params = {
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "num_workers": 1
    }
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
                mean=[103.6788, 113.7296, 119.6984],
                std=[72.6042, 68.8787, 70.1635],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )
    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[103.6788, 113.7296, 119.6984],
                std=[72.6042, 68.8787, 70.1635],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )

    cocodata = COCODataset(image_path, mask_path, transform=transform)
    cocoloader = DataLoader(cocodata, **params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images",
        required=True,
        type=str,
        help="Path to train/val/test directory"
    )
    FLAGS, _ = parser.parse_known_args()
    main()
