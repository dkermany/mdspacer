import os
import sys
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod
from tqdm import tqdm
from glob import glob
from sklearn.metrics import (
    jaccard_score, accuracy_score, precision_score, recall_score
)
from unet.dataset import InferenceDataset, COCODataset, CoNSePDataset

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def BGR2RGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_pred_f = torch.flatten(y_pred)
    y_true_f = torch.flatten(y_true)

    intersection = (y_pred_f * y_true_f).sum()
    y_sum = y_pred_f.sum() + y_true_f.sum() 

    return (2. * intersection + smooth) / (y_sum + smooth)

def dice_coef_multiclass(y_true, y_pred, n_classes=80):
    dice = 0.
    print(torch.unique(y_true).shape)
    #for i in range(n_classes):

    #    weight = (torch.numel(y_true) / n_classes) / torch.numel(y_true[y_true==i])
    #    print(y_true.shape, torch.unique(y_true))
    #    print(f"{i}/{n_classes} - {torch.numel(y_true[y_true==i])}/{torch.numel(y_true)}")
    #    dice += weight * dice_coef(y_true[i,:,:], y_pred[i,:,:])
    #return dice / n_classes
    return 1

def pixelwise_accuracy(y, y_pred):
    return (y_pred == y).sum() / y.shape[1]



def get_COCO_transforms(image_size: int) -> dict[str, A.Compose]:
    train_transform = A.Compose(
        [
            A.Resize(
                height=int(image_size*1.13),
                width=int(image_size*1.13),
            ),
            A.RandomCrop(height=image_size, width=image_size),
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
            A.Resize(height=image_size, width=image_size),
            A.Normalize(
                mean=[0.4690, 0.4456, 0.4062],
                std=[0.2752, 0.2701, 0.2847],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )
    return {
        "train": train_transform,
        "val": val_transform,
        "inference": val_transform, # use val_transform for inference
    }

def get_inference_loader(
    image_path: str,
    image_ext: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    transform: A.Compose
) -> tuple[Dataset, DataLoader[object]]:
    """
    Loads dataset and dataloaders for inference image set. Unlike training
    dataloaders, inference dataloaders do not load or return a mask array.
    Additionally inference dataloaders do not assume subdirectories, such
    as "train" or "val", but assume that inference images are found
    immediately in the provided directory path

    Arguments:
        - transform (A.Compose): albumentations transform to perform on
                                 inference set. Typically limited to
                                 resizing and normalization for inference
                                 and evaluation datasets

    Returns:
        - tuple[Dataset, DataLoader]
            - Dataset: PyTorch Dataset of inference images
            - DataLoader[object]: PyTorch DataLoader containing images in
                                  provided path
    """
    inference_ds = InferenceDataset(
        image_path,
        image_ext,
        transform=transform,
    )
    inference_loader = DataLoader(
        inference_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return inference_ds, inference_loader

def get_CoNSeP_loaders(
    image_path: str,
    mask_path: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    transforms: dict[str, A.Compose],
    image_ext: str = "png",
    mask_ext: str = "png",
) -> tuple[DataLoader[object]]:
    train_ds = CoNSePDataset(
        os.path.join(image_path, "train2017"),
        os.path.join(mask_path, "train2017"),
        image_ext=image_ext,
        mask_ext=mask_ext,
        transform=transforms["train"]
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_ds = CoNSePDataset(
        os.path.join(image_path, "val2017"),
        os.path.join(mask_path, "val2017"),
        image_ext=image_ext,
        mask_ext=mask_ext,
        transform=transforms["val"]
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader

def get_COCO_loaders(
    image_path: str,
    mask_path: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    transforms: dict[str, A.Compose],
    image_ext: str = "jpg",
    mask_ext: str = "png",
) -> tuple[DataLoader[object]]:

    train_ds = COCODataset(
        os.path.join(image_path, "train2017"),
        os.path.join(mask_path, "train2017"),
        image_ext=image_ext,
        mask_ext=mask_ext,
        transform=transforms["train"],
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_ds = COCODataset(
        os.path.join(image_path, "val2017"),
        os.path.join(mask_path, "val2017"),
        image_ext=image_ext,
        mask_ext=mask_ext,
        transform=transforms["val"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader



