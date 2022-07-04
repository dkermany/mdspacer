import sys
import cv2
import numpy as np
import torch
import albumentations as A
from os.path import join, splitext, basename, normpath
from scipy.io import loadmat
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from glob import glob

"""
Masks are assumed to have pixel values corresponding to category/class id
"""

def get_filenames(
    path: str,
    ext: str = "*",
    fullpath: bool = True
) -> list[str]:
    """
    Returns all files with given extension `ext` (without the period) at the
    given directory `path` in a sorted list of absolute paths or as basenames
    """
    filenames = glob(join(path, f"*.{ext}"))
    if len(filenames) == 0:
        err_msg = f"No files with extension '{ext}' found at path: '{path}'"
        raise ValueError(err_msg)
    if fullpath:
        return sorted(filenames)
    return sorted([basename(normpath(f)) for f in filenames])


class BaseDataset(Dataset):
    """
    Base Dataset class
    """
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_ext: str = "png",
        mask_ext: str = "png",
        transform: A.Compose = None
    ):
        """
        Initialize instance variables
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_ext = image_ext
        self.mask_ext = mask_ext
        self.transform = transform

        self.image_filenames = get_filenames(
            self.image_dir,
            ext=self.image_ext,
            fullpath=False,
        )
        self.mask_filenames = [
            f"{splitext(i)[0]}.{self.mask_ext}" for i in self.image_filenames
        ]

        if not self.image_filenames:
            err = "No images with ext '{self.image_ext}' at '{self.image_dir}'"
            raise ValueError(err)
        if not self.mask_filenames:
            err = "No masks with ext '{self.mask_ext}' at '{self.mask_dir}'"
            raise ValueError(err)

    def __len__(self) -> int:
        """
        Returns length of the discovered image filenames
        """
        return len(self.image_filenames)

    def get_class_weights(self, dataloader: DataLoader[object]):
        """
        Returns Tensor of class weights
        """
        raise NotImplementedError

    def get_dataset_mean_and_std(
        self, dataloader: DataLoader[object]
    ) -> tuple[torch.Tensor]:
        """
        Mean and STD over batch, height, and width, but not over the channels
        std = sqrt(E[X^2] - (E[X])^2)
        https://towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c
        Only need to run once to determine mean & std for a dataset of specific
        size
        """
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0
        for data, _ in tqdm(dataloader):
            # Final batch size may not be equal to batch_size, therefore batches 
            # are weighted by size 
            current_batch_size = data.size()[0]
            batch_weight = float(current_batch_size) / dataloader.batch_size

            # mean over batch, height, and width but not over channels
            channels_sum += batch_weight * torch.mean(data, dim=[0,2,3])
            channels_squared_sum += batch_weight * torch.mean(data**2, dim=[0,2,3])

            num_batches += batch_weight

        mean = channels_sum / num_batches
        std = ((channels_squared_sum / num_batches) - (mean ** 2)) ** 0.5

        return mean, std

class CoNSePDataset(BaseDataset):
    def __getitem__(self, index: int):
        """
        returns image (torch.Size(3, 256, 256)) and mask (torch.Size(256, 256))
        """
        if (splitext(self.image_filenames[index])[0] !=
                splitext(self.mask_filenames[index])[0]):
            raise ValueError("Image and Mask names are not identical")

        image_path = join(self.image_dir, self.image_filenames[index])
        mask_path = join(self.mask_dir, self.mask_filenames[index])

        image = Tensor(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))

        # Load in mask info from .mat file as numpy array
        x = loadmat(mask_path)

        # Add inflammatory classes (1, 2) in 0th layer
        mask = Tensor((x['type_map']==1).astype(int) +\
                      (x['type_map']==2).astype(int))
        mask = mask[:, :, None]

        # Add epithelial classes (3, 4) in 1th layer
        temp = Tensor((x['type_map']==3).astype(int) +\
                      (x['type_map']==4).astype(int))[:, :, None]
        mask = torch.concat((mask,temp), axis=2)

        # Add spindle-shaped classes (5, 6, 7) in 2nd layer
        temp = Tensor((x['type_map']==5).astype(int) +\
                      (x['type_map']==6).astype(int) +\
                      (x['type_map']==7).astype(int))[:, :, None]
        mask = torch.concat((mask,temp), axis=2)

        mask = mask.float()
        mask[mask >= 1] = 1

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image, mask = augmentations["image"], augmentations["mask"]

        # convert from channel_last format to channel_first
        return image.permute(2,0,1), mask.permute(2,0,1)

class COCODataset(BaseDataset):
    def __getitem__(self, index: int):
        """
        returns image (torch.Size(3, 256, 256)) and mask (torch.Size(256, 256))
        """
        if (splitext(self.image_filenames[index])[0] ==
                splitext(self.mask_filenames[index])[0]):
            raise ValueError("Image and Mask names are not identical")

        image_path = join(self.image_dir, self.image_filenames[index])
        mask_path = join(self.mask_dir, self.mask_filenames[index])

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image, mask = augmentations["image"], augmentations["mask"]

        return image, mask
