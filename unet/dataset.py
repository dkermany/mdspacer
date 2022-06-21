import os
import sys
import cv2
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools"))
)
from utils import get_filenames

class COCODataset(Dataset):
    """
        Masks are assumed to have pixel values corresponding to category/class id
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_filenames = get_filenames(
                self.image_dir,
                ext="jpg",
                basename=True
        )
        self.mask_filenames = [
                f"{os.path.splitext(i)[0]}.png" for i in self.image_filenames
        ]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        """
        returns image (torch.Size(3, 256, 256)) and mask (torch.Size(256, 256))
        """
        assert (os.path.splitext(self.image_filenames[index])[0] ==
                os.path.splitext(self.mask_filenames[index])[0])

        image_path = os.path.join(self.image_dir, self.image_filenames[index])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[index])

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image, mask = augmentations["image"], augmentations["mask"]

        #print(f"Image - shape: {image.shape} range: {np.min(image)}-{np.max(image)} dtype: {image.dtype}")
        #print(f"Mask - shape: {mask.shape} range: {np.min(mask)}-{np.max(mask)} dtype: {mask.dtype}")
        #print(f"Mask shape: {mask.shape}\nImage shape: {image.shape}")
        return image, mask

    def get_class_weights(self, dataloader):
        """
        Returns Tensor of class weights
        TODO: Implement version of this that does not crash!
        """
        y = torch.Tensor()
        for i, (_, y_batch) in tqdm(enumerate(dataloader)):
            y = torch.cat((y, torch.flatten(y_batch)))
        return compute_class_weight("balanced", classes=torch.unique(y), y=y)

    def get_dataset_mean_and_std(self, dataloader):
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

