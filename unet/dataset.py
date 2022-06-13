import os
import sys
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
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
        assert (os.path.splitext(self.image_filenames[index])[0] == 
                os.path.splitext(self.mask_filenames[index])[0])
        
        image_path = os.path.join(self.image_dir, self.image_filenames[index])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[index])

        image = cv2.imread(image_path).astype(np.float32)
        mask = cv2.imread(mask_path).astype(np.float32)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image, mask = augmentations["image"], augmentations["mask"]
        
        # print(f"Image - shape: {image.shape} range: {np.min(image)}-{np.max(image)} dtype: {image.dtype}")
        # print(f"Mask - shape: {mask.shape} range: {np.min(mask)}-{np.max(mask)} dtype: {mask.dtype}")
        return image, mask

    def get_dataset_mean_and_std(self, batch_size=4):
        """
        Mean and STD over batch, height, and width, but not over the channels
        std = sqrt(E[X^2] - (E[X])^2)
        https://towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c
        """
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0
        loader = DataLoader(self, batch_size=batch_size, shuffle=False)
        for data, _ in tqdm(loader):
            # Final batch size may not be equal to batch_size, therefore batches 
            # are weighted by size 
            current_batch_size = data.size()[0]
            batch_weight = float(current_batch_size) / loader.batch_size

            channels_sum += batch_weight * torch.mean(data, dim=[0,2,3])
            channels_squared_sum += batch_weight * torch.mean(data**2, dim=[0,2,3])

            num_batches += batch_weight

        mean = channels_sum / num_batches
        std = ((channels_squared_sum / num_batches) - (mean ** 2)) ** 0.5

        return mean, std


if __name__ == "__main__":
    # Test
    image_dir = "D:\\Datasets\\Segmentation\\COCO\\val2017"
    mask_dir = "D:\\Datasets\\Segmentation\\COCO\\masks\\val2017"
    dataset = COCODataset(image_dir, mask_dir)
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    # next(iter(dataloader))
    print(dataset.get_dataset_mean_and_std())
    