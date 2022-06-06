"""
 Usage: 
 python cocoutils.py
        --images /home/dkermany/data/COCO/val2017/ 
        --json /home/dkermany/data/COCO/annotations/instances_val2017.json
"""

from pycocotools.coco import COCO
from utils import create_directory
from tqdm import tqdm
import os
import numpy as np
import cv2
import argparse

def main():
    # Initialize the COCO api
    coco = COCO(FLAGS.json)

    # Category IDs:
    category_ids = coco.getCatIds()
    categories = coco.loadCats(category_ids)
    category_names = [category["name"] for category in categories]
    
    # Get all image IDs
    image_ids = coco.getImgIds()

    # Loop through each image ID
    for image_id in tqdm(image_ids):
        # Get image
        image_info = coco.loadImgs([image_id])[0]
        image_filename = image_info["file_name"]
        image_path = os.path.join(FLAGS.images, image_filename)
        image = cv2.imread(image_path)
        
        # Get annotations for image
        annotation_ids = coco.getAnnIds(imgIds=[image_id])
        annotations = coco.loadAnns(annotation_ids)

        # Create label mask
        height, width = image.shape[0], image.shape[1]
        mask = np.zeros((height, width))
        for annotation in annotations:
            mask = np.maximum(
                    mask,
                    coco.annToMask(annotation)*annotation["category_id"]
            )

        # Save segmentation mask
        output_path = os.path.join(
                os.path.dirname(os.path.normpath(FLAGS.images)), 
                "masks",
                f"{os.path.splitext(image_filename)[0]}.png"
        )
        create_directory(os.path.dirname(output_path))
        cv2.imwrite(output_path, mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to dataset"
    )
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to COCO JSON annotations"
    )
    
    FLAGS, _ = parser.parse_known_args()
    main()
