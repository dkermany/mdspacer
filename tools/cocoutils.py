from pycocotools.coco import COCO
import numpy as np
import os
import cv2
import argparse

def main():
    # Initialize the COCO api
    coco = COCO(FLAGS.json)

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
