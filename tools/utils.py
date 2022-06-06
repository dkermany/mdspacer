import os
from glob import glob

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def get_filenames(path, ext="*"):
    return glob(f"{path}/*.{ext}")
