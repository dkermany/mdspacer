import os
from glob import glob

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def get_filenames(path, ext="*", basename=False):
    filenames = glob(os.path.join(path, f"*.{ext}"))
    if not basename:
        return sorted(filenames)
    return sorted([os.path.basename(os.path.normpath(f)) for f in filenames])
