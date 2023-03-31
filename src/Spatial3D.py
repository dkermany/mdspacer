import numpy as np
import time
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import nrrd
import multiprocessing as mp
import raster_geometry as rg
import sys
import os
from scipy import spatial, stats
from functools import reduce
from tqdm import tqdm
from glob import glob
from oiffile import OifFile
from src.OifImageViewer import OifImageViewer
from src.Ripley import Ripley, CrossRipley

def get_image_filepaths(path, ext="oib"):
    filenames = sorted(glob(f"{path}/*.{ext}"))
    if len(filenames) == 0:
        raise ValueError(f"ValueError: no files of type .{ext} found in {path}")

# TODO add support for units other than microns
def load_oib(oib_path):
    """
    Given path to .OIB file, loads into OifImageViewer and returns ...
    Currently assumes all units are given in microns
    oib_path (str): path to single .OIB file
    """
    with OifFile(oib_path) as oif:
        viewer = OifImageViewer(oif)
        x_step, y_step, z_step = viewer.md["x_step"], viewer.md["y_step"], viewer.md["z_step"]
        if viewer.md["z_unit"] == "nm":
            z_step /= 1000.
    return viewer, x_step, y_step, z_step

def load_mask(mask_path):
    mask, header = nrrd.read(mask_path)
    return mask.T

def get_filename(filepath):
    return os.path.splitext(os.path.basename(os.path.normpath(filepath)))[0]

# TODO: Add functionality to load in tiffs
def process_path(path, ext="oib"):
    """
    path (str): path to folder containing 
    """    
    path = os.path.normpath(path)
    if os.path.isfile(path):
        return [path]
    else:
        return get_image_filepaths(path, ext=ext)

def load_points_from_csv(csv_path, filename, x_step, y_step):
    csv = pd.read_csv(csv_path).dropna()
    csv = csv[csv.Filename == filename].reset_index(drop=True)
    points_dict = csv.to_dict("index")
    points = []
    for idx, tumor_info in points_dict.items():
        x_um, y_um, z_slice = map(int, (tumor_info["x (um)"], tumor_info["y (um)"], tumor_info["z (slice)"]))
        x, y, z = map(int, (x_um / x_step, y_um / y_step, z_slice-1))
        points.append([z, y, x])
    return np.array(points, dtype=np.float64)

def load_points_from_npy(npy_path):
    return np.flip(np.load(npy_path).T, axis=1) # put points into Z,Y,X format (N, 3)


def main(image_path, mask_path, csv_path):
    filepaths = process_path(image_path)
    cache = mp.Manager().dict()
    for filepath in filepaths:
        filename = get_filename(filepath)
        viewer, x_step, y_step, z_step = load_oib(filepath)
        mask = load_mask(mask_path)

        points_a = load_points_from_csv(csv_path)
        points_b = load_points_from_npy(npy_path)
        radii = np.arange(2, 100)

        results_w = []
        ripley_w = CrossRipley(points_a, points_b, radii, mask, boundary_correction=True)
        # Sort by radii
        K_w, L_w, H_w = map(sorted, ripley_w.run_ripley(N_PROCESSES))
        # Organize into list of [Radius, K, L, H, Type] 
        results_w += [(k[0], k[1], l[1], h[1], "multivariate") for k, l, h in zip(K_w, L_w, H_w)]
        rstats_w = pd.DataFrame(results_w, columns=["Radius (r)", "K(r)", "L(r)", "H(r)", "Type"])
        rstats_w.to_csv(f"/home/dkermany/ripley_results/{filename}_rstats_w.csv")


if __name__ == "__main__":
    image_path = ""
    mask_path = ""
    csv_path = "" # tumor_csv
    npy_path = "" # ng2 points
    N_PROCESSES = 63
    main(image_path, mask_path, csv_path, npy_path)