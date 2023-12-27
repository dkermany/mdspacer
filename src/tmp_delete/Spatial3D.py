import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import nrrd
import os
import argparse
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

def _load_points_from_csv(csv_path):
    csv = pd.read_csv(csv_path).dropna().reset_index(drop=True)
    points_dict = csv.to_dict("index")
    points = []
    for idx, point in points_dict.items():
        x, y, z = map(round, map(float, (point["x"], point["y"], point["z"])))
        points.append([z, y, x])
    return np.array(points, dtype=np.float64)

def _load_points_from_npy(npy_path):
    points = np.flip(np.load(npy_path).T, axis=1) # put points into Z,Y,X format (N, 3)
    assert len(points.shape) == 2 and points.shape[1] == 3
    return points

def load_points(path):
    ext = os.path.splitext(path)[1]
    if ext == ".csv":
        return _load_points_from_csv(path)
    elif ext == ".npy":
        return _load_points_from_npy(path)
    raise NotImplementedError("Only CSV and NPY formats for loading points are currently implemented")

def main():
    mask = load_mask(FLAGS.mask)

    points_a = load_points(FLAGS.points_a)
    points_b = load_points(FLAGS.points_b)
    radii = np.arange(2, FLAGS.max_radius)

    results_w = []
    ripley_w = CrossRipley(points_a, points_b, radii, mask, boundary_correction=True)
    # run ripley algorithm with boundary correction and sort by radii
    K_w, L_w, H_w = map(sorted, ripley_w.run_ripley(N_PROCESSES))
    # Organize into list of [Radius, K, L, H, Type] 
    results_w += [(k[0], k[1], l[1], h[1], "multivariate") for k, l, h in zip(K_w, L_w, H_w)]
    rstats_w = pd.DataFrame(results_w, columns=["Radius (r)", "K(r)", "L(r)", "H(r)", "Type"])
    rstats_w.to_csv(f"/home/dkermany/ripley_results/{filename}_rstats_w.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--points_a",
        type=str,
        required=True,
        description="(.csv or .npy) Path to the first pointset to use as anchor points in multivariate Ripley K calculation"
    )
    parser.add_argument(
        "--points_b",
        type=str,
        required=True,
        description="(.csv or .npy) Path to the second pointset to use in multivariate Ripley K calculation"
    )
    parser.add_argument(
        "--mask",
        type=str,
        required=False,
        default="",
        # TODO add functionality for tif masks
        description="(.seg.nrrd) Path to binary mask of 3D image. Required for boundary correction and weight calculation"
    )
    parser.add_argument(
        "--max_radius",
        type=int,
        required=False,
        default=100,
        description="Maximum search radius for Ripley K calculation. Radii from 2 to max_radius utilized: (2, 3, 4, ..., max_radius)"
    )
    FLAGS, _ = parser.parse_known_args()

    N_PROCESSES = 63
    main()