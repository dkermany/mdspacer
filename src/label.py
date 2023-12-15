import matplotlib
import pyclesperanto_prototype as cle
cle.select_device('RTX')
# print("Using OpenCL device " + cle.get_device().name)

import math
import numpy as np
import pandas as pd
import cv2
import tifffile
import os
import sys
import nrrd
import cupy as cp
import cc3d
import dijkstra3d
import itertools
import argparse
from dotenv import load_dotenv
from skimage.morphology import skeletonize_3d
from cupyimg.scipy.ndimage.morphology import binary_hit_or_miss
from tqdm import tqdm
from glob import glob

from oiffile import OifFile
from src.OifImageViewer import OifImageViewer
from src.kernels import get_unique_kernels, load_from_cache

def get_oib_files(path, ext="oib"):
    return sorted(glob(f"{path}/*.{ext}"))
    
def normalize(x, min, max):
    return (x - min) / (max - min)

def replace_np_values(arr: np.ndarray, map: dict) -> np.ndarray:
    fn = np.vectorize(lambda x: map.get(x, 0) * 255)
    return fn(arr)

def dict_to_pointlist(tumor_dict: dict) -> np.ndarray:
    pointlist = np.zeros((3, len(tumor_dict)))
    for idx, tumor_info in tumor_dict.items():
        x_um, y_um, z_slice = map(int, (tumor_info["x (um)"], tumor_info["y (um)"], tumor_info["z (slice)"]))
        x, y, z = map(int, (x_um / x_step, y_um / y_step, z_slice-1))
        
        pointlist[0][idx] = x
        pointlist[1][idx] = y
        pointlist[2][idx] = z
        
    return pointlist

def get_unique_orientations(cube):
    """
    List all possible unique variations of the given 3d array involving both rotation and reflection
    Inspired by @Colonel Panic at https://stackoverflow.com/questions/33190042/how-to-calculate-all-24-rotations-of-3d-array
    """
    if np.unique(cube, return_counts=True)[1][1] < 4:
        print("WARNING: <4 1s found within kernel. 4 is needed to detect branches. Ignore if using 2 for finding tips")

    variations = []

    def rotations4(cube, axes):
        """List the four rotations of the given 3d array in the plane spanned by the given axes."""
        for i in range(4):
            variations.append(np.rot90(cube, i, axes))

    for arr in [cube, np.flip(cube, axis=0), np.flip(cube, axis=1), np.flip(cube, axis=2)]:
    # for arr in [cube]:
        # imagine shape is pointing in axis 0 (up)
        # 4 rotations about axis 0
        rotations4(arr, (1,2))

        # rotate 180 about axis 1, now shape is pointing down in axis 0
        # 4 rotations about axis 0
        rotations4(np.rot90(arr, 2, axes=(0,2)), (1,2))

        # rotate 90 or 270 about axis 1, now shape is pointing in axis 2
        # 8 rotations about axis 2
        rotations4(np.rot90(arr, 1, axes=(0,2)), (0,1))
        rotations4(np.rot90(arr, -1, axes=(0,2)), (0,1))

        # rotate about axis 2, now shape is pointing in axis 1
        # 8 rotations about axis 1
        rotations4(np.rot90(arr, 1, axes=(0,1)), (0,2))
        rotations4(np.rot90(arr, -1, axes=(0,1)), (0,2))

    return np.unique(variations, axis=0)

def trim_zeros(arr):
    """Returns a trimmed view of an n-D array excluding any outer
    regions which contain only zeros.
    """
    if isinstance(arr, cp.ndarray):
        nonzero = cp.nonzero
    elif isinstance(arr, np.ndarray):
        nonzero = np.nonzero
    else:
        raise ValueError("arr needs to be np or cp ndarray type")

    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in nonzero(arr))
    return arr[slices]

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Args:
    point1 (array-like): An array-like object representing the first point.
    point2 (array-like): An array-like object representing the second point.

    Returns:
    float: The Euclidean distance between the two points.
    """
    # Use NumPy's linalg.norm function to calculate the Euclidean distance
    return np.linalg.norm(np.array(point1)-np.array(point2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image"
    )
    parser.add_argument(
        "--ROOTPATH",
        type=str,
        required=True,
        help="Path to project root dir"
    )
    parser.add_argument(
        "--BONEPATH",
        type=str,
        required=True,
        help="Path to data root dir"
    )
    FLAGS, _ = parser.parse_known_args()
    
    LIBPATH = os.path.join(FLAGS.ROOTPATH, "lib")

    filename = os.path.splitext(os.path.basename(FLAGS.image))[0]
    
    # Initialize progress bar
    pbar = tqdm(range(10), desc=filename, leave=True)

    with OifFile(FLAGS.image) as oif:
        viewer = OifImageViewer(oif)
        x_step, y_step, z_step = map(float, (viewer.md["x_step"], viewer.md["y_step"], viewer.md["z_step"]))
        if viewer.md["z_unit"] == "nm":
            z_step /= 1000.


    mask_path = os.path.join(FLAGS.BONEPATH, f"masks/{filename}.seg.nrrd")

    # Load mask from NRRD
    mask, header = nrrd.read(mask_path)
    mask = mask.T
    # mask = mask[:mask.shape[0]//3, :, :] #For FV10__20190620_111343

    # Convert mask from uint8 to uint16
    mask = mask.astype(np.uint16)
    mask *= 2**16 - 1

    tumor_csv_path = os.path.join(FLAGS.BONEPATH, "tumor_locations_02_08_2023.csv")

    # Load csv and drop rows with N/A's
    tumor_csv = pd.read_csv(tumor_csv_path).dropna()

    # Update progress bar
    pbar.update()

    # 
    # Tumor Location Marking
    # ----------------------

    # Filter tumors for the image in focus
    tumor_csv = tumor_csv[tumor_csv.Filename == filename].reset_index(drop=True)

    tumor_dict = tumor_csv.to_dict("index")

    tumor_labels = np.zeros(viewer._arr.shape[1:], dtype=np.uint16)

    for idx, tumor_info in tumor_dict.items():
        x_um, y_um, z_slice = map(int, (tumor_info["x (um)"], tumor_info["y (um)"], tumor_info["z (slice)"]))
        x, y, z = map(int, (x_um / x_step, y_um / y_step, z_slice-1))

        sphere_radius = 7 # micron
        sphere_z_slices = math.ceil(sphere_radius / z_step)

        min_z_step = max(0, z-sphere_z_slices)
        max_z_step = min(z+sphere_z_slices, viewer._arr.shape[1])

        # Label tumor
        for z_prime in range(min_z_step, max_z_step):
            cv2.circle(
                tumor_labels[z_prime],
                (x, y),
                radius=round((sphere_radius**2 - min((z_step*abs(z_prime-z))**2, sphere_radius**2))**0.5),
                color=(4096,4096,4096),
                thickness=-1,
            )

    original_tumor = cle.push(viewer.get_array()[0])
    masked_tumor = cle.mask(original_tumor, mask)

    combined_tumor_image = cle.pull(masked_tumor.copy())

    tumor_mask = tumor_labels != 0
    if tumor_labels.shape == combined_tumor_image.shape:
        combined_tumor_image[tumor_mask] = tumor_labels[tumor_mask]
    else:
        ValueError("arrays must have equal shape")

    del combined_tumor_image, original_tumor, masked_tumor

    tumor_pointlist = dict_to_pointlist(tumor_dict)

    del tumor_labels

    # Update progress bar
    pbar.update()

    # Vessel Segmentation
    # -----------------------------
    original_vessels = cle.push(viewer.get_array()[2])
    masked_vessels = cle.mask(original_vessels, mask)

    del original_vessels 

    img_gaussian = cle.gaussian_blur(masked_vessels, sigma_x=2, sigma_y=2, sigma_z=2)

    del masked_vessels

    backgrund_subtracted = cle.top_hat_box(img_gaussian, radius_x=10, radius_y=10, radius_z=5)

    thresh2 = cle.threshold_otsu(backgrund_subtracted)

    # Morphological openning
    erosion = cle.erode_box(thresh2)
    dilation = cle.dilate_box(erosion)

    del erosion, img_gaussian, backgrund_subtracted, thresh2

    # Convert dilation from uint8 to uint16
    dilation = dilation.astype(np.uint16)
    dilation *= 2**10 - 1 #2**16 - 1

    # Find Vessel Bifurcations
    # ----


    skeleton = skeletonize_3d(dilation)

    # total_image = np.concatenate((total_image, np.expand_dims(cle.pull(skeleton), axis=0)), axis=0)
    del dilation

    # Update progress bar
    pbar.update()

    kernel_cache_path = os.path.join(LIBPATH, "unique_kernels.npy")
    if os.path.exists(kernel_cache_path):
        # Load unique kernels from cache
        # print("Unique kernel cache found! Loading...")
        kernels = load_from_cache(kernel_cache_path)
        # print(f"{len(kernels)} kernels loaded!")
    else:
        # Get unique kernels
        # print("Kernel cache not found! Generating...")
        kernels = get_unique_kernels()

    # Update progress bar
    pbar.update()

    branch_pts_img = cp.zeros(skeleton.shape, dtype=int)
    cp_skeleton = cp.asarray(skeleton)
    for kernel in tqdm(kernels, disable=True):
        branch_pts_img = cp.logical_or(
            binary_hit_or_miss(cp_skeleton, structure1=cp.asarray(kernel)),
            branch_pts_img,
        )
    branch_pts_img = cp.asnumpy(branch_pts_img.astype(np.uint8) * 255)

    # 24411 old
    branch_points = np.nonzero(branch_pts_img)

    np.save(os.path.join(FLAGS.BONEPATH,
                         f"branch_points/{filename}_branch_points.npy"),
            np.array(branch_points))

    dilated_branch_pts_img = cle.dilate_box(branch_pts_img)

    subtracted_pts = cle.binary_subtract(skeleton, dilated_branch_pts_img)

    del branch_pts_img, dilated_branch_pts_img, cp_skeleton, skeleton

    # Update progress bar
    pbar.update()

    # Removes segments less than 5 pixels long
    subtracted_pts = cc3d.dust(cle.pull(subtracted_pts), threshold=5, in_place=False)
    vessel_segments2, N = cc3d.connected_components(subtracted_pts, return_N=True)

    tip_kernels = []

    # Filter 1
    tip_kernels.append(np.array([[[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]
                                 ],
                                 [[0, 0, 1],
                                  [0, 1, 0],
                                  [0, 0, 0]
                                 ],
                                 [[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]]]))

    # Filter 2
    tip_kernels.append(np.array([[[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]
                                 ],
                                 [[0, 1, 0],
                                  [0, 1, 0],
                                  [0, 0, 0]
                                 ],
                                 [[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]]]))

    # Filter 3
    tip_kernels.append(np.array([[[0, 0, 1],
                                  [0, 0, 0],
                                  [0, 0, 0]
                                 ],
                                 [[0, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 0]
                                 ],
                                 [[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]]]))

    unique_tip_kernels = []
    for kernel in tip_kernels:
        unique_tip_kernels.extend(get_unique_orientations(kernel))

    # Update progress bar
    pbar.update()

    # Remove 0 from list
    vessel_segments = cp.asarray(vessel_segments2)
    segment_labels = cp.unique(vessel_segments)[1:]

    tortuosity_lib = {}
    for label in tqdm(segment_labels):
        segment = cp.where(vessel_segments==int(label), True, False)
        cropped_segment = trim_zeros(segment)

        tips_img = cp.zeros(cropped_segment.shape, dtype=int)
        for kernel in unique_tip_kernels:
            kernel = cp.asarray(kernel)
            tips_img = cp.logical_or(binary_hit_or_miss(cropped_segment, structure1=kernel), tips_img)
        tips_img = cp.asnumpy(tips_img.astype(np.uint8) * 255)
        n_tips = tips_img.sum() / 255

        # Check that there are only 2 tips found. 
        if n_tips == 2:
            start_pt = [i[0] for i in np.nonzero(tips_img)]
            end_pt = [i[1] for i in np.nonzero(tips_img)]

        # More than 2 tips indicates that a branching point was not identified and subtracted properly.
        elif n_tips > 2:
            # print(f"Error with label: {label}")
            # print(cropped_segment.shape)
            cropped_segment = cp.asnumpy(cropped_segment)
            z, y, x = np.nonzero(tips_img)

            paths = []
            for tip_a, tip_b in itertools.combinations(zip(z, y, x), 2):
                path = dijkstra3d.binary_dijkstra(cropped_segment, source=tip_a, target=tip_b)
                paths.append((tip_a, tip_b, path))

            start_pt, end_pt, longest_path = sorted(paths, key=lambda x: x[2].shape[0], reverse=True)[0]
            cropped_segment = np.zeros(cropped_segment.shape, dtype=int)
            cropped_segment[tuple(longest_path.T)] = 1

        # Less than 2 tips means this is an unrecoverable segment. Set tortuosity to 0 so it is removed in the next step
        # and skip
        else:
            print(f"Skipping abnormal segment #{int(label)}...")
            tortuosity_lib[int(label)] = 0
            continue

        euclid_dist = euclidean_distance(start_pt, end_pt)
        geodesic_dist = cropped_segment.sum()
        tortuosity = 1. * geodesic_dist / euclid_dist

        # tortuosity_lib[int(label)] = min(float(tortuosity), 3.)
        tortuosity_lib[int(label)] = float(tortuosity)

    # print("Finished calculating tortuosity for each segment")
    # print("Now replacing label with tortuosity value in full image")

    # Update progress bar
    pbar.update()


    # Removes zeros (0)
    tortuosity_lib = {k: v for k, v in tortuosity_lib.items() if v > 0}


    # Remove top n from dictionary
    n = 25
    sorted_t_items = sorted(tortuosity_lib.items(), reverse=True, key=lambda x: x[1])
    t_key = {k: v for i, (k, v) in enumerate(sorted_t_items) if i > n}

    t_values = np.array(list(t_key.values()))
    t_std = np.std(t_values)
    t_mean = np.mean(t_values)
    # print(f"Tortuosity Mean: {t_mean:.4f} +/- {t_std:.4f}")

    t_threshold = t_mean + 2*t_std
    # print(f"Tortuosity Threshold set to {t_threshold}")

    t_min, t_max = np.min(t_values), np.max(t_values)

    del subtracted_pts

    del segment

    normalized_map = {k: normalize(v, t_min, t_max) for k, v in t_key.items()}
    result_array = replace_np_values(cp.asnumpy(vessel_segments), normalized_map)

    result_values = result_array[result_array>0].flatten()
    result_std = np.std(result_values)
    result_mean = np.mean(result_values)
    # print(f"Tortuosity Mean: {result_mean:.4f} +/- {result_std:.4f}")

    result_threshold = result_mean + 3*result_std
    # print(f"Tortuosity Threshold set to {t_threshold}")

    result_min, result_max = np.min(result_values), np.max(result_values)
    # print(result_min, result_max)

    bin_result_array = result_array.copy()
    low_thresh_idx = np.where(np.logical_and(bin_result_array > 0,
                                             bin_result_array <= result_threshold))
    high_thresh_idx = np.where(np.logical_and(bin_result_array > 0,
                                              bin_result_array > result_threshold))
    bin_result_array[low_thresh_idx] = 55
    bin_result_array[high_thresh_idx] = 255

    # extract centroids from high tortuosity segments as pointlist
    bin_result_array[bin_result_array < 255] = 0
    labeled_tortuous_segments = cle.connected_components_labeling_box(cle.push(bin_result_array))
    tortuous_segments_pointlist = cle.centroids_of_labels(labeled_tortuous_segments)

    # Update progress bar
    pbar.update()

    np.save(os.path.join(FLAGS.BONEPATH,
                         f"tortuous_segment_centroids/{filename}_tortuous_segment_centroid.npy"),
            np.array(tortuous_segments_pointlist))
    del bin_result_array

    original_ng2 = cle.push(viewer.get_array()[1])

    masked_ng2 = cle.mask(original_ng2, mask)

    del original_ng2

    img_gaussian = cle.gaussian_blur(masked_ng2, sigma_x=2, sigma_y=2, sigma_z=2)

    del masked_ng2

    backgrund_subtracted = cle.top_hat_box(img_gaussian, radius_x=10, radius_y=10, radius_z=5)

    thresh1 = np.where(backgrund_subtracted > 1200, 65535, 0)

    # Morphological opening
    kernel = np.ones((5,5), np.uint8)
    erosion = cle.erode_box(thresh1) #thresh2
    dilation = cle.dilate_box(erosion)

    del erosion, thresh1, backgrund_subtracted, img_gaussian

    labels = cle.connected_components_labeling_box(dilation)

    smalls_filtered_out = cle.exclude_labels_outside_size_range(labels, None, 400, 10000)

    a = cle.centroids_of_labels(smalls_filtered_out)

    np.save(os.path.join(FLAGS.BONEPATH,
                         f"NG2_Centroids/{filename}_NG2_centroids.npy"),
            np.array(a))

    # Update progress bar
    pbar.update()
