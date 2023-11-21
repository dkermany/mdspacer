import pyclesperanto_prototype as cle
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import nrrd
import numpy as np
import cupy as cp
import cc3d
import dijkstra3d
import matplotlib.pyplot as plt

from cupyimg.scipy.ndimage.morphology import binary_hit_or_miss
from skimage.morphology import skeletonize_3d

from tqdm import tqdm
from oiffile import OifFile
from src.OifImageViewer import OifImageViewer
from src.kernels import get_unique_kernels, get_unique_tip_kernels, 
                        load_from_cache
from src.utils import trim_zeros, euclidean_distance, normalize,
                      replace_np_values, create_directory


def main():
    filename = os.path.splitext(os.path.basename(FLAGS.image_path))[0]
    mask = load_mask(os.path.join(FLAGS.mask_path, f"{filename}.seg.nrrd"))

    # Get vessel channel
    vessels = load_oib(FLAGS.image_path)[2]
    
    # get vessel segmentation of image
    vessel_seg = get_vessel_segmentation(vessels, mask)
    
    # skeletonize binary 3d vessel segmentation
    skeleton = cp.asarray(skeletonize_3d(vessel_seg))
    del vessel_seg

    # match branch points
    branch_pts, branch_pts_img = get_branch_pts(skeleton)
    output_path = os.path.join(FLAGS.output_path,
                               f"{filename}_branch_points.npy")

    # save branch points to file
    save_branch_pts(output_path, branch_pts)

    tortuous_centroids = get_tortuous_vessels(branch_pts_img,
                                              skeleton,
                                              filename)


def get_tortuous_vessels(branch_pts_img, skeleton, filename):
    # Dilate branch points
    dilated_pts = cle.dilate_box(branch_pts_img)
    del branch_pts_img

    # Subtract branch points from skeleton
    pts_subtracted = cle.binary_subtract(skeleton, dilated_pts)
    del dilated_pts, skeleton

    # Remove segments less than 5 pixels long
    pts_subtracted = cc3d.dust(cle.pull(pts_subtracted),
                               threshold=5,
                               in_place=False)

    # Group connected components
    vessel_segments, N = cc3d.connected_components(pts_subtracted,
                                                   return_N=True)
    del pts_subtracted
    
    vessel_segments = cp,asarray(vessel_segments)
    segment_labels = cp.unique(vessel_segments)
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

    print("Finished calculating tortuosity for each segment")
    print("Now replacing label with tortuosity value in full image")

    # Remove zeros
    tortuosity_lib = {k: v for k, v in tortuosity_lib.items() if v > 0}

    # Remove top n from dictionary
    n = 25
    sorted_t_items = sorted(tortuosity_lib.items(), reverse=True,
                            key=lambda x: x[1])
    t_key = {k: v for i, (k, v) in enumerate(sorted_t_items) if i > n}
    t_values = np.array(list(t_key.values()))

    # Calculate mean, std, an threshold based on
    std = np.std(t_values)
    mean = np.mean(t_values)
    
    threshold = mean + (3 * std)
    
    # plot tortuosity histogram and save as PNG
    plot_hist(t_values, threshold, filename)

    # Normalize tortuosity
    normalized_tort = {k: normalize(v, t_min, t_max)
                       for k, v in t_key.items()}

    # Replace segment pixel values with segment tortuosity value
    result_array = replace_np_values(cp.asnumpy(vessel_segments),
                                     normalized_map)

    # Create binarize tortuosity map with threshold
    bin_result_array = result_array.copy()
    low_thresh_idx = np.where(np.logical_and(bin_result_array > 0,
                                             bin_result_array <= result_threshold))
    high_thresh_idx = np.where(np.logical_and(bin_result_array > 0,
                                              bin_result_array > result_threshold))
    bin_result_array[low_thresh_idx] = 55
    bin_result_array[high_thresh_idx] = 255
   
    # TODO Write functions to plot and save tortuosity heatmap + binary


def plot_heatmap():
    pass

def plot_binary_tortuosity():
    pass

def plot_hist(values, threshold, filename):
    
    plt.figure(figsize=(5,5))
    
    idx = values > threshold
    plt.axvline(x=threshold, linestyle="dotted", linewidth=1.25,
                color="grey")
    plt.hist([values[~idx], values[idx]], color=["b", "r"], bins=50,
             width=7)

    output_path = os.path.join(FLAGS.output_path,
                               "tortuosity",
                               f"{filename}_seg_tort.png")
    create_directory(output_path)
    plt.savefig(output_path, bbox_inches="tight")    


def save_branch_pts(path, branch_pts):
    np.save(path, np.array(branch_pts))
    print(f"Branch pts saved: {path}")

# TODO: generalize loaded modules
def get_branch_pts(skeleton):
    def load_kernels():
        if os.path.exists(FLAGS.cache_path):
            # Load unique kernels from cache
            print("Unique kernel cache found! Loading...")
            kernels = load_from_cache(FLAGS.cache_path)
            print(f"{len(kernels)} kernels loaded!")
            return kernels

        # Get unique kernels
        print("Kernel cache not found! Generating...")
        kernels = get_unique_kernels()
        return kernels

    branch_pts_img = cp.zeros(skeleton.shape, dtype=int)
    for kernel in tqdm(kernels):
        branch_pts_img = cp.logical_or(
            binary_hit_or_miss(skeleton, structure1=cp.asarray(kernel)),
            branch_pts_img
        )
    branch_pts_img = cp.asnumpy(branch_pts.img.astype(np.uint8) * 255)
    branch_pts = np.nonzero(branch_pts_img)
    return branch_pts, branch_pts_img


def get_vessel_segmentation(vessels, mask):
    # mask image
    masked_vessels = cle.mask(vessels, mask)
    del vessels

    # add mild blurring
    blurred_vessels = cle.gaussian_blur(masked_vessels,
                                        sigma_x=2,
                                        sigma_y=2,
                                        sigma_z=1,)
    del masked_vessels

    # top hat background subtraction
    tophat_vessels = cle.top_hat_box(blurred_vessels,
                                     radius_x=10,
                                     radius_y=10,
                                     radius_z=5,)
    del blurred_vessels

    # Threshold image
    thresh_vessels = cle.threshold_otsu(tophat_vessels)
    del tophat_vessels

    # Morphological opening - erosion followed by dilation
    erosion = cle.erode_box(thresh_vessels)
    del thresh_vessels

    dilation = cle.dilate_box(erosion)
    del erosion

    # Convert to uint16
    dilation = dilation.astype(np.uint16)
    dilation *= 2**10 - 1

    return dilation

def load_oib(path):
    with OifFile(path) as oif:
        viewer = OifImageViewer(oif)
        # steps = map(float, [viewer.md[i] for i in ["x_step", "y_step","z_step"]]
        # x_step, y_step, z_step = tuple(steps)
        # if viewer.md["z_unit"] == "nm":
        #     z_step /= 1000. 

    return viewer.get_array()


def load_mask(path):
    # Load mask from NRRD
    mask, header = nrrd.read(path)
    mask = mask.T
    
    # Convert mask from uint8 to uint16
    mask = mask.astype(np.uint16)
    mask *= 2**16 - 1
    return mask
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to .OIB file"
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default="/home/dkermany/data/Bone_Project/masks/",
        help="Path to masks folder"
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="../lib/unique_kernels.npy",
        help="Path to .OIB file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/dkermany/data/Bone_Project/branch_points/",
        help="Path to output"
    )
    FLAGS, _ = parser.parse_known_args()
    main()
     
