import os
import numpy as np
import cupy as cp

def create_directory(path):
    """
    Creates directory at path if not exists
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

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
    # Use numpy's linalg.norm function to calculate the Euclidean distance
    return np.linalg.norm(np.array(point1)-np.array(point2))

def normalize(x, min, max):
    return (x - min) / (max - min)

def replace_np_values(arr: np.ndarray, map: dict) -> np.ndarray:
    fn = np.vectorize(lambda x: map.get(x, 0) * 255)
    return fn(arr)

