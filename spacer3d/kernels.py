import numpy as np
import itertools
from tqdm import tqdm

BASE_KERNEL = [[[0,0,0],
                [0,0,0],
                [0,0,0],
               ],
               [[0,0,0],
                [0,1,0],   
                [0,0,0],
               ],
               [[0,0,0],
                [0,0,0],
                [0,0,0]]]

def get_unique_tip_kernels():
    unique_tip_kernels = []
    tip_kernels = _get_tip_kernels()
    for kernel in tip_kernels:
        unique_tip_kernels.extend(_get_unique_orientations(kernel))
    return unique_tip_kernels

def get_unique_kernels(path):
    kernels = []
    for kernel in _get_kernels() + _load_generated_txt_kernels(path):
        kernels.extend(_get_unique_orientations(kernel))
    return _find_unique_kernels(kernels)

def load_from_cache(path):
    kernels = []
    unique_kernels = np.load(path)
    for kernel in unique_kernels:
        kernels.extend(_get_unique_orientations(kernel))
    return kernels

def _load_generated_txt_kernels(path):
    def get_kernels_from_indices(indices):
        kernels = []
        for i in indices:
            kernel = np.array(BASE_KERNEL, dtype=int).flatten()
            for j in i:
                kernel[j-1] = 1
            kernels.append(np.reshape(kernel, (3,3,3)))
        return kernels

    with open(path, "r") as f:
        fourway_indices = [list(map(int, i.split())) for i in f.readlines()]
        fourway_kernels = get_kernels_from_indices(fourway_indices)
    return fourway_kernels


def _get_unique_orientations(cube):
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

def _find_unique_kernels(kernels, save_path="../../lib/unique_kernels.npy"):
    def _has_duplicates(a, b):
        """
        Returns True if there are any duplicates
        """
        # combinations = [p for p in itertools.product(a, b)]
        combinations = itertools.product(a, b)
        return any([np.array_equal(i, j) for i, j in combinations])

    unique_kernels = []
    for a in tqdm(kernels):
        for b in unique_kernels:
            a_orientations = _get_unique_orientations(a)
            b_orientations = _get_unique_orientations(b)
            if _has_duplicates(a_orientations, b_orientations):
                break
        else:
            unique_kernels.append(a)

    np.save(save_path, unique_kernels)
    print("Kernel cache generated successfully!")
    return unique_kernels
    

def _get_tip_kernels():
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

    return tip_kernels
    

def _get_kernels():
    kernels = []

    # Format: [[front],
    #          [middle],
    #          [back]]

    # Filter 1
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0],
                             ],
                             [[0, 1, 0],
                              [1, 1, 1],
                              [0, 0, 0],
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0],]]))

    # Filter 2
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 1],
                              [0, 0, 0]
                             ],
                             [[0, 1, 0],
                              [1, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 3
    kernels.append(np.array([[[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 1, 0],
                              [1, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 4
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]
                             ],
                             [[0, 1, 0],
                              [1, 1, 0],
                              [0, 0, 1]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 5
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 1]
                             ],
                             [[0, 1, 0],
                              [1, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 6
    kernels.append(np.array([[[0, 0, 1],
                              [0, 0, 0],
                              [1, 0, 0]
                             ],
                             [[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 7
    kernels.append(np.array([[[0, 0, 1],
                              [0, 0, 0],
                              [0, 0, 0]
                             ],
                             [[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [1, 0, 0]]]))

    # Filter 8
    kernels.append(np.array([[[0, 0, 1],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 9
    kernels.append(np.array([[[0, 0, 1],
                              [0, 0, 0],
                              [0, 0, 0]
                             ],
                             [[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 10
    kernels.append(np.array([[[0, 0, 1],
                              [0, 0, 0],
                              [0, 0, 0]
                             ],
                             [[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 1]]]))

    # Filter 11
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 1]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[1, 0, 1],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 12
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[1, 0, 1],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 13
    kernels.append(np.array([[[0, 0, 1],
                              [0, 0, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[1, 0, 1],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 14
    kernels.append(np.array([[[0, 1, 0],
                              [0, 0, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[1, 0, 1],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 15
    kernels.append(np.array([[[0, 0, 0],
                              [1, 0, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[1, 0, 1],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 16
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]
                             ],
                             [[1, 0, 1],
                              [0, 1, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 17
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[1, 0, 1],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 18
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 1]
                             ],
                             [[1, 0, 1],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 19
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]
                             ],
                             [[1, 0, 1],
                              [0, 1, 0],
                              [1, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 20
    kernels.append(np.array([[[0, 0, 1],
                              [0, 0, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 1, 0]
                             ],
                             [[1, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))


    # Filter 21
    kernels.append(np.array([[[1, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 1],
                              [0, 1, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 22
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 1],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 1, 0]
                             ],
                             [[1, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 23
    kernels.append(np.array([[[0, 0, 1],
                              [0, 0, 0],
                              [1, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[1, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 24
    kernels.append(np.array([[[1, 0, 0],
                              [0, 0, 1],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 25
    kernels.append(np.array([[[1, 0, 1],
                              [0, 0, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 26
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]
                             ],
                             [[0, 1, 0],
                              [0, 1, 0],
                              [0, 0, 1]
                             ],
                             [[0, 0, 0],
                              [1, 0, 0],
                              [0, 0, 0]]]))

    # Filter 27
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 1],
                              [0, 0, 0]
                             ],
                             [[0, 1, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [1, 0, 0],
                              [0, 0, 0]]]))

    # Filter 28
    kernels.append(np.array([[[0, 0, 0],
                              [1, 0, 1],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 29
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[1, 0, 0],
                              [0, 0, 1],
                              [0, 0, 0]]]))

    # Filter 31
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]
                             ],
                             [[0, 0, 1],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 32
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]
                             ],
                             [[0, 1, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 33
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]
                             ],
                             [[1, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 34
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [1, 0, 1]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))


    # Filter 35
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]
                             ],
                             [[0, 0, 0],
                              [1, 0, 0],
                              [0, 0, 0]]]))

    # Filter 36
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 1],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))


    # Filter 37
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]
                             ],
                             [[0, 0, 0],
                              [0, 0, 1],
                              [0, 0, 0]]]))

    # Filter 38
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 1],
                              [0, 1, 0],
                              [0, 0, 1]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))


    # Filter 39
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 1, 0],
                              [0, 1, 0],
                              [0, 0, 1]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 40
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [1, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 1]]]))

    # Filter 41
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 1],
                              [0, 0, 0]]]))

    # Filter 42
    kernels.append(np.array([[[0, 1, 0],
                              [0, 0, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [1, 1, 1],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # (4-way branches)
    # Filter 43 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 1, 0],
                              [1, 1, 1],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 44 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]
                             ],
                             [[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 45 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 1, 0],
                              [1, 1, 1],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 46 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 1, 0],
                              [1, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 1],
                              [0, 0, 0]]]))

    # Filter 50 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [1, 1, 0],
                              [0, 0, 1]
                             ],
                             [[0, 1, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 51 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [1, 1, 1],
                              [0, 0, 0]
                             ],
                             [[0, 1, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 52 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [1, 0, 1]
                             ],
                             [[0, 1, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 1]]]))

    # Filter 53 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 1, 0],
                              [0, 1, 0],
                              [1, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 1]]]))

    # Filter 54 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 1]
                             ],
                             [[0, 1, 0],
                              [0, 1, 0],
                              [1, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 1]]]))

    # Filter 55 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 1]
                             ],
                             [[0, 1, 0],
                              [1, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 1]]]))

    # Filter 56 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 1, 0],
                              [1, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 1]]]))

    # Filter 57 
    kernels.append(np.array([[[0, 0, 0],
                              [1, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]
                             ],
                             [[0, 1, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 58 
    kernels.append(np.array([[[1, 0, 0],
                              [0, 0, 0],
                              [1, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]
                             ],
                             [[0, 1, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 59 
    kernels.append(np.array([[[0, 0, 0],
                              [1, 0, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 1],
                              [0, 1, 0],
                              [0, 0, 1]
                             ],
                             [[0, 1, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 60 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [1, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 1, 0],
                              [0, 0, 0],
                              [0, 0, 1]]]))

    # Filter 62 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [1, 1, 0],
                              [0, 0, 1]
                             ],
                             [[0, 1, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 65 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]
                             ],
                             [[0, 1, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 66 
    kernels.append(np.array([[[0, 1, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [1, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 1]]]))

    # Filter 67 
    kernels.append(np.array([[[0, 1, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [1, 0, 0],
                              [0, 0, 1]]]))

    # Filter 68 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 1, 0],
                              [1, 0, 0],
                              [0, 0, 1]]]))

    # Filter 69 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 1],
                              [1, 0, 0],
                              [0, 0, 1]]]))

    # Filter 71 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [1, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[1, 0, 1],
                              [0, 0, 0],
                              [0, 0, 1]]]))

    # Filter 72 
    kernels.append(np.array([[[1, 0, 0],
                              [0, 0, 0],
                              [1, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 1],
                              [0, 0, 0],
                              [0, 0, 1]]]))

    # Filter 73 
    kernels.append(np.array([[[1, 0, 1],
                              [0, 0, 0],
                              [1, 0, 1]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 74 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [1, 0, 0]
                             ],
                             [[1, 0, 1],
                              [0, 0, 0],
                              [0, 0, 1]]]))

    # Filter 75 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 1, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 1],
                              [1, 0, 0]]]))

    # Filter 77 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 1, 0],
                              [0, 0, 0],
                              [1, 0, 1]]]))

    # Filter 78 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 1],
                              [0, 0, 0]
                             ],
                             [[0, 1, 0],
                              [0, 1, 0],
                              [1, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 1],
                              [0, 0, 0]]]))

    # Filter 79 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 1],
                              [0, 0, 0]
                             ],
                             [[0, 1, 0],
                              [1, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 1],
                              [0, 0, 0]]]))

    # Filter 80 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 1, 0],
                              [0, 0, 0],
                              [1, 0, 0]]]))

    # Filter 81 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 1, 0],
                              [0, 0, 0],
                              [0, 0, 1]]]))

    # Filter 82 
    kernels.append(np.array([[[1, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 1],
                              [0, 0, 0],
                              [1, 0, 0]]]))

    # Filter 85
    kernels.append(np.array([[[0, 0, 0],
                              [1, 0, 0],
                              [0, 0, 0]
                             ],
                             [[0, 1, 0],
                              [1, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 1]]]))

    # Filter 86
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]
                             ],
                             [[1, 0, 1],
                              [1, 1, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 87
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 1]
                             ],
                             [[0, 1, 0],
                              [0, 1, 1],
                              [1, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 88
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]
                             ],
                             [[1, 0, 1],
                              [0, 1, 0],
                              [1, 0, 1]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 89
    kernels.append(np.array([[[1, 0, 1],
                              [0, 0, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 1, 0],
                              [0, 0, 0],
                              [0, 0, 1]]]))

    # Filter 91 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 1],
                              [0, 1, 0],
                              [1, 0, 1]
                             ],
                             [[1, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 93 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 1],
                              [0, 0, 0]
                             ],
                             [[0, 1, 0],
                              [0, 1, 1],
                              [1, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 94 
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 1],
                              [0, 1, 0]
                             ],
                             [[0, 1, 0],
                              [1, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))


    # Filter 95 
    kernels.append(np.array([[[1, 0, 0],
                              [0, 0, 0],
                              [1, 0, 0]
                             ],
                             [[0, 0, 1],
                              [0, 1, 0],
                              [0, 0, 1]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 96 
    kernels.append(np.array([[[1, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 1],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 97
    kernels.append(np.array([[[0, 1, 0],
                              [0, 0, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [1, 1, 0],
                              [1, 0, 1]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 99
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 1],
                              [1, 1, 0],
                              [1, 0, 1]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 100
    kernels.append(np.array([[[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [1, 1, 0],
                              [1, 0, 1]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 102
    kernels.append(np.array([[[1, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [1, 1, 0],
                              [1, 0, 1]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 103
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [1, 1, 0],
                              [1, 0, 1]
                             ],
                             [[0, 0, 1],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    # Filter 104
    kernels.append(np.array([[[0, 0, 0],
                              [1, 0, 0],
                              [1, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 1],
                              [0, 0, 0],
                              [0, 0, 1]]]))

    # Filter 105
    kernels.append(np.array([[[1, 0, 0],
                              [1, 0, 0],
                              [1, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 1]]]))

    # Filter 106
    kernels.append(np.array([[[0, 0, 0],
                              [1, 0, 0],
                              [1, 0, 0]
                             ],
                             [[0, 1, 0],
                              [0, 1, 0],
                              [0, 0, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 1]]]))

    # Filter 107 (5-way branch)
    kernels.append(np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [1, 0, 0]
                             ],
                             [[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]
                             ],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))

    return kernels
