import numpy as np
import multiprocessing as mp
import raster_geometry as rg
from scipy import spatial
from functools import reduce
from tqdm import tqdm

cache = mp.Manager().dict()

class Ripley():
    def __init__(
        self,
        points: np.ndarray,
        radii: list,
        mask: np.ndarray,
    ):
        self.points = points
        self.radii = radii
        if mask:
            self.mask = mask.astype(np.uint8)
            self.boundary_correction = True

        self.volume_shape = self.mask.shape
        self._validate_inputs()

        self.tree = spatial.cKDTree(self.points)
        self.study_volume = reduce(lambda x, y: x * y, self.volume_shape)

    def run_ripley(self, processes=32):
        # Paralellize ripley across processes using the multiprocessing library
        with mp.Manager() as m:
            # Need to store results in mp.Manager() to track across processes
            self.results = {"K": m.list(), "L": m.list(), "H": m.list()}
            with mp.Pool(processes) as pool:
                list(tqdm(pool.imap(self._calc_ripley, self.radii), total=len(self.radii)))
            return list(self.results["K"]), list(self.results["L"]), list(self.results["H"])

    def _calc_ripley(self, radius):
        # For each radius, loop through each point and count points
        # within the radius
        print("Running univariate Ripley K...")
        nb_count = 0
        for z, y, x in self.points:
            if self.boundary_correction:
                weight = self.calculate_weight(radius, (z, y, x))
                # If weight is zero (i.e. target sphere not in mask), move on
                if weight == 0:
                    continue
            else:
                weight = 1.0

            # query_ball_point() includes the index of the current point as well
            # so 1 is subtracted from the count 
            nb_count += (len(self.tree.query_ball_point([z, y, x], radius)) - 1) / weight

         # calculating 3D Ripley's functions (K, L, H)
        N = self.points.shape[0]
        K = nb_count * self.study_volume / (N * (N - 1))
        L = ((3. / 4) * (K / np.pi)) ** (1. / 3)
        H = L - radius
        
        # Verify K/L values positive
        if K < 0 or L < 0:
            raise ValueError(f"K/L values should not be negative. nb_count: {nb_count}, volume: {self.volume_shape}, N: {N}")

        self.results["K"].append((radius, K))
        self.results["L"].append((radius, L))
        self.results["H"].append((radius, H))

    def calculate_weight(self, radius, center):
        """
        Calculate the proportion of a sphere within a study volume.

        Args:
        radius (int): The radius of the sphere.
        center (tuple): A 3-tuple containing the z, y, x coordinates of the center of the sphere.

        Returns:
        float: The proportion of the sphere within the study volume.
        """
        # Ensure that the radius is greater than zero
        assert radius > 0

        # Check if cached weight for this coordinate exists
        key = center + (radius,)
        if key in cache:
            return cache[key]
        
        # Draw the target sphere in a 3D NumPy array at the specified position
        target = np.zeros(self.volume_shape, dtype=np.uint8)
        draw_sphere_in_volume(target, radius, center)

        # Bitwise and operation between the sphere and mask to calculate intersection
        target = target & self.mask

        # Calculate the sum (volume) of the target and reference sphere arrays
        target = target.sum()

        # Reference is calculated instead of simulated for speed increase
        reference = (4 / 3) * np.pi * (radius ** 3)

        # Ensure that the reference sphere has a non-zero volume
        assert reference > 0

        # Return the proportion of the sphere within the study volume
        # Since target volume is estimated whereas reference volume is calculated,
        # there is a small but noticable margin of error when radius < 10, there
        # we set maxiumum value of 1.0
        weight = min(target / reference, 1.0)

        # Save weight to cache
        cache[key] = weight
        
        return weight

    def _validate_inputs(self):
        # Check if self.points is a list or numpy array
        if not isinstance(self.points, (list, np.ndarray)):
            e = f"Expected {np.ndarray}, received {type(self.points)}"
            raise ValueError(e)

        # Convert self.points to numpy array if it is a list
        if not isinstance(self.points, np.ndarray):
            self.points = np.array(self.points)

        # Check if self.points array has two dimensions
        if len(self.points.shape) != 2:
            e = f"Expected self.points array to have 2 dimensions, but got array with shape {self.points.shape}"
            raise ValueError(e)

        # Check if the self.points array second dimension length is 3 (x, y, z)
        if self.points.shape[1] != 3:
            e = f"Expected self.points array to have shape (None, 3), but got array with shape {self.points.shape}"
            raise ValueError(e)

        # Check if the self.points array has at least 3 points
        if self.points.shape[0] < 3:
            e = f"Expected self.points array to have at least 3 points"
            raise ValueError(e)

        # Check if radii is list-like or number-like
        if not isinstance(self.radii, (np.ndarray, list, int, float)):
            e = f"Expected {(np.ndarray, list, int, float)}, received {type(self.radii)}"
            raise ValueError(e)

        # if only one radius given as int, convert to list
        if not isinstance(self.radii, (np.ndarray, list)):
            self.radii = [self.radii]


### Multivariate Ripley Class
class CrossRipley(Ripley):
    def __init__(
        self,
        points_i: np.ndarray,
        points_j: np.ndarray,
        radii: list,
        mask: np.ndarray,
    ):
        self.points_i = points_i
        self.points_j = points_j
        self.radii = radii

        if mask:
            self.mask = mask.astype(np.uint8)
            self.boundary_correction = True
        
        self.volume_shape = self.mask.shape
        self._validate_inputs()

        self.i_tree = spatial.cKDTree(self.points_i)
        self.j_tree = spatial.cKDTree(self.points_j)
        self.study_volume = reduce(lambda x, y: x * y, self.volume_shape)


    def test_ripley(self):
        self.results = {"K": [], "L": [], "H": []}
        for r in tqdm(self.radii):
            self._calc_ripley(r)
        return list(self.results["K"]), list(self.results["L"]), list(self.results["H"])

    # TODO: Rewrite univariate _calc_ripley function for multivariate case
    def _calc_ripley(self, radius):
        """
        Calculate 3D multivariate Ripley's functions (K_ij, L_ij, H_ij) for a given radius and
        and apply weight coefficient.

        For each radius, loop through each point_i and count number of point_j within the radius.
        If boundary_correction is True, calculate the weight coefficient based on the volume of
        the search sphere within the study volume.

        Args:

            radius (float): the radius for which to calculate Ripley's functions.

        Raises:
            ValueError: if K/L values are negative.

        Returns:
            None. Results are stored in self.results.

        """
        # For each radius, loop through each point and count points
        # within the radius
        nb_count = 0
        for z, y, x in self.points_i:
            # print(f"{i}/{len(self.points_i)}", end="\r")
            if self.boundary_correction:
                weight = self.calculate_weight(radius, (z, y, x))
                # If weight is zero (i.e. target sphere not in mask), move on
                if weight == 0:
                    continue
            else:
                weight = 1.0

            # Since the i point is not included within the j_tree, we do not subtract 1
            # as done in the univariate implementation
            
            nb_count += (len(self.j_tree.query_ball_point([z, y, x], radius, workers=-1))) / weight

            # global pbar
            # pbar.update()

        # calculating 3D Ripley's functions (K_ij, L_ij, H_ij)
        N_i = self.points_i.shape[0]
        N_j = self.points_j.shape[0]
        K_ij = nb_count * self.study_volume / (N_i * N_j)
        L_ij = ((3. / 4) * (K_ij / np.pi)) ** (1. / 3)
        H_ij = L_ij - radius
        
        # Verify K/L values positive
        if K_ij < 0 or L_ij < 0:
            raise ValueError(f"K/L values should not be negative. nb_count: {nb_count}, N_i: {N_i}, N_j: {N_j}")

        self.results["K"].append((radius, K_ij))
        self.results["L"].append((radius, L_ij))
        self.results["H"].append((radius, H_ij))

    def _validate_inputs(self):
        # Check if self.points_i is a list or numpy array
        if not isinstance(self.points_i, (list, np.ndarray)):
            e = f"Expected {np.ndarray}, received {type(self.points_i)}"
            raise ValueError(e)

        # Check if self.points_j is a list or numpy array
        if not isinstance(self.points_j, (list, np.ndarray)):
            e = f"Expected {np.ndarray}, received {type(self.points_j)}"
            raise ValueError(e)

        # Convert self.points_i to numpy array if it is a list
        if not isinstance(self.points_i, np.ndarray):
            self.points_i = np.array(self.points_i)
            
        # Convert self.points_j to numpy array if it is a list
        if not isinstance(self.points_j, np.ndarray):
            self.points_j = np.array(self.points_j)

        # Check if self.points_i array has two dimensions
        if len(self.points_i.shape) != 2:
            e = f"Expected self.points_i array to have 2 dimensions, but got array with shape {self.points_i.shape}"
            raise ValueError(e)

        # Check if self.points_j array has two dimensions
        if len(self.points_j.shape) != 2:
            e = f"Expected self.points_j array to have 2 dimensions, but got array with shape {self.points_j.shape}"
            raise ValueError(e)

        # Check if the self.points_i array second dimension length is 3 (x, y, z)
        if self.points_i.shape[1] != 3:
            e = f"Expected self.points_i array to have shape (None, 3), but got array with shape {self.points_i.shape}"
            raise ValueError(e)

        # Check if the self.points_j array second dimension length is 3 (x, y, z)
        if self.points_j.shape[1] != 3:
            e = f"Expected self.points_j array to have shape (None, 3), but got array with shape {self.points_j.shape}"
            raise ValueError(e)

        # Check if the self.points_i array has at least 3 points
        if self.points_i.shape[0] < 3:
            e = f"Expected self.points_i array to have at least 3 points"
            raise ValueError(e)

        # Check if the self.points_j array has at least 3 points
        if self.points_j.shape[0] < 3:
            e = f"Expected self.points_j array to have at least 3 points"
            raise ValueError(e)

        # Check if radii is list-like or number-like
        if not isinstance(self.radii, (np.ndarray, list, int, float)):
            e = f"Expected {(np.ndarray, list, int, float)}, received {type(self.radii)}"
            raise ValueError(e)

        # if only one radius given as int, convert to list
        if not isinstance(self.radii, (np.ndarray, list)):
            self.radii = [self.radii]

        # if points are not within volume, raise error
        for p in [self.points_i, self.points_j]:
            assert all(x < self.volume_shape[2] for x in p[:, 2])
            assert all(y < self.volume_shape[1] for y in p[:, 1])
            assert all(z < self.volume_shape[0] for z in p[:, 0])


def draw_sphere_in_volume(volume: np.ndarray, radius: int, position: tuple) -> None:
    """
    Draw a sphere in a given 3D NumPy array at a specified position.

    Args:
    volume (numpy.ndarray): The 3D NumPy array in which the sphere will be drawn.
    radius (int): The radius of the sphere.
    position (tuple): A 3-tuple containing the z, y, x coordinates of the position in the array where the sphere will be drawn.

    Returns:
    None
    """
    
    # Create an empty 3D NumPy array with dimensions equal to twice the radius plus one
    size = 2 * (radius + 1)

    # Calculate the midpoint of the sphere unit array
    midpoint = [size / 2] * 3

    # Generate a unit sphere using the rg library's superellipsoid function
    sphere = rg.nd_superellipsoid(size, radius, position=midpoint,
                                  rel_sizes=False, rel_position=False).astype(np.int_)

    # Extract the z, y, x coordinates of the position where the sphere will be drawn
    z, y, x = map(round, position)

    # Calculate the delta change needed to center the sphere at the specified position
    d = (size//2)

    # Calculate the minimum and maximum indices for the z, y, x axes of the volume array
    # print(position, radius, x-d, x+d, max(x - d, 0), min(x + d, volume.shape[2]))
    zmin, zmax = max(z - d, 0), min(z + d, volume.shape[0])
    ymin, ymax = max(y - d, 0), min(y + d, volume.shape[1])
    xmin, xmax = max(x - d, 0), min(x + d, volume.shape[2])

    # Calculate the minimum indices for the z, y, x axes of the sphere array
    szmin = abs(z - d) if z - d < 0 else 0
    symin = abs(y - d) if y - d < 0 else 0
    sxmin = abs(x - d) if x - d < 0 else 0

    # Calculate the amount to cut off of the ends of the z, y, x axes of the sphere array
    szmax = abs(volume.shape[0] - (z + d)) if z + d > volume.shape[0] else 0
    symax = abs(volume.shape[1] - (y + d)) if y + d > volume.shape[1] else 0
    sxmax = abs(volume.shape[2] - (x + d)) if x + d > volume.shape[2] else 0
    # assert x+d < volume.shape[2]
    
    # Trim the sphere array to fit within the trimmed volume array
    sphere = sphere[szmin:sphere.shape[0]-szmax, symin:sphere.shape[1]-symax, sxmin:sphere.shape[2]-sxmax]
    # Place the sphere within the larger volume array at the specified position

    # print(z, y, x, d, "|", zmin,zmax,"|",ymin,ymax,"|",xmin,xmax, "|",sphere.shape, "|", sxmin, sxmax)
    volume[zmin:zmax, ymin:ymax, xmin:xmax] = sphere