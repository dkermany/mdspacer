import os
import argparse
import numpy as np
import multiprocessing as mp
import raster_geometry as rg
import pandas as pd
import time
import nrrd
from scipy import spatial, stats
from functools import reduce
from tqdm import tqdm
from oiffile import OifFile
from spacer3d.OifImageViewer import OifImageViewer
import matplotlib.pyplot as plt
import seaborn as sns

cache = {}

def load_mask(path):
    # Load mask from NRRD
    mask, header = nrrd.read(path)
    mask = mask.T
    #mask = mask[:mask.shape[0]//3, :, :] 
                        
    print(mask.shape)
    print(mask.dtype)
    print("Axes: ZYX")

    return mask

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
    return np.linalg.norm(point1-point2)

def set_aspect_ratio_equal(ax) -> None:
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    ax.set_box_aspect((xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]))

def get_sphere_mask_intersection(
        mask: np.ndarray,
        radius: int,
        position: tuple) -> np.ndarray:
    """
    Draw a sphere in a given 3D NumPy array at a specified position.

    Args:
    mask (numpy.ndarray): The 3D NumPy array in which the sphere will be drawn.
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

    # Calculate the minimum and maximum indices for the z, y, x axes of the mask array
    # print(position, radius, x-d, x+d, max(x - d, 0), min(x + d, mask.shape[2]))
    zmin, zmax = max(z - d, 0), min(z + d, mask.shape[0])
    ymin, ymax = max(y - d, 0), min(y + d, mask.shape[1])
    xmin, xmax = max(x - d, 0), min(x + d, mask.shape[2])

    # Calculate the minimum indices for the z, y, x axes of the sphere array
    szmin = abs(z - d) if z - d < 0 else 0
    symin = abs(y - d) if y - d < 0 else 0
    sxmin = abs(x - d) if x - d < 0 else 0

    # Calculate the amount to cut off of the ends of the z, y, x axes of the sphere array
    szmax = abs(mask.shape[0] - (z + d)) if z + d > mask.shape[0] else 0
    symax = abs(mask.shape[1] - (y + d)) if y + d > mask.shape[1] else 0
    sxmax = abs(mask.shape[2] - (x + d)) if x + d > mask.shape[2] else 0
    # assert x+d < mask.shape[2]
    
    # Trim the sphere array to fit within the trimmed mask array
    sphere = sphere[szmin:sphere.shape[0]-szmax, symin:sphere.shape[1]-symax, sxmin:sphere.shape[2]-sxmax]
    # Place the sphere within the larger mask array at the specified position

    # Return intersection between mask subset and sphere
    return mask[zmin:zmax, ymin:ymax, xmin:xmax] & sphere


### Ripley Class 
class Ripley():
    """
    Ripley Class for spatial point pattern analysis.
    """
    def __init__(
            self,
            points_i: np.ndarray,
            radii: list,
            mask: np.ndarray,
            boundary_correction: bool = True,
            disable_progress: bool = False,
    ):
        """
        Initialize a Ripley object.

        Args:
        points_i (np.ndarray): A 2D NumPy array of shape (N, 3) for 3D and (N, 2) for 2D representing the coordinates of N points in space.
        radii (list): A list of radii at which to calculate Ripley's K, L, and H functions.
        mask (np.ndarray): A binary mask representing the study volume.
        boundary_correction (bool, optional): Whether to apply boundary correction. Defaults to True.
        disable_progress (bool, optional): Whether to disable progress bar. Defaults to False.
        """
        self.points_i = points_i
        self.radii = radii
        self.mask = mask.astype(np.uint8)
        self.volume_shape = self.mask.shape
        self.boundary_correction = boundary_correction
        self.disable_progress = disable_progress
        self._validate_inputs()

        self.i_tree = spatial.cKDTree(self.points_i)
        self.study_volume = reduce(lambda x, y: x * y, self.volume_shape)

    @staticmethod
    def worker(fn, task_queue, result_queue):
        while True:
            task = task_queue.get()
            if task is None:  # 'None' is the signal to stop.
                break
            index, radius = task
            result = fn(radius)
            result_queue.put((index, result))

    def run(self, n_processes=32):
        """
        Run Ripley's analysis for the specified radii.

        Args:
        n_processes (int, optional): Number of processes to use for parallel computation. Defaults to 32.

        Returns:
        tuple: A tuple containing lists of K, L, and H values for each radius.
        """
        task_queue = mp.Queue()
        result_queue = mp.Queue()

        # Start worker processes
        processes = [mp.Process(target=self.worker, args=(self._calc_ripley, task_queue, result_queue))
                     for _ in range(n_processes)]
        for p in processes:
            p.start()

        # Distribute tasks
        for index, input_data in enumerate(self.radii):
            task_queue.put((index, input_data))

        # Signal the end of tasks
        for _ in processes:
            task_queue.put(None)

        # Collect results
        results = [None] * len(self.radii)
        for _ in range(len(self.radii)):
            index, result = result_queue.get()
            results[index] = result

        # Wait for all worker processes to finish
        for p in processes:
            p.join()

        K, L, H, metrics = zip(*results)
        flattened_metrics = [i for s in metrics for i in s]
        
        return K, L, H, flattened_metrics

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
        # target = np.zeros(self.volume_shape, dtype=np.uint8)
        # draw_sphere_in_volume(target, radius, center)

        # Bitwise and operation between the sphere and mask to calculate intersection
        # target = target & self.mask
        intersection = get_sphere_mask_intersection(self.mask, radius, center)

        # Calculate the sum (volume) of the target and reference sphere arrays
        intersection_vol = intersection.sum()

        # Reference is calculated instead of simulated for speed increase
        reference = (4 / 3) * np.pi * (radius ** 3)

        # Ensure that the reference sphere has a non-zero volume
        assert reference > 0

        # Return the proportion of the sphere within the study volume
        # Since target volume is estimated whereas reference volume is calculated,
        # there is a small but noticable margin of error when radius < 10, there
        # we set maxiumum value of 1.0
        weight = min(intersection_vol / reference, 1.0)

        # Save weight to cache
        cache[key] = weight
        
        return weight

    @staticmethod
    def plot_performance(run_times):
        plt.figure(figsize=(5,5))
        df = pd.DataFrame(run_times, columns=["radius", "time", "type"])
        sns.lineplot(data=df, x="radius", y="time", hue="type")
        plt.show()

    def _validate_inputs(self):
        """Validate the input parameters to ensure they meet the required criteria.

        Raises:
        ValueError: If any of the input parameters do not meet the required criteria.
        """                # pool.map(self._calc_ripley, self.radii)
        # Check if self.points_i is a list or numpy array
        if not isinstance(self.points_i, (list, np.ndarray)):
            e = f"Expected {np.ndarray}, received {type(self.points_i)}"
            raise ValueError(e)

        # Convert self.points_i to numpy array if it is a list
        if not isinstance(self.points_i, np.ndarray):
            self.points_i = np.array(self.points_i)

        # Check if self.points_i array has two dimensions
        if len(self.points_i.shape) != 2:
            e = f"Expected self.points_i array to have 2 dimensions, but got array with shape {self.points_i.shape}"
            raise ValueError(e)

        # Check if the self.points_i array second dimension length is 3 (x, y, z)
        if self.points_i.shape[1] != 3:
            e = f"Expected self.points_i array to have shape (None, 3), but got array with shape {self.points_i.shape}"
            raise ValueError(e)

        # Check if the self.points_i array has at least 3 points
        if self.points_i.shape[0] < 3:
            e = f"Expected self.points_i array to have at least 3 points"
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
    """
    CrossRipley Class for multivariate spatial point pattern analysis.
    Inherits from Ripley class.
    """

    def __init__(
            self,
            points_i: np.ndarray,
            points_j: np.ndarray,
            radii: list,
            mask: np.ndarray,
            boundary_correction: bool = True,
            disable_progress: bool = False,
    ):
        """
        Initialize a CrossRipley object.

        Args:
        points_i (np.ndarray): A 2D NumPy array of shape (N, 3) representing the coordinates of points of type 'i' in 3D space.
        points_j (np.ndarray): A 2D NumPy array of shape (M, 3) representing the coordinates of points of type 'j' in 3D space.
        radii (list): A list of radii at which to calculate Ripley's K, L, and H functions.
        mask (np.ndarray): A 3D binary mask representing the study volume.
        boundary_correction (bool, optional): Whether to apply boundary correction. Defaults to True.
        """
        # Call the parent class's __init__ method using super()
        super().__init__(
            points_i=points_i,
            radii=radii,
            mask=mask,
            boundary_correction=boundary_correction,
            disable_progress=disable_progress,
        )

        # Assign points_j to the self.points_j attribute and validate
        self.points_j = points_j
        self._validate_cross_inputs()

        # Create a new tree for points_j
        self.j_tree = spatial.cKDTree(self.points_j)


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
        # print(f"radius = {radius}", flush=True)
        nb_count = 0
        running_weights = []
        running_trees = []
        for z, y, x in self.points_i:
            start_time = time.time()

            if self.boundary_correction:
                weight = self.calculate_weight(radius, (z, y, x))
                # If weight is zero (i.e. target sphere not in mask), move on
                if weight == 0:
                    continue
            else:
                weight = 1.0

            end_time = time.time()
            weight_time = end_time - start_time
            running_weights.append(1000*weight_time)

            # Since the i point is not included within the j_tree, we do not subtract 1
            # as done in the univariate implementation
            
            start_time = time.time()

            nb_count += (len(self.j_tree.query_ball_point([z, y, x], radius, workers=-1))) / weight

            end_time = time.time()
            tree_time = end_time - start_time
            running_trees.append(1000*tree_time)

        # calculating 3D Ripley's functions (K_ij, L_ij, H_ij)
        N_i = self.points_i.shape[0]
        N_j = self.points_j.shape[0]
        K_ij = nb_count * self.study_volume / (N_i * N_j)
        L_ij = ((3. / 4) * (K_ij / np.pi)) ** (1. / 3)
        H_ij = L_ij - radius
        
        # Verify K/L values positive
        if K_ij < 0 or L_ij < 0:
            raise ValueError(f"K/L values should not be negative. nb_count: {nb_count}, volume: {self.volume_shape}, N_i: {N_i}, N_j: {N_j}")

        return (
            K_ij,
            L_ij,
            H_ij,
            [(radius, sum(running_weights) / len(running_weights), "weights"),
             (radius, sum(running_trees) / len(running_trees), "trees"),]
        )

    def _validate_cross_inputs(self):
        # Check if self.points_j is a list or numpy array
        if not isinstance(self.points_j, (list, np.ndarray)):
            e = f"Expected {np.ndarray}, received {type(self.points_j)}"
            raise ValueError(e)

        # Convert self.points_j to numpy array if it is a list
        if not isinstance(self.points_j, np.ndarray):
            self.points_j = np.array(self.points_j)

        # Check if self.points_j array has two dimensions
        if len(self.points_j.shape) != 2:
            e = f"Expected self.points_j array to have 2 dimensions, but got array with shape {self.points_j.shape}"
            raise ValueError(e)

        # Check if the self.points_j array second dimension length is 3 (x, y, z)
        if self.points_j.shape[1] != 3:
            e = f"Expected self.points_j array to have shape (None, 3), but got array with shape {self.points_j.shape}"
            raise ValueError(e)

        # Check if the self.points_j array has at least 3 points
        if self.points_j.shape[0] < 3:
            e = f"Expected self.points_j array to have at least 3 points"
            raise ValueError(e)

        # if only one radius given as int, convert to list
        if not isinstance(self.radii, (np.ndarray, list)):
            self.radii = [self.radii]

        # if points are not within volume, raise error
        for p in [self.points_j]:
            assert all(x < self.volume_shape[2] for x in p[:, 2])
            assert all(y < self.volume_shape[1] for y in p[:, 1])
            assert all(z < self.volume_shape[0] for z in p[:, 0])

def run_ripley(
        points_i: np.ndarray,
        points_j: np.ndarray,
        mask: np.ndarray,
        radii: np.ndarray,
        boundary_correction: bool = False,
        n_processes: int = 32,
        n_line: int = None,
        disable_progress: bool = False) -> list:
    """
    Execute the Ripley's K-function analysis for two sets of points within a specified mask.

    This function computes Ripley's cross K-function, L-function, and H-function for two sets 
    of points. The analysis is based on a range of specified radii and can include boundary 
    correction. The results are sorted by radii and organized in a structured format.

    Parameters:
    points_i (list of tuples): The first set of points (e.g., (x, y) coordinates) for analysis.
    points_j (list of tuples): The second set of points for the cross K-function analysis.
    mask (array-like): A mask or region within which the analysis is conducted.
    radii (list): A list of radii values for which the K-function is computed.
    boundary_correction (bool, optional): Whether to apply boundary correction. Defaults to True.
    n_processes (int, optional): The number of processes to use for computation. Defaults to 1.
    n_line (int, optional): An additional parameter to include in the results, if provided.
    disable_progress (bool, optional): Set to True to disable tqdm progress bar for Ripley calculations

    Returns:
    pd.DataFrame: A DataFrame that contains the radius, K-function value, L-function value, 
                  H-function value, and optionally the n_line value, sorted by radii.
    """
    # Initialize the CrossRipley object with the provided points, radii, mask, and boundary correction setting.
    # This object will be used to run Ripley's K-function analysis.
    ripley = CrossRipley(
        points_i,
        points_j,
        radii,
        mask,
        boundary_correction,
        disable_progress,
    )
    
    # Run the analysis using the specified number of processes and sort the results (K, L, H functions) by radii.
    # The map function applies the sorted function to each element (K, L, H) returned by ripley.run.
    # K_w, L_w, H_w, metrics = map(sorted, ripley.run(n_processes))
    K, L, H, metrics = ripley.run(n_processes)
    
    # Organize the results into a structured format for easy interpretation.
    # If n_line is specified, it is included in each tuple; otherwise, only K, L, and H values are included.
    # The comprehension iterates over zipped K, L, H tuples and constructs a result tuple for each set of values.
    rstats = [(r, k, l, h, n_line) if n_line else (r, k, l, h) 
               for k, l, h, r in zip(K, L, H, radii)]

    # print("Plot performance")
    # ripley.plot_performance(metrics)

    columns = ["Radius (r)", "K(r)", "L(r)", "H(r)"] + (["Line"] if n_line else [])
    results = pd.DataFrame(rstats, columns=columns)
    # Return the organized list of tuples as the function's result.
    return results

def monte_carlo(
        points_i,
        mask,
        radii,
        points_j=None,
        n_samples=5,
        boundary_correction=True,
        disable_progress=False,
        n_processes=32
):
    """
    Conducts a Monte Carlo simulation using the Ripley's K-function for spatial data analysis.

    This function can perform either univariate or bivariate spatial analysis based on the input points.
    It generates random points within a specified mask for univariate analysis or shuffles the
    labels of two point sets for bivariate analysis. The Ripley's K-function is computed over
    a range of radii for these points.
    Parameters:
    - points_i (array-like): The first set of points for the analysis.
    - mask (array-like): A binary mask defining the study area, where 1 indicates valid area.
    - radii (array-like): An array of radii values to calculate Ripley
    - points_j (array-like, optional): The second set of points for bivariate analysis.
    - n_samples (int): The number of Monte Carlo samples to draw.
    - boundary_correction (bool): Whether to apply boundary correction.
    - disable_progress (bool): Whether to show the progress bars for Ripley's calculations
    - n_processes (int): Number of processes to use for parallel computation.

    Returns:
    - pd.DataFrame: A DataFrame containing the results of the Monte Carlo simulation.
    """

    def generate_random_points(points, mask):
        """
        Generate random points within the mask for univariate comparisons.

        This function creates a set of points that are randomly distributed within the valid
        areas defined by the mask. Each generated point must fall within the mask.

        Parameters:
        - points (array-like): The original set of points.
        - mask (array-like): The binary mask defining valid areas.

        Returns:
        - array-like: An array of randomly generated points within the mask.
        """
        CSR_points = np.empty((points.shape[0], 3), dtype=np.uint16)
        for i in range(points.shape[0]):
            while True:
                # Generate random point
                z, y, x = map(int, [stats.uniform.rvs(0, mask.shape[j]) for j in range(3)])
                if mask[z, y, x] == 1:
                    CSR_points[i] = np.array([z, y, x])
                    break
                # pool.map(self._calc_ripley, self.radii)
        return CSR_points

    def shuffle_labels(points_i, points_j):
        """
        Shuffles the identity of points for multivariate comparisons.

        This function combines two sets of points and randomly shuffles their order. This is
        used in bivariate analysis to test the null hypothesis of no spatial interaction
        between the two point sets.

        Parameters:
        - points_i (array-like): The first set of points.
        - points_j (array-like): The second set of points.

        Returns:
        - tuple of array-like: Two arrays of shuffled points.
        """
        combined_arr = np.concatenate((points_i, points_j), axis=0)
        np.random.shuffle(combined_arr)  # inplace shuffle
        return np.split(combined_arr, [points_i.shape[0]])

    # Initialize an empty list to store results
    total_results = []
    for n_line in tqdm(range(n_samples), disable=disable_progress):

        # Univariate
        if points_j is None:
            CSR_points = generate_random_points(points_i, mask)
            results = run_ripley(
                CSR_points,
                CSR_points, mask,
                radii,
                boundary_correction,
                n_processes=n_processes,
                n_line=n_line,
                disable_progress=disable_progress
            )

        # Multivariate
        else:
            shuffled_1, shuffled_2 = shuffle_labels(points_i, points_j)
            results = run_ripley(
                shuffled_1,
                shuffled_2,
                mask,
                radii,
                boundary_correction,
                n_processes=n_processes,
                n_line=n_line,                # pool.map(self._calc_ripley, self.radii)
                disable_progress=disable_progress
            )

        # Collect results
        total_results.append(results)

    # Create a DataFrame from the results
    return pd.concat(total_results, ignore_index=True)

def load_tumor_locations(path, filename, steps):
    """
    Load tumor locations from a CSV file, filter rows with missing values, and convert them to a 3D numpy array.

    Parameters:
    path (str): The file path to the CSV file containing tumor data.
    filename (str): The name of the file to filter tumor data by.
    steps (tuple): A tuple containing three values (x_step, y_step, z_step) for converting tumor locations from micrometers to steps.

    Returns:
    numpy.ndarray: A 3D numpy array containing the tumor locations in steps.
    
    This function reads a CSV file at the specified path, drops rows with missing values, and filters the data by the given filename.
    It then converts the x, y, and z coordinates from micrometers to steps using the provided steps tuple and stores them in a 3D numpy array.
    The resulting numpy array contains the tumor locations in steps and is returned as the output of the function.
    """
    # Load CSV data and drop rows with missing values (N/A's)
    tumor_csv = pd.read_csv(path).dropna()
    
    # Filter rows in the CSV where the 'Filename' column matches the given 'filename'
    # Reset the index of the filtered DataFrame
    tumor_csv = tumor_csv[tumor_csv.Filename == filename].reset_index(drop=True)
    
    # Extract individual step values for x, y, and z
    x_step, y_step, z_step = steps

    # Convert the filtered CSV data into a dictionary where each row is an entry
    tumor_dict = tumor_csv.to_dict("index")
    
    # Initialize an empty list to store tumor location points
    tumor_points = []
    
    # Iterate through the tumor information entries in the dictionary
    for idx, tumor_info in tumor_dict.items():
        # Extract x (um), y (um), and z (slice) values and convert them to integers
        x_um, y_um, z_slice = map(int, (tumor_info["x (um)"], tumor_info["y (um)"], tumor_info["z (slice)"]))
        
        # Calculate the corresponding x, y, and z coordinates based on step values
        x, y, z = map(int, (x_um / x_step, y_um / y_step, z_slice - 1))
        
        # Append the calculated coordinates as a list to the tumor_points list
        # in (Z, Y, X) format
        tumor_points.append([z, y, x])
    
    # Convert the list of tumor location points to a NumPy array with dtype float64
    tumor_points = np.array(tumor_points, dtype=np.float64)

    # Return the NumPy array containing tumor location points
    return tumor_points

def load_OIB(path):
    filename = os.path.splitext(os.path.basename(path))[0]
    with OifFile(path) as oif:
        viewer = OifImageViewer(oif)
        x_step, y_step, z_step = map(float, (viewer.md["x_step"], viewer.md["y_step"], viewer.md["z_step"]))
        if viewer.md["z_unit"] == "nm":
            z_step /= 1000.
    return filename, viewer, (x_step, y_step, z_step)

def calculate_pvalues(observed_ripleyK, simulated_ripleyKs):
    """
    Calculate p-values for Ripley's K function at each radius.

    Parameters:
    observed_ripleyK (array-like): An array of Ripley's K values calculated from the observed data for each radius.
    simulated_ripleyKs (array-like of array-like): A 2D array where each row represents the Ripley's K values 
                                                    from a single Monte Carlo simulation across the same radii as the observed data.

    Returns:
    np.array: An array of p-values for each radius.
    """
    # Initialize an array to hold p-values for each radius
    p_values = np.zeros_like(observed_ripleyK)

    # Calculate p-value for each radius
    for i in range(len(observed_ripleyK)):
        # For each radius, count how many simulated Ripley's K values are as extreme as or more extreme than the observed value
        extreme_count = np.sum(simulated_ripleyKs[:, i] >= observed_ripleyK[i])
        
        # Calculate the p-value as the proportion of simulations that are as or more extreme than the observed
        p_values[i] = extreme_count / simulated_ripleyKs.shape[0]

    return p_values

def main(FLAGS):
    filename, viewer, steps = load_OIB(FLAGS.oib_path)
    tumor_points = load_tumor_locations(FLAGS.tumor_path, filename, steps)
     
    ng2_points = np.flip(np.load(os.path.join(FLAGS.ng2_path,
                                              f"{filename}_NG2_centroids.npy")).T, axis=1) # put points into Z,Y,X format (N, 3)
    branch_points = np.load(os.path.join(FLAGS.branch_path,
                                         f"{filename}_branch_points.npy")).T # branch points do not need to be flipped to be formatted correctly
    tvc_points = np.rint(np.flip(np.load(os.path.join(FLAGS.tvc_path,
                                                      f"{filename}_tortuous_segment_centroid.npy")).T, axis=1)).astype(int) # put points into Z,Y,X format (N, 3)
    mask = load_mask(os.path.join(FLAGS.mask_path, f"{filename}.seg.nrrd"))

    all_points = {
        "tumor": tumor_points,
        "ng2": ng2_points,
        "branch": branch_points,
        "tvc": tvc_points
    }
    
    radii = np.arange(2, 100) 
    
    ## Run univariate comparisons
    #for name, points in all_points.items():
    #    print(f"Running univariate analyses on: {name} points")
    #    random_u_rstats = monte_carlo(points, mask, radii, n_samples=100, n_processes=55, boundary_correction=False)
    #    u_results = run_ripley(points, points, mask, radii, n_processes=55, boundary_correction=False)
    #    u_rstats = pd.DataFrame(u_results, columns=["Radius (r)", "K(r)", "L(r)", "H(r)"])
    #
    #    # Uncomment to save rstats to csv
    #
    #    random_u_rstats.to_csv(os.path.join(FLAGS.output_dir, f"{filename}_random_univariate_{name}_rstats.csv"))
    #    u_rstats.to_csv(os.path.join(FLAGS.output_dir, f"{filename}_univariate_{name}_rstats.csv"))

    # Run multivariate comparisons with tumor
    if filename != "FV10__20181004_122358":
        for name, points in all_points.items():
            if name != "tumor":
                print(f"Running multivariate analyses between tumor and {name} points")
                random_m1_results = monte_carlo(tumor_points, mask, radii, points, n_samples=100, n_processes=55, boundary_correction=False)
                m1_results = run_ripley(tumor_points, points, mask, radii, n_processes=55, boundary_correction=False)
        
                # Uncomment to save rstats to csv
                random_m1_results.to_csv(os.path.join(FLAGS.output_dir,
                                                     f"{filename}_random_multivariate_tumor_{name}_rstats.csv"))
                m1_results.to_csv(os.path.join(FLAGS.output_dir,
                                              f"{filename}_multivariate_tumor_{name}_rstats.csv"))

    # Run multivariate comparisons with ng2 
    for name, points in all_points.items():
        if name != "tumor" and name != "ng2":
            print(f"Running multivariate analyses between ng2 and {name} points")
            random_m2_results = monte_carlo(ng2_points, mask, radii, points, n_samples=100, n_processes=55, boundary_correction=False)
            m2_results = run_ripley(ng2_points, points, mask, radii, n_processes=55, boundary_correction=False)
    
            # Uncomment to save rstats to csv
            random_m2_results.to_csv(os.path.join(FLAGS.output_dir,
                                                 f"{filename}_random_multivariate_ng2_{name}_rstats.csv"))
            m2_results.to_csv(os.path.join(FLAGS.output_dir,
                                         f"{filename}_multivariate_ng2_{name}_rstats.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oib_path",
        type=str,
        required=True,
        help="path to oib file"
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        required=True,
        help="path to mask file"
    )
    parser.add_argument(
        "--tumor_path",
        type=str,
        required=True,
        help="path to tumor points csv file"
    )
    parser.add_argument(
        "--branch_path",
        type=str,
        required=True,
        help="path to branch points npy file"
    )
    parser.add_argument(
        "--ng2_path",
        type=str,
        required=True,
        help="path to ng2 points npy file"
    )
    parser.add_argument(
        "--tvc_path",
        type=str,
        required=True,
        help="path to tortuous vessel centroid points npy file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/dkermany/ripley_results/",
        help="path to output directory"
    )
    
    FLAGS, _ = parser.parse_known_args()
    main(FLAGS)
