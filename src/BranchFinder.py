import multiprocessing as mp
import cupy as cp
import numpy as np
from cupyimg.scipy.ndimage.morphology import binary_hit_or_miss
from tqdm import tqdm
from src.kernels import get_branch_kernels

class BranchFinder:
    def __init__(self, skeleton):
        self.skeleton = skeleton
        self.kernels = get_branch_kernels()

    def run(self, processes=1):
        with mp.Manager() as m:
            self.z, self.x, self.y = m.list(), m.list(), m.list()
            with mp.Pool(processes) as pool:
                list(tqdm(pool.imap(self._find_branches, self.kernels), total=len(self.kernels)))

    def _find_branches(self, kernel):
        kernel = cp.asarray(kernel)
        z, y, x = cp.nonzero(binary_hit_or_miss(self.skeleton, structure1=kernel).astype(int))
        # self.z.extend(z)
        # self.y.extend(y)
        # self.x.extend(x)

        # print(z, flush=True)
        # print(type(z), flush=True)
    #branch_pts_img = cp.asnumpy(bf.run().astype(np.uint8) * 255)



if __name__ == "__main__":
    print("testing BranchFinder.py")