import numpy as np
from scipy.io import loadmat


def load_hdf5_file(path: str) -> np.ndarray:
    data = loadmat(path)
    return data["Coreg_GT"]