import numpy as np
import pacfish as pf
from scipy.io import loadmat


def load_experimental_result(path: str) -> np.ndarray:
    data = loadmat(path)
    return data["sinogram"].T


def load_simulation_result(path: str) -> np.ndarray:
    data = pf.load_data(path)
    return data.binary_time_series_data
    
