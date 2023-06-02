import numpy as np

SIMULATION_PATH = "../../results/optical_baseline_noise_ipasc.hdf5"
EXPERIMENTAL_PATH = "/mnt/bmpi/Data/Mirre van der Wal/Data/2023-05-30 experimental reconstruction/experimental_sinogram.mat"


def calculate_mse(result: np.ndarray, expectation: np.ndarray) -> float:
    return np.mean(np.square(result - expectation))


if __name__ == "__main__":
    import simpa as sp
    import os.path

    calculate_mse(SIMULATION_PATH, EXPERIMENTAL_PATH)
