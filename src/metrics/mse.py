import numpy as np

SIMULATION_PATH = "../../results/optical_baseline_noise_ipasc.hdf5"
EXPERIMENTAL_PATH = "/mnt/bmpi/Data/Mirre van der Wal/Data/2023-05-30 experimental reconstruction/experimental_sinogram.mat"


def calculate_mse(result: np.ndarray, expectation: np.ndarray) -> float:
    return np.mean(np.square(result - expectation))


if __name__ == "__main__":
    import pacfish as pf
    from scipy.io import loadmat

    simulation = pf.load_data(SIMULATION_PATH).binary_time_series_data
    experiment_raw = loadmat(EXPERIMENTAL_PATH)
    experiment = experiment_raw["sinogram"].T

    mse = calculate_mse(simulation, experiment)
    print("Mean squared error: ", mse)
