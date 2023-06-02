import numpy as np

SIMULATION_PATH = "../../results/optical_baseline_noise_ipasc.hdf5"
EXPERIMENTAL_PATH = "/mnt/bmpi/Data/Mirre van der Wal/Data/2023-05-30 experimental reconstruction/experimental_sinogram.mat"


def error_map(result: np.ndarray, expectation: np.ndarray, plot=False) -> np.ndarray:
    error = result - expectation

    if plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.title("Error map")
        plt.imshow(error, aspect=8)
        plt.xlabel("Time step [ ]")
        plt.ylabel("Detector index [ ]")
        plt.colorbar(label="Pressure error [a.u.]")

        plt.show()

    return error


if __name__ == "__main__":
    import pacfish as pf
    from scipy.io import loadmat

    simulation = pf.load_data(SIMULATION_PATH).binary_time_series_data
    experiment_raw = loadmat(EXPERIMENTAL_PATH)
    experiment = experiment_raw["sinogram"].T

    error_map(simulation, experiment, plot=True)
