import pacfish as pf
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import os.path

# Define resulting mean and standard deviation
MEAN = 10
STDDEV = 1

# Define path to sinogram to normalize when this file is run directly
# by the Python interpreter

# simulation
INSPECT_PATH = "../../results/optical_attenuation_size_ipasc.hdf5"


def normalize_sinogram(data_path, experimental=False, mean=MEAN, stddev=STDDEV):
    """
    Normalize a sinogram according to given mean and standard deviation.

    :param str data_path: Path to the data file. This can either be a simulation result (stored as ``.hdf5``) or
                          an experimental result (stored under the ``sinogram`` tag in a ``.mat`` file)
    :param bool experimental: Whether the given data file is an experimental result
    :param float mean: The target mean for the normalized sinogram
    :param float stddev: The target standard deviation for the normalized sinogram

    :return: The normalized data
    """

    if experimental:
        data_raw = loadmat(data_path)
        data = data_raw["sinogram"].T
    else:
        data = pf.load_data(data_path).binary_time_series_data

    # Scale data to right standard deviation
    old_stddev = np.std(data)
    data *= stddev/old_stddev

    # Shift data to right mean value
    old_mean = np.mean(data)
    data += mean - old_mean

    return data


if __name__ == "__main__":
    normalized = normalize_sinogram(os.path.abspath(INSPECT_PATH))

    # Inspect result
    plt.figure()
    plt.imshow(normalized, aspect=8)
    plt.title("Normalized sinogram")
    plt.xlabel("Time step")
    plt.ylabel("Detector element")
    plt.colorbar(label="Pressure [a.u.]")
    plt.show()
