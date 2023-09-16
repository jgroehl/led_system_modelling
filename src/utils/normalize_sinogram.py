import matplotlib.pyplot as plt
import os.path
from src.utils.file_loading import *

# Define resulting mean and standard deviation
MEAN = 0
STDDEV = 1

# Define path to sinogram to normalize when this file is run directly
# by the Python interpreter
INSPECT_PATH = "../../results/optical_attenuation_size_ipasc.hdf5"


def normalize_sinogram(data, mean=MEAN, stddev=STDDEV):
    """
    Normalize a sinogram according to given mean and standard deviation.

    :param np.ndarray data: The data that we will normalize
    :param float mean: The target mean for the normalized sinogram
    :param float stddev: The target standard deviation for the normalized sinogram

    :return: The normalized data
    """

    # Scale data to right standard deviation
    old_stddev = np.std(data)
    data *= stddev/old_stddev

    # Shift data to right mean value
    old_mean = np.mean(data)
    data += mean - old_mean

    return data


if __name__ == "__main__":
    path = os.path.abspath(INSPECT_PATH)
    data = load_simulation_result(path)
    # data = load_experimental_result(path)

    normalized = normalize_sinogram(data)

    # Inspect result
    plt.figure()
    plt.imshow(normalized, aspect=8)
    plt.title("Normalized sinogram")
    plt.xlabel("Time step")
    plt.ylabel("Detector element")
    plt.colorbar(label="Pressure [a.u.]")
    plt.show()
