import pacfish as pf
import matplotlib.pyplot as plt
import numpy as np
import os.path

# Define resulting mean and standard deviation
MEAN = 10
STDDEV = 1

# Define path to sinogram to normalize when this file is run directly
# by the Python interpreter
INSPECT_PATH = "/tmp/results/sinogram.hdf5"


def normalize_sinogram(data_path):
    data = pf.load_data(data_path).binary_time_series_data

    # Scale data to right standard deviation
    stddev = np.std(data)
    data *= STDDEV/stddev

    # Shift data to right mean value
    mean = np.mean(data)
    data += MEAN - mean

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
