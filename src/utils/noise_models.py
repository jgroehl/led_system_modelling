import os.path

import pacfish as pf
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

SNR = 2


def add_noise(path_to_ipasc_hdf5: str, path_to_noise_mat: str, snr: int=SNR):
    """
    Add noise to a simulated sinogram based on measured noise from the system.

    :param str path_to_ipasc_hdf5: Path to the simulated sinogram without noise (.hdf5)
    :param str path_to_noise_mat: Path to the noise file (.mat)
    :param int snr: Signal to noise ratio used to scale the noise
    """

    sinogram = pf.load_data(path_to_ipasc_hdf5)
    noise_data = loadmat(path_to_noise_mat)

    normal_noise = noise_data["NoiseSinogram"]
    mean_local_noise = np.swapaxes(np.mean(normal_noise, axis=2), 0, 1)
    std_local_noise = np.swapaxes(np.std(normal_noise, axis=2), 0, 1)

    mean_data = np.mean(np.abs(sinogram.binary_time_series_data))
    mean_noise = np.mean(np.abs(mean_local_noise))

    scaled_noise = np.random.normal(mean_local_noise, std_local_noise) / mean_noise * mean_data / SNR

    # TODO APPLY THE SIGMOID POST_PROCESSING EXACTLY HOW FRANCIS APPLIES IT TO THE DATA, TOO

    sinogram.binary_time_series_data += scaled_noise
    pf.write_data(path_to_ipasc_hdf5.replace("_ipasc", "_noise_ipasc"), sinogram)


if __name__ == "__main__":
    data_path = "../../results/optical_attenuation_size_ipasc.hdf5"
    noise_path = "../../resources/NoiseMeasurement.mat"
    add_noise(os.path.abspath(data_path), os.path.abspath(noise_path))
