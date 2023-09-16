import os.path

import simpa as sp
import pacfish as pf
import numpy as np
from scipy.io import loadmat
from src.utils.normalize_sinogram import normalize_sinogram

SNR = 2


def add_noise(path_to_ipasc_hdf5: str, path_to_noise_mat: str, snr: int = SNR, path_manager_path: str = ""):
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

    scaled_noise = np.random.normal(mean_local_noise, std_local_noise) / mean_noise * mean_data / snr

    # TODO APPLY THE SIGMOID POST_PROCESSING EXACTLY HOW FRANCIS APPLIES IT TO THE DATA, TOO

    sinogram.binary_time_series_data += scaled_noise
    new_data_path = path_to_ipasc_hdf5.replace("_ipasc", "_noise_ipasc")
    print("New data path: ", new_data_path)
    pf.write_data(new_data_path, sinogram)

    # Run normalization
    path_manager = sp.PathManager(path_manager_path)
    data_name = new_data_path.split("/")[-1].replace("_ipasc.hdf5", "")
    normalized_sinogram = normalize_sinogram(new_data_path)
    save_path = os.path.abspath(
        os.path.join("../", path_manager.get_hdf5_file_save_path(), f"{data_name}_normalized.npy"))
    np.save(save_path, normalized_sinogram)


if __name__ == "__main__":
    data_path = "../../results/optical_attenuation_size_ipasc.hdf5"
    noise_path = "../../resources/NoiseMeasurement.mat"
    add_noise(os.path.abspath(data_path), os.path.abspath(noise_path), path_manager_path="../path_config.env")
