import pacfish as pf
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def add_noise(path_to_ipasc_hdf5: str, path_to_noise_mat:str):
    pa_data = pf.load_data(path_to_ipasc_hdf5)
    noise_data = loadmat(path_to_noise_mat)

    normal_noise = noise_data["NoiseSinogram"]
    mean_noise = np.swapaxes(np.mean(normal_noise, axis=2), 0, 1)
    std_noise = np.swapaxes(np.std(normal_noise, axis=2), 0, 1)

    ts = pa_data.binary_time_series_data.copy()

    # TODO SCALE THE SIMULATED TIME SERIES BY A FACTOR TO BE DETERMINED FROM A CALIBRATION PHANTOM
    # TODO THIS SCALING WILL HAVE TO BE DETERMINED FOR EACH SIMULATION PIPELINE INDIVIDUALLY
    ts = ts * 250000

    print(np.shape(ts))

    ts_noise = ts.copy() + np.random.normal(mean_noise, std_noise)

    # TODO APPLY THE SIGMOID POST_PROCESSING EXACTLY HOW FRANCIS APPLIES IT TO THE DATA, TOO

    pa_data.binary_time_series_data = ts_noise
    pf.write_data(path_to_ipasc_hdf5.replace("_ipasc", "_noise_ipasc"), pa_data)
