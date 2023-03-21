import pacfish as pf
import numpy as np


def add_noise(path_to_ipasc_hdf5: str):
    pa_data = pf.load_data(path_to_ipasc_hdf5)

    ts = pa_data.binary_time_series_data.copy()

    ts_noise = ts.copy() + np.random.normal(0, 10, size=np.shape(ts))
    pa_data.binary_time_series_data = ts_noise
    pf.write_data(path_to_ipasc_hdf5.replace("_ipasc", "_noise1_ipasc"), pa_data)

    ts_noise_2 = ts.copy() * np.random.normal(1, 0.2, size=np.shape(ts))
    pa_data.binary_time_series_data = ts_noise_2
    pf.write_data(path_to_ipasc_hdf5.replace("_ipasc", "_noise2_ipasc"), pa_data)
