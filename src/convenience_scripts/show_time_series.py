import matplotlib.pyplot as plt
import pacfish as pf
import simpa as sp
import os.path

base_dir = os.path.join(os.path.curdir, "../../")
path_manager = sp.PathManager("../path_config.env")
BASE_PATH = os.path.abspath(os.path.join(base_dir, path_manager.get_hdf5_file_save_path()))

plt.figure(figsize=(10, 12))

plt.subplot(4, 2, 1)
plt.title("baseline")
plt.imshow(pf.load_data(os.path.join(BASE_PATH, "baseline_ipasc.hdf5")).binary_time_series_data[:, :1024], aspect=8)
plt.colorbar()

plt.subplot(4, 2, 3)
plt.title("+ attenuation")
plt.imshow(pf.load_data(os.path.join(BASE_PATH, "attenuation_ipasc.hdf5")).binary_time_series_data[:, :1024], aspect=8)
plt.colorbar()

plt.subplot(4, 2, 5)
plt.title("+ directivity")
plt.imshow(pf.load_data(os.path.join(BASE_PATH, "attenuation_size_ipasc.hdf5")).binary_time_series_data[:, :1024], aspect=8)
plt.colorbar()

plt.subplot(4, 2, 7)
plt.title("+ frequency response")
plt.imshow(pf.load_data(os.path.join(BASE_PATH, "attenuation_size_frequencyresponse_ipasc.hdf5")).binary_time_series_data[:, :1024], aspect=8)
plt.colorbar()

plt.subplot(4, 2, 2)
plt.title("optical")
plt.imshow(pf.load_data(os.path.join(BASE_PATH, "optical_ipasc.hdf5")).binary_time_series_data[:, :1024], aspect=8)
plt.colorbar()

plt.subplot(4, 2, 4)
plt.title("+ attenuation")
plt.imshow(pf.load_data(os.path.join(BASE_PATH, "optical_attenuation_ipasc.hdf5")).binary_time_series_data[:, :1024], aspect=8)
plt.colorbar()

plt.subplot(4, 2, 6)
plt.title("+ directivity")
plt.imshow(pf.load_data(os.path.join(BASE_PATH, "optical_attenuation_size_ipasc.hdf5")).binary_time_series_data[:, :1024], aspect=8)
plt.colorbar()

plt.subplot(4, 2, 8)
plt.title("+ frequency response")
plt.imshow(pf.load_data(os.path.join(BASE_PATH, "optical_attenuation_size_frequencyresponse_ipasc.hdf5")).binary_time_series_data[:, :1024], aspect=8)
plt.colorbar()

plt.tight_layout()
plt.show()
