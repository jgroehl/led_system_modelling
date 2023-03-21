import matplotlib.pyplot as plt
import pacfish as pf
BASE_PATH = "C:/tmp_results/"

plt.figure(figsize=(10, 12))

plt.subplot(4, 2, 1)
plt.title("baseline")
plt.imshow(pf.load_data(BASE_PATH + "baseline_ipasc.hdf5").binary_time_series_data[:, :1024], aspect=8)
plt.colorbar()

plt.subplot(4, 2, 3)
plt.title("+ attenuation")
plt.imshow(pf.load_data(BASE_PATH + "attenuation_ipasc.hdf5").binary_time_series_data[:, :1024], aspect=8)
plt.colorbar()

plt.subplot(4, 2, 5)
plt.title("+ directivity")
plt.imshow(pf.load_data(BASE_PATH + "attenuation_size_ipasc.hdf5").binary_time_series_data[:, :1024], aspect=8)
plt.colorbar()

plt.subplot(4, 2, 7)
plt.title("+ frequency response")
plt.imshow(pf.load_data(BASE_PATH + "attenuation_size_frequencyresponse_ipasc.hdf5").binary_time_series_data[:, :1024], aspect=8)
plt.colorbar()

plt.subplot(4, 2, 2)
plt.title("optical")
plt.imshow(pf.load_data(BASE_PATH + "optical_ipasc.hdf5").binary_time_series_data[:, :1024], aspect=8)
plt.colorbar()

plt.subplot(4, 2, 4)
plt.title("+ attenuation")
plt.imshow(pf.load_data(BASE_PATH + "optical_attenuation_ipasc.hdf5").binary_time_series_data[:, :1024], aspect=8)
plt.colorbar()

plt.subplot(4, 2, 6)
plt.title("+ directivity")
plt.imshow(pf.load_data(BASE_PATH + "optical_attenuation_size_ipasc.hdf5").binary_time_series_data[:, :1024], aspect=8)
plt.colorbar()

plt.subplot(4, 2, 8)
plt.title("+ frequency response")
plt.imshow(pf.load_data(BASE_PATH + "optical_attenuation_size_frequencyresponse_ipasc.hdf5").binary_time_series_data[:, :1024], aspect=8)
plt.colorbar()

plt.tight_layout()
plt.show()
