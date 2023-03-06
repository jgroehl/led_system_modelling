import simpa as sp
import numpy as np
import matplotlib.pyplot as plt

# CHANGE THIS PATH TO POINT TO THE DESIRED SIMPA OUTPUT
HDF5_PATH = r"D:\ipasc_output\francis2.hdf5"


p0 = sp.load_data_field(HDF5_PATH, sp.Tags.DATA_FIELD_INITIAL_PRESSURE, 800)
p0 = np.log(p0)
x, y, z = p0.shape
spacing = (4/x) * 10

plt.subplot(1, 3, 1)
plt.imshow(p0[:, int(y/2), :], extent=[0, x*spacing, 0, z*spacing])

plt.subplot(1, 3, 2)
plt.imshow(p0[int(x/2), :, :], extent=[0, y*spacing, 0, z*spacing])

plt.subplot(1, 3, 3)
plt.imshow(p0[:, :, int(z/2)], extent=[0, x*spacing, 0, y*spacing])

plt.show()

