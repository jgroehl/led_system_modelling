import simpa as sp
import numpy as np
import matplotlib.pyplot as plt

# CHANGE THIS PATH TO POINT TO THE DESIRED SIMPA OUTPUT
HDF5_PATH = r"D:\ipasc_output\francis2.hdf5"


p0 = sp.load_data_field(HDF5_PATH, sp.Tags.DATA_FIELD_INITIAL_PRESSURE, 800)
p0 = np.log(p0)
x, y, z = p0.shape
print(x, y, z)
spacing = (5/x) * 10

plt.subplot(1, 2, 1)
plt.imshow(p0[:, int(y/2), :].T, extent=[0, x*spacing, z*spacing, 0])

plt.subplot(1, 2, 2)
plt.imshow(p0[int(x/2), :, :].T, extent=[0, y*spacing, z*spacing, 0])

plt.show()

