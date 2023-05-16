import simpa as sp
import numpy as np
import matplotlib.pyplot as plt
import os.path

base_dir = os.path.join(os.path.curdir, "../")
path_manager = sp.PathManager("../path_config.env")

HDF5_PATH = os.path.join(base_dir, path_manager.get_hdf5_file_save_path(), "optical.hdf5")

p0 = sp.load_data_field(HDF5_PATH, sp.Tags.DATA_FIELD_INITIAL_PRESSURE, 800)
# p0 = np.log(p0)
x, y, z = p0.shape
print("Simulation dimensions:\n", x, y, z)
spacing = (5/x) * 10

plt.figure()
plt.imshow(p0[:, int(y/2), :].T, extent=[0, x*spacing, z*spacing, 0])
plt.xlabel("x-position [a.u.]")
plt.ylabel("z-position [a.u.]")
plt.colorbar(label="Absorption [a.u.]")
plt.title("Absorption map")

plt.figure()
plt.imshow(p0[int(x/2), :, :].T, extent=[0, y*spacing, z*spacing, 0])
plt.xlabel("y-position [a.u.]")
plt.ylabel("z-position [a.u.]")
plt.colorbar(label="Absorption [a.u.]")
plt.title("Absorption map (side view)")

plt.show()

