import simpa as sp
import numpy as np
import matplotlib.pyplot as plt
import os.path

def show_absorption(data_path: str, wavelength=800):
    """
    Show the absorption map of a simulation.

    :param str data_path: The path to the optical simulation result
    """

    # Load data
    p0 = sp.load_data_field(data_path, sp.Tags.DATA_FIELD_INITIAL_PRESSURE, wavelength)
    x, y, z = p0.shape
    spacing = 50 / x # assuming a 50 mm sample size in x-direction

    # Show absorption at y = y_max / 2 (sample plane)
    plt.figure()
    plt.imshow(p0[:, int(y/2), :].T, extent=[0, x*spacing, z*spacing, 0])
    plt.xlabel("x-position [mm]")
    plt.ylabel("z-position [mm]")
    plt.colorbar(label="Absorption [a.u.]")
    plt.title("Absorption map")

    # Show absorption at x = x_max / 2 (side view of sample)
    plt.figure()
    plt.imshow(p0[int(x/2), :, :].T, extent=[0, y*spacing, z*spacing, 0])
    plt.xlabel("y-position [mm]")
    plt.ylabel("z-position [mm]")
    plt.colorbar(label="Absorption [a.u.]")
    plt.title("Absorption map (side view)")

    plt.show()


if __name__ == "__main__":
    base_dir = os.path.join(os.path.curdir, "../../")
    path_manager = sp.PathManager("../path_config.env")

    data_path = os.path.join(base_dir, path_manager.get_hdf5_file_save_path(), "optical.hdf5")
    show_absorption(data_path, wavelength=850)

