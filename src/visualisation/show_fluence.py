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
    fluence = sp.load_data_field(data_path, sp.Tags.DATA_FIELD_INITIAL_PRESSURE, wavelength)
    # fluence = sp.load_data_field(data_path, sp.Tags.DATA_FIELD_GRUNEISEN_PARAMETER, wavelength)
    # sp.Tags.DA
    x, y, z = fluence.shape
    spacing = 50 / x # assuming a 50 mm sample size in x-direction

    fluence /= np.max(fluence[:, int(y/2), :])

    # Show fluence at y = y_max / 2 (sample plane)
    plt.figure()
    plt.imshow(fluence[:, int(y/2), :].T, extent=[0, x*spacing, z*spacing, 0], cmap="coolwarm")
    plt.xlabel("x-position [mm]")
    plt.ylabel("z-position [mm]")
    plt.colorbar(label=r"$p_0$ (normalized) [~$Pa$]")
    plt.title("Initial pressure")

    # Show absorption at x = x_max / 2 (side view of sample)
    # plt.figure()
    # plt.imshow(p0[int(x/2), :, :].T, extent=[0, y*spacing, z*spacing, 0])
    # plt.xlabel("y-position [mm]")
    # plt.ylabel("z-position [mm]")
    # plt.colorbar(label="Absorbed energy [a.u.]")
    # plt.title("Absorption map (side view)")

    plt.show()


if __name__ == "__main__":
    base_dir = os.path.join(os.path.curdir, "../../")
    path_manager = sp.PathManager("../path_config.env")
    #
    # data_path = "//ad.utwente.nl/TNW/BMPI/Data/Mirre van der Wal/Data/2023-05-17 first simulation of 'veins' phantom/results/optical.hdf5"
    # show_absorption(data_path, wavelength=800)

    # data_path = os.path.join(base_dir, path_manager.get_hdf5_file_save_path(), "optical_baseline.hdf5")
    data_path = "//ad.utwente.nl/tnw/bmpi/data/mirre van der wal/data/2023-06-19 coregistered vein phantom (incl images)/results/optical_baseline.hdf5"
    # data_path = "//ad.utwente.nl/tnw/bmpi/data/mirre van der wal/data/2023-06-20 coregistered psf phantom (incl images)/results/optical_baseline.hdf5"
    # data_path = r"\\ad.utwente.nl\TNW\BMPI\Data\Mirre van der Wal\Data\2023-06-25 Fluence visualisation/optical_baseline.hdf5"
    show_absorption(data_path, wavelength=850)

