import matplotlib.pyplot as plt
import numpy as np

NOISE = False


def show_sinogram(plot_title, data, show=False):
    """
    Show a sinogram (pressure data over time for all detectors).
    Plot can be shown immediately when using show=True, but this will cause the code to stop running and wait for the
    figure to be closed before continuing.

    :param str plot_title: The title to use for the sinogram plot
    :param np.ndarray data: The pressure data
    :param bool show: Whether to immediately show the result
    """

    Nx, Nt = np.shape(data)
    xlim = Nt / 40e6

    plt.figure()
    plt.title(plot_title)
    plt.imshow(data, aspect=xlim/Nx, extent=[0, xlim, Nx, 0], cmap="coolwarm", vmin=-15, vmax=15)
    plt.xlabel("Time [s]")
    plt.ylabel("Detector index")
    plt.colorbar(label="Pressure [a.u.]")

    if show:
        plt.show()


if __name__ == "__main__":
    import simpa as sp
    import os.path

    path_manager = sp.PathManager("../path_config.env")
    base_dir = "../../"

    simulations = {
        "baseline": "baseline",
        "acoustic attenuation": "attenuation",
        "detector size": "size",
        "frequency response": "frequencyresponse",
        "acoustic attenuation + detector size": "attenuation_size",
        "acoustic attenuation + detector size + frequency response": "attenuation_size_frequencyresponse",
    }

    noise_str = "_noise" if NOISE else ""

    # fig_dir = "//ad.utwente.nl/tnw/bmpi/data/mirre van der wal/data/2023-06-21 veins - improved frequency response (incl images)/sinograms/"
    # fig_dir = "//ad.utwente.nl/tnw/bmpi/data/mirre van der wal/data/2023-06-20 coregistered psf phantom (incl images)/sinograms/"
    fig_dir = "//ad.utwente.nl/TNW/BMPI/Data/Mirre van der Wal/Data/2023-06-26 3d vascular phantom (Francis' request)/3d/sinograms/"

    for optical in [False, True]:
        optical_str = "optical_" if optical else ""
        for (title, filename) in simulations.items():
            if optical:
                title = "Optical + Acoustic:\n" + title
            else:
                title = "Acoustic:\n" + title

            # save_dir = os.path.abspath(os.path.join(base_dir, path_manager.get_hdf5_file_save_path()))
            save_dir = "//ad.utwente.nl/TNW/BMPI/Data/Mirre van der Wal/Data/2023-06-26 3d vascular phantom (Francis' request)/3d/results/"
            full_filename = optical_str + filename + noise_str + "_normalized.npy"

            path = os.path.join(save_dir, full_filename)
            data = np.load(path)

            show_sinogram(title, data)
            plt.savefig(fig_dir + optical_str + filename + noise_str + ".png")

    experimental_path = "//ad.utwente.nl/tnw/bmpi/Data/Mirre van der Wal/Data/2023-06-02 coregistered and normalized results/experimental.npy"
    # experimental_path = "//ad.utwente.nl/tnw/bmpi/Data/Mirre van der Wal/Data/2023-06-09 PSF data/sinogram-5.npy"
    experiment = np.load(experimental_path)
    #
    from src.utils.normalize_sinogram import normalize_sinogram
    experiment = normalize_sinogram(experiment)
    #
    show_sinogram("Experimental", experiment)
    plt.savefig(fig_dir + "experimental.png")

    # plt.show()
