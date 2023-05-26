import matplotlib.pyplot as plt
import pacfish as pf

NOISE = True


def show_sinogram(plot_title, data_path, show=False):
    """
    Show a sinogram (pressure data over time for all detectors).
    Plot can be shown immediately when using show=True, but this will cause the code to stop running and wait for the
    figure to be closed before continuing.

    :param str plot_title: The title to use for the sinogram plot
    :param str data_path: The absolute path to the data file
    :param bool show: Whether to immediately show the result
    """
    data = pf.load_data(data_path).binary_time_series_data

    plt.figure()
    plt.title(plot_title)
    plt.imshow(data, aspect=8)
    plt.xlabel("Time step [ ]")
    plt.ylabel("Detector index [ ]")
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

    for optical in [False, True]:
        optical_str = "optical_" if optical else ""
        for (title, filename) in simulations.items():
            if optical:
                title = "Optical + Acoustic:\n" + title
            else:
                title = "Acoustic:\n" + title
            save_dir = os.path.abspath(os.path.join(base_dir, path_manager.get_hdf5_file_save_path()))
            full_filename = optical_str + filename + noise_str + "_ipasc.hdf5"
            path = os.path.join(save_dir, full_filename)

            show_sinogram(title, path)

    plt.show()
