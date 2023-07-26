import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

NOISE = False

# EXPERIMENTAL_PATH = "//ad.utwente.nl/tnw/bmpi/Data/Mirre van der Wal/Data/2023-06-02 coregistered and normalized results/experimental.npy"
EXPERIMENTAL_PATH = "//ad.utwente.nl/TNW/BMPI/Data/Mirre van der Wal/Data/2023-06-09 PSF data/sinogram-5.npy"


def view_freqs(data: np.ndarray, plot_title: str):
    frequency = 40e6                        # Hz
    time_step = 1 / frequency               # s

    num_transducers, num_time_points = np.shape(data)

    fft_frequencies = fftfreq(num_time_points, time_step)[:num_time_points//2] / 1e6    # MHz

    transducer_frequencies = np.zeros((num_transducers, num_time_points//2))
    for transducer_index, transducer_data in enumerate(data):
        transducer_frequencies[transducer_index] = 2 / num_time_points * np.abs(fft(transducer_data)[:num_time_points//2])

    transducer_frequencies -= np.min(transducer_frequencies)
    transducer_frequencies /= np.max(transducer_frequencies)

    plt.figure()
    plt.plot(fft_frequencies, transducer_frequencies[64])
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Intensity [a.u.]")
    plt.title(plot_title)

    # plt.figure()
    # plt.imshow(transducer_frequencies, aspect=fft_frequencies[-1]/num_transducers, extent=[fft_frequencies[0], fft_frequencies[-1], 0, num_transducers], cmap="coolwarm")
    # plt.xlabel("Frequency [MHz]")
    # plt.ylabel("Transducer element")
    # plt.colorbar(label="Normalized intensity [a.u.]")
    # plt.title(plot_title)

    return transducer_frequencies


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

    experiment = np.load(EXPERIMENTAL_PATH)

    save_dir = os.path.abspath(os.path.join(base_dir, path_manager.get_hdf5_file_save_path()))
    fig_dir = "//ad.utwente.nl/tnw/bmpi/data/mirre van der wal/data/2023-06-20 coregistered psf phantom (incl images)/frequencies/"
    # fig_dir = "//ad.utwente.nl/tnw/bmpi/Data/Mirre van der Wal/Data/2023-06-21 veins - improved frequency response (incl images)/frequencies/"

    for optical in [False, True]:
        for (title, filename) in simulations.items():
            optical_str = "optical_" if optical else ""
            full_filename = optical_str + filename + noise_str + "_normalized.npy"
            path = os.path.join(save_dir, full_filename)
            simulation = np.load(path)

            title = ("Optical + " if optical else "") + "Acoustic:\n" + title
            view_freqs(simulation, title)

            plt.savefig(fig_dir + optical_str + filename + noise_str + ".png")


    view_freqs(experiment, "Experimental")
    plt.savefig(fig_dir + "experiment" + ".png")

    # plt.show()
