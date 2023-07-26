import numpy as np
from skimage.metrics import structural_similarity

EXPERIMENTAL_PATH = "/mnt/bmpi/Data/Mirre van der Wal/Data/2023-06-02 coregistered and normalized results/experimental.npy"
from src.visualisation.show_frequencies import view_freqs


NOISE = True


def calculate_ssim(result: np.ndarray, expectation: np.ndarray) -> float:
    return structural_similarity(result, expectation, data_range=expectation.max() - expectation.min(), full=False)


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
    # from scipy.io import loadmat
    # data = loadmat(EXPERIMENTAL_PATH)
    # experiment = data["P"]["sinrgm"][0][0][:, :, 4]

    save_dir = os.path.abspath(os.path.join(base_dir, path_manager.get_hdf5_file_save_path()))

    for optical in [False, True]:
        for (title, filename) in simulations.items():
            optical_str = "optical_" if optical else ""
            full_filename = optical_str + filename + noise_str + "_normalized.npy"
            path = os.path.join(save_dir, full_filename)
            simulation = np.load(path)

            # ssim = calculate_ssim(simulation, experiment)

            simulation_freqs = view_freqs(simulation, "")
            experiment_freqs = view_freqs(experiment, "")
            ssim = calculate_ssim(simulation_freqs, experiment_freqs)

            title = ("Optical + " if optical else "") + "Acoustic: " + title

            print(f"Structural similarity ({title}): ", ssim)
