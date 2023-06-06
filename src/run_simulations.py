import os.path

from utils.simulate import simulate
from utils.noise_models import add_noise

# Define paths to data and noise files
RESOURCES_DIR = "../resources/"
DATA_FILE = "veins-coregistered.npz"    # located in RESOURCES_DIR/phantoms/
NOISE_FILE = "NoiseMeasurement.mat"     # located in RESOURCES_DIR/


def run_simulations(data_file: str, noise_file: str) -> None:
    """
    Run all photoacoustic simulation pipelines on a phantom with a given noise file.
    This will run acoustic simulations, as well as a photoacoustic simulations.
    Both will get simulated multiple times, with the following additional features:
    - None (baseline)
    - Acoustic attenuation
    - Detector size
    - Frequency response
    - Acoustic attenuation + detector size
    - Acoustic attenuation + detector size + frequency response
    It will also apply noise based on a specified noise file over all simulations.

    :param str data_file: The absolute path to the phantom 'ground truth' file (.npz file)
    :param str noise_file: The absolute path to the noise measurement (.mat file)
    """

    # Acoustic simulation
    path = simulate(data_file, "baseline")
    add_noise(path, noise_file)

    # Acoustic simulation + acoustic attenuation
    path = simulate(data_file, "attenuation", model_acoustic_attenuation=True)
    add_noise(path, noise_file)

    # Acoustic simulation + detector size
    path = simulate(data_file, "size", model_detector_size=True)
    add_noise(path, noise_file)

    # Acoustic simulation + frequency response
    path = simulate(data_file, "frequencyresponse", model_frequency_response=True)
    add_noise(path, noise_file)

    # Acoustic simulation + acoustic attenuation + detector size
    path = simulate(data_file, "attenuation_size", model_acoustic_attenuation=True,
                    model_detector_size=True)
    add_noise(path, noise_file)

    # Acoustic simulation + acoustic attenuation + detector size + frequency response
    path = simulate(data_file, "attenuation_size_frequencyresponse", model_acoustic_attenuation=True,
                    model_detector_size=True, model_frequency_response=True)
    add_noise(path, noise_file)

    # Run optical forward model
    optical_path = simulate(data_file, "optical_baseline", optical_model=True)
    add_noise(optical_path, noise_file)
    optical_path_simpa = optical_path.replace("_ipasc", "")

    # Optical & acoustic simulation + acoustic attenuation
    path = simulate(data_file, "optical_attenuation", optical_model=True, model_acoustic_attenuation=True,
                    load_initial_pressure_path=optical_path_simpa)
    add_noise(path, noise_file)

    # Optical & acoustic simulation + detector size
    path = simulate(data_file, "optical_size", optical_model=True, model_detector_size=True,
                    load_initial_pressure_path=optical_path_simpa)
    add_noise(path, noise_file)

    # Optical & acoustic simulation + frequency response
    path = simulate(data_file, "optical_frequencyresponse", optical_model=True, model_frequency_response=True,
                    load_initial_pressure_path=optical_path_simpa)
    add_noise(path, noise_file)

    # Optical & acoustic simulation + acoustic attenuation + detector size
    path = simulate(data_file, "optical_attenuation_size", optical_model=True, model_acoustic_attenuation=True,
                    load_initial_pressure_path=optical_path_simpa, model_detector_size=True)
    add_noise(path, noise_file)

    # Optical & acoustic simulation + acoustic attenuation + detector size + frequency response
    path = simulate(data_file, "optical_attenuation_size_frequencyresponse", optical_model=True,
                    model_acoustic_attenuation=True, model_frequency_response=True,
                    load_initial_pressure_path=optical_path_simpa, model_detector_size=True)
    add_noise(path, noise_file)


if __name__ == "__main__":
    # Construct absolute paths from the relative paths defined above
    resources_dir = os.path.abspath(RESOURCES_DIR)

    data_path = os.path.join(resources_dir, "phantoms", DATA_FILE)
    noise_path = os.path.join(resources_dir, NOISE_FILE)

    run_simulations(data_path, noise_path)
