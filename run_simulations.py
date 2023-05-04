import os.path

from utils.simulate import simulate
from utils.noise_models import add_noise

# Define relative paths to data and noise files
DATA_FILE = "phantoms/ground_truth.npz"
NOISE_FILE = "resources/NoiseMeasurement.mat"

# Construct absolute paths from the relative paths defined above
abs_data_file = os.path.abspath(DATA_FILE)
abs_noise_file = os.path.abspath(NOISE_FILE)

path = simulate(abs_data_file, "baseline")
add_noise(path, abs_noise_file)

path = simulate(abs_data_file, "attenuation", model_acoustic_attenuation=True)
add_noise(path, abs_noise_file)

path = simulate(abs_data_file, "size", model_detector_size=True)
add_noise(path, abs_noise_file)

path = simulate(abs_data_file, "frequencyresponse", model_frequency_response=True)
add_noise(path, abs_noise_file)

path = simulate(abs_data_file, "attenuation_size", model_acoustic_attenuation=True,
                model_detector_size=True)
add_noise(path, abs_noise_file)

path = simulate(abs_data_file, "attenuation_size_frequencyresponse", model_acoustic_attenuation=True,
                model_detector_size=True, model_frequency_response=True)
add_noise(path, abs_noise_file)

# We only have to run the optical forward model once per phantom digital twin.
optical_path = simulate(abs_data_file, "optical", optical_model=True)
add_noise(optical_path, abs_noise_file)
optical_path_simpa = optical_path.replace("_ipasc", "")

path = simulate(abs_data_file, "optical_attenuation", optical_model=True, model_acoustic_attenuation=True,
                load_initial_pressure_path=optical_path_simpa)
add_noise(path, abs_noise_file)

path = simulate(abs_data_file, "optical_size", optical_model=True, model_detector_size=True,
                load_initial_pressure_path=optical_path_simpa)
add_noise(path, abs_noise_file)

path = simulate(abs_data_file, "optical_frequencyresponse", optical_model=True, model_frequency_response=True,
                load_initial_pressure_path=optical_path_simpa)
add_noise(path, abs_noise_file)

path = simulate(abs_data_file, "optical_attenuation_size", optical_model=True, model_acoustic_attenuation=True,
                load_initial_pressure_path=optical_path_simpa, model_detector_size=True)
add_noise(path, abs_noise_file)

path = simulate(abs_data_file, "optical_attenuation_size_frequencyresponse", optical_model=True,
                model_acoustic_attenuation=True, model_frequency_response=True,
                load_initial_pressure_path=optical_path_simpa, model_detector_size=True)
add_noise(path, abs_noise_file)
