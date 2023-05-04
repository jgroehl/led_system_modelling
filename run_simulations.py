from utils.simulate import simulate
from utils.noise_models import add_noise

DATA_FILE = r"C:\Users\grohl01\Downloads\data\ground_truth.npz"
PATH_TO_NOISE = r"C:\Users\grohl01\Downloads\NoiseMeasurementAllframes.mat"
# path = "C:/tmp_results/baseline_ipasc.hdf5"

path = simulate(DATA_FILE, "baseline")
add_noise(path, PATH_TO_NOISE)

path = simulate(DATA_FILE, "attenuation", model_acoustic_attenuation=True)
add_noise(path, PATH_TO_NOISE)

path = simulate(DATA_FILE, "size", model_detector_size=True)
add_noise(path, PATH_TO_NOISE)

path = simulate(DATA_FILE, "frequencyresponse", model_frequency_response=True)
add_noise(path, PATH_TO_NOISE)

path = simulate(DATA_FILE, "attenuation_size", model_acoustic_attenuation=True,
                model_detector_size=True)
add_noise(path, PATH_TO_NOISE)

path = simulate(DATA_FILE, "attenuation_size_frequencyresponse", model_acoustic_attenuation=True,
                model_detector_size=True, model_frequency_response=True)
add_noise(path, PATH_TO_NOISE)

# We only have to run the optical forward model once per phantom digital twin.
optical_path = simulate(DATA_FILE, "optical", optical_model=True)
add_noise(optical_path, PATH_TO_NOISE)
optical_path_simpa = optical_path.replace("_ipasc", "")

path = simulate(DATA_FILE, "optical_attenuation", optical_model=True, model_acoustic_attenuation=True,
                load_initial_pressure_path=optical_path_simpa)
add_noise(path, PATH_TO_NOISE)

path = simulate(DATA_FILE, "optical_size", optical_model=True, model_detector_size=True,
                load_initial_pressure_path=optical_path_simpa)
add_noise(path, PATH_TO_NOISE)

path = simulate(DATA_FILE, "optical_frequencyresponse", optical_model=True, model_frequency_response=True,
                load_initial_pressure_path=optical_path_simpa)
add_noise(path, PATH_TO_NOISE)

path = simulate(DATA_FILE, "optical_attenuation_size", optical_model=True, model_acoustic_attenuation=True,
                load_initial_pressure_path=optical_path_simpa, model_detector_size=True)
add_noise(path, PATH_TO_NOISE)

path = simulate(DATA_FILE, "optical_attenuation_size_frequencyresponse", optical_model=True,
                model_acoustic_attenuation=True, model_frequency_response=True,
                load_initial_pressure_path=optical_path_simpa, model_detector_size=True)
add_noise(path, PATH_TO_NOISE)
