import simpa as sp
import uuid
import os.path

from simpa import Settings, Tags
from simpa.utils.tissue_properties import TissueProperties


def generate_base_settings(path_manager: sp.PathManager,
                           speed_of_sound=1480,
                           volume_name=str(uuid.uuid4()),
                           wavelength=800,
                           random_seed=1337) -> Settings:
    """
    This function will return the base settings which are already optimised for
    the IPASC standardised image reconstruction project needs.

    Settings for the respective volume generation module will need to be called
    separately.

    :return: dict
        A `simpa.Settings` instance with optimised entries for the IPASC image
        reconstruction project.
    """

    settings = Settings()
    settings[Tags.SIMULATION_PATH] = os.path.abspath(os.path.join("../", path_manager.get_hdf5_file_save_path()))
    settings[Tags.VOLUME_NAME] = volume_name
    settings[Tags.DIM_VOLUME_X_MM] = 50
    settings[Tags.DIM_VOLUME_Y_MM] = 25
    settings[Tags.DIM_VOLUME_Z_MM] = 50
    settings[Tags.SPACING_MM] = settings[Tags.DIM_VOLUME_X_MM] / 1024 * 2  # 1024 x 512 x 1024 grid is ideal for k-Wave
    settings[Tags.DATA_FIELD_SPEED_OF_SOUND] = speed_of_sound
    settings[Tags.WAVELENGTHS] = [wavelength]
    settings[Tags.RANDOM_SEED] = random_seed
    settings[Tags.GPU] = True
    settings[Tags.DO_FILE_COMPRESSION] = True
    settings[Tags.DO_IPASC_EXPORT] = False  # we use the k-Wave IPASC export

    settings.set_acoustic_settings({
        Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
    })

    settings.set_optical_settings({
        Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e6,
        Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
        Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
        Tags.MCX_ASSUMED_ANISOTROPY: 0.9,
    })

    settings.set_reconstruction_settings({
        Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
        Tags.ACOUSTIC_SIMULATION_3D: False,
        Tags.KWAVE_PROPERTY_ALPHA_POWER: 0.00,
        Tags.KWAVE_PROPERTY_SENSOR_RECORD: "p",
        Tags.KWAVE_PROPERTY_PMLInside: False,
        Tags.KWAVE_PROPERTY_PMLSize: [31, 32],
        Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
        Tags.KWAVE_PROPERTY_PlotPML: False,
        Tags.RECORDMOVIE: False,
        Tags.MOVIENAME: "visualization_log",
        Tags.ACOUSTIC_LOG_SCALE: True,
        Tags.DATA_FIELD_SPEED_OF_SOUND: speed_of_sound,
        Tags.DATA_FIELD_ALPHA_COEFF: 0.01,
        Tags.DATA_FIELD_DENSITY: 1000,
        Tags.SPACING_MM: settings[Tags.SPACING_MM],
        Tags.SENSOR_SAMPLING_RATE_MHZ: 40
    })

    settings["FieldOfViewCropping"] = Settings({
        Tags.DATA_FIELD: TissueProperties.property_tags})

    return settings