# SPDX-FileCopyrightText: 2023 Francis Kalloor Joseph
# SPDX-FileCopyrightText: 2023 Janek Gröhl
# SPDX-License-Identifier: MIT

import simpa as sp
from simpa import Tags
import numpy as np
from utils.settings import generate_base_settings
from utils.ipasc_simpa_kwave_adapter import IpascSimpaKWaveAdapter
from scipy.ndimage import zoom
from utils.cyberdyne_led_array_system import CyberdyneLEDArraySystem


def simulate(data_path, data_name,
             optical_model=False,
             model_detector_size=False,
             model_acoustic_attenuation=False,
             model_frequency_response=False,
             load_initial_pressure_path=None):

    path_manager = sp.PathManager()
    settings = generate_base_settings(path_manager, volume_name=data_name)

    # extract information on geometry and spacing for the purposes of volume creation and device definition
    dim_x_mm = settings[Tags.DIM_VOLUME_X_MM]
    dim_y_mm = settings[Tags.DIM_VOLUME_Y_MM]
    dim_z_mm = settings[Tags.DIM_VOLUME_Z_MM]
    spacing = settings[Tags.SPACING_MM]


    # ###################################################################################
    # VOLUME CREATION
    # Using the SIMPA volume creation module with the segmentation based volume creator
    # ###################################################################################

    sizes = np.round(np.asarray([dim_x_mm, dim_y_mm, dim_z_mm]) / spacing).astype(int)
    sx, _, sz = sizes
    label_volume = np.zeros(sizes)

    # Load the label mask from the ground truth data
    label_mask = np.load(data_path)["gt"].T

    # scale the label mask based on the difference in spacing between the input and output
    input_spacing = 0.078125
    label_mask = np.round(zoom(label_mask, input_spacing/spacing, order=0)).astype(int)
    mx, mz = np.shape(label_mask)
    dx = int((sx - mx) / 2)
    dz = int((sz - mz) / 2)
    label_volume[dx:-dx, int(sizes[1]/2), dz:-dz] = label_mask

    # Define a segmentation mapping to assign optical properties to the background and the structures
    def segmentation_class_mapping():
        if model_acoustic_attenuation:
            alpha = sp.StandardProperties.ALPHA_COEFF_WATER
        else:
            alpha = 0.0
        ret_dict = dict()
        ret_dict[1] = (sp.MolecularCompositionGenerator()
                       .append(sp.Molecule(name="structure",
                                           absorption_spectrum=sp.AbsorptionSpectrumLibrary.CONSTANT_ABSORBER_ARBITRARY(1.0),
                                           volume_fraction=1.0,
                                           scattering_spectrum=sp.ScatteringSpectrumLibrary.CONSTANT_SCATTERING_ARBITRARY(
                                                sp.StandardProperties.WATER_MUS),
                                           anisotropy_spectrum=sp.AnisotropySpectrumLibrary.CONSTANT_ANISOTROPY_ARBITRARY(
                                                sp.StandardProperties.WATER_G),
                                           density=sp.StandardProperties.DENSITY_WATER,
                                           speed_of_sound=sp.StandardProperties.SPEED_OF_SOUND_WATER,
                                           alpha_coefficient=alpha
                            ))
                       .get_molecular_composition(sp.SegmentationClasses.BLOOD))
        ret_dict[0] = (sp.MolecularCompositionGenerator()
                       .append(sp.Molecule(name="water",
                                           absorption_spectrum=sp.AbsorptionSpectrumLibrary().CONSTANT_ABSORBER_ARBITRARY(0.0),
                                           volume_fraction=1.0,
                                           scattering_spectrum=sp.ScatteringSpectrumLibrary.CONSTANT_SCATTERING_ARBITRARY(
                                                sp.StandardProperties.WATER_MUS),
                                           anisotropy_spectrum=sp.AnisotropySpectrumLibrary.CONSTANT_ANISOTROPY_ARBITRARY(
                                                sp.StandardProperties.WATER_G),
                                           density=sp.StandardProperties.DENSITY_WATER,
                                           speed_of_sound=sp.StandardProperties.SPEED_OF_SOUND_WATER,
                                           alpha_coefficient=alpha
                        ))
                       .get_molecular_composition(sp.SegmentationClasses.WATER))
        return ret_dict

    settings.set_volume_creation_settings({
        Tags.INPUT_SEGMENTATION_VOLUME: label_volume,
        Tags.SEGMENTATION_CLASS_MAPPING: segmentation_class_mapping(),

    })
    acoustic_settings = settings.get_acoustic_settings()
    acoustic_settings["frequency_response"] = model_frequency_response
    acoustic_settings["detector_size"] = model_detector_size

    if optical_model:
        # For this simulation: Use the created absorption map as the input initial pressure
        acoustic_settings[Tags.DATA_FIELD] = Tags.DATA_FIELD_INITIAL_PRESSURE
        if load_initial_pressure_path is not None:
            initial_pressure = sp.load_data_field(load_initial_pressure_path, sp.Tags.DATA_FIELD_INITIAL_PRESSURE, 800)
            pipeline = [
                sp.SegmentationBasedVolumeCreationAdapter(settings),
                IpascSimpaKWaveAdapter(settings, initial_pressure=initial_pressure)
            ]
        else:
            pipeline = [
                sp.SegmentationBasedVolumeCreationAdapter(settings),
                sp.MCXAdapter(settings),
                IpascSimpaKWaveAdapter(settings)
            ]
    else:
        acoustic_settings[Tags.DATA_FIELD] = Tags.DATA_FIELD_ABSORPTION_PER_CM
        pipeline = [
            sp.SegmentationBasedVolumeCreationAdapter(settings),
            IpascSimpaKWaveAdapter(settings)
        ]

    # Create the Cyberdyne LED-based system as the photoacoustic model
    device = CyberdyneLEDArraySystem(device_position_mm=np.array([dim_x_mm/2,
                                                                 dim_y_mm/2,
                                                                 0]),
                                     field_of_view_extent_mm=np.asarray([-20, 20, 0, 0, 0, 40]))

    sp.simulate(simulation_pipeline=pipeline,
                settings=settings,
                digital_device_twin=device)

    return path_manager.get_hdf5_file_save_path() + f"/{data_name}_ipasc.hdf5"
