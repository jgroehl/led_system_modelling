# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.core.device_digital_twins import PhotoacousticDevice, \
    LinearArrayDetectionGeometry, PencilBeamIlluminationGeometry
from utils.cone_beam_illumination import ConeBeamIlluminationGeometry
import numpy as np


class CyberdyneLEDArraySystem(PhotoacousticDevice):
    """
    This class represents a digital twin of the Cyberdyne Photoacoustic system with an LED bar attached to
    either side of the transducer.

    """

    def __init__(self, device_position_mm: np.ndarray = None,
                 field_of_view_extent_mm: np.ndarray = None):
        """
        :param device_position_mm: Each device has an internal position which serves as origin for internal \
        representations of e.g. detector element positions or illuminator positions.
        :type device_position_mm: ndarray
        :param field_of_view_extent_mm: Field of view which is defined as a numpy array of the shape \
        [xs, xe, ys, ye, zs, ze], where x, y, and z denote the coordinate axes and s and e denote the start and end \
        positions.
        :type field_of_view_extent_mm: ndarray
        """
        super(CyberdyneLEDArraySystem, self).__init__(device_position_mm=device_position_mm)

        self.mediprene_membrane_height_mm = 1
        self.probe_height_mm = 43.2
        self.focus_in_field_of_view_mm = 8
        self.detection_geometry_position_vector = np.add(self.device_position_mm,
                                                         np.array([0, 0, self.focus_in_field_of_view_mm]))

        if field_of_view_extent_mm is None:
            self.field_of_view_extent_mm = np.asarray([-20, 20, 0, 0, 0, 40])
        else:
            self.field_of_view_extent_mm = field_of_view_extent_mm

        self.field_of_view_extent_mm[4] -= self.focus_in_field_of_view_mm
        self.field_of_view_extent_mm[5] -= self.focus_in_field_of_view_mm

        detection_geometry = LinearArrayDetectionGeometry(device_position_mm=self.detection_geometry_position_vector,
                                                          pitch_mm=0.315,
                                                          number_detector_elements=128,
                                                          sampling_frequency_mhz=40,
                                                          field_of_view_extent_mm=np.asarray([-20, 20, 0, 0, 0, 40]))

        self.set_detection_geometry(detection_geometry)

        illum = self.create_illumination_pattern()
        for ill_idx in range(illum["N"]):
            pos = (np.asarray([illum["x_coordinates"][ill_idx],
                              illum["y_coordinates"][ill_idx],
                              illum["z_coordinates"][ill_idx]]) * 1000 +
                   device_position_mm)
            self.add_illumination_geometry(ConeBeamIlluminationGeometry(
                                           device_position_mm=pos,
                                           source_direction_vector=illum["direction"][ill_idx],
                                           opening_angle=np.deg2rad(illum["numerical_apperature"] / 2)
                                           ))

    def create_illumination_pattern(self) -> dict:
        """
        Adapted script from Maura Dantuma
        """
        grid_size = 5.5e-2
        TransFocus = 0
        Nx = 744
        dx = grid_size / Nx
        full_angle = 120
        n_src = 8 * 36
        height_probe = 0
        height_phantom = 3.785e-2

        x_vec = np.linspace(-Nx / 2, Nx / 2, Nx) * dx
        y_vec = x_vec
        Ny = Nx
        Nz = int(height_phantom / dx)
        z_vec = np.linspace(0, Nz, Nz) * dx

        [_, _, Z] = np.meshgrid(x_vec, y_vec, z_vec)
        mask = (Z <= height_phantom)

        setting = dict()
        setting["dx"] = dx
        setting["Nx"] = round(Nx)
        setting["Ny"] = round(Ny)
        setting["Nz"] = round(Nz)
        setting["Nxyz"] = [round(Nx), round(Ny), round(Nz)]
        setting["N"] = np.prod([round(Nx), round(Ny), round(Nz)])
        setting["x_vec"] = x_vec
        setting["y_vec"] = y_vec
        setting["z_vec"] = z_vec
        setting["mask"] = mask

        bar_length = 50e-3
        dx_LEDs = 1.3511e-3
        dz_LEDs = (6 / 3) * 1e-3

        x_row1 = np.linspace(-bar_length / 2 + dx_LEDs, bar_length / 2, 36)
        x_row2 = np.linspace(-bar_length / 2, bar_length / 2 - dx_LEDs, 36)
        x_bar = np.hstack([x_row1, x_row2, x_row1, x_row2])

        Angle = 50
        y_base = (TransFocus - 3e-3 - np.sin(np.deg2rad(Angle)) * 5e-3)
        y_row1 = (y_base - (dz_LEDs * 1.5) * np.sin(np.deg2rad(Angle))) * np.ones((36,))
        y_row2 = (y_base - (dz_LEDs * 0.5) * np.sin(np.deg2rad(Angle))) * np.ones((36,))
        y_row3 = (y_base + (dz_LEDs * 0.5) * np.sin(np.deg2rad(Angle))) * np.ones((36,))
        y_row4 = (y_base + (dz_LEDs * 1.5) * np.sin(np.deg2rad(Angle))) * np.ones((36,))
        y_bar3 = np.hstack([y_row1, y_row2, y_row3, y_row4])

        z_base = height_probe + 6e-3 + np.sin(np.deg2rad(Angle)) * 5e-3 / np.tan(np.deg2rad(Angle))
        z_row1 = (z_base + (dz_LEDs * 1.5) * np.cos(np.deg2rad(Angle))) * np.ones((36,))
        z_row2 = (z_base + (dz_LEDs * 0.5) * np.cos(np.deg2rad(Angle))) * np.ones((36,))
        z_row3 = (z_base - (dz_LEDs * 0.5) * np.cos(np.deg2rad(Angle))) * np.ones((36,))
        z_row4 = (z_base - (dz_LEDs * 1.5) * np.cos(np.deg2rad(Angle))) * np.ones((36,))
        z_bar3 = np.hstack([z_row1, z_row2, z_row3, z_row4])

        vector = np.asarray([0, y_base, z_base])
        v_bar1 = (vector / np.linalg.norm(vector) * np.ones((3, 144)).T).copy()

        y_base = (TransFocus - 3e-3 - np.sin(np.deg2rad(Angle)) * 5e-3)
        y_row1 = (y_base + (dz_LEDs * 1.5) * np.sin(np.deg2rad(-Angle))) * np.ones((36,))
        y_row2 = (y_base + (dz_LEDs * 0.5) * np.sin(np.deg2rad(-Angle))) * np.ones((36,))
        y_row3 = (y_base - (dz_LEDs * 0.5) * np.sin(np.deg2rad(-Angle))) * np.ones((36,))
        y_row4 = (y_base - (dz_LEDs * 1.5) * np.sin(np.deg2rad(-Angle))) * np.ones((36,))
        y_bar4 = np.hstack([y_row1, y_row2, y_row3, y_row4])

        z_base = height_probe - 5e-3 - np.sin(np.deg2rad(Angle)) * 5e-3 / np.tan(np.deg2rad(Angle))
        z_row4 = (z_base + (dz_LEDs * 1.5) * np.cos(np.deg2rad(Angle))) * np.ones((36,))
        z_row3 = (z_base + (dz_LEDs * 0.5) * np.cos(np.deg2rad(Angle))) * np.ones((36,))
        z_row2 = (z_base - (dz_LEDs * 0.5) * np.cos(np.deg2rad(Angle))) * np.ones((36,))
        z_row1 = (z_base - (dz_LEDs * 1.5) * np.cos(np.deg2rad(Angle))) * np.ones((36,))
        z_bar4 = np.hstack([z_row1, z_row2, z_row3, z_row4])

        vector = np.asarray([0, -y_base, -z_base])
        v_bar2 = (vector / np.linalg.norm(vector) * np.ones((3, 144)).T).copy()

        src = dict()
        src["x_coordinates"] = np.asarray(np.hstack([x_bar, x_bar]))
        src["y_coordinates"] = np.asarray(np.hstack([z_bar3, z_bar4]))
        src["z_coordinates"] = np.abs(np.asarray(np.hstack([y_bar3, y_bar4])))
        src["direction"] = np.asarray(np.vstack([v_bar1, v_bar2]))
        src["numerical_apperature"] = full_angle
        src["N"] = n_src
        src["type"] = 'cone'

        return src

    def serialize(self) -> dict:
        serialized_device = self.__dict__
        device_dict = {"CyberdyneLEDArraySystem": serialized_device}
        return device_dict

    @staticmethod
    def deserialize(dictionary_to_deserialize):
        deserialized_device = CyberdyneLEDArraySystem()
        for key, value in dictionary_to_deserialize.items():
            deserialized_device.__dict__[key] = value
        return deserialized_device
