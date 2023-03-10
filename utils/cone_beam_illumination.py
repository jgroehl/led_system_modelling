# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.core.device_digital_twins import IlluminationGeometryBase
from simpa.utils import Tags


class ConeBeamIlluminationGeometry(IlluminationGeometryBase):
    """
    This class represents a pencil beam illumination geometry.
    The device position is defined as the exact position of the pencil beam.
    """

    def __init__(self, device_position_mm=None, source_direction_vector=None, field_of_view_extent_mm=None, opening_angle=None):
        super(ConeBeamIlluminationGeometry, self).__init__(device_position_mm, source_direction_vector, field_of_view_extent_mm)
        if opening_angle is not None:
            self.opening_angle = opening_angle
        else:
            self.opening_angle = 0

    def get_mcx_illuminator_definition(self, global_settings) -> dict:
        source_type = "cone"

        spacing = global_settings[Tags.SPACING_MM]

        device_position = list(self.device_position_mm / spacing + 0.5)

        source_direction = list(self.normalized_source_direction_vector)

        source_param1 = [self.opening_angle, 0, 0, 0]

        source_param2 = [0, 0, 0, 0]

        return {
            "Type": source_type,
            "Pos": list(device_position),
            "Dir": source_direction,
            "Param1": source_param1,
            "Param2": source_param2
        }

    def serialize(self) -> dict:
        serialized_device = self.__dict__
        return {"PencilBeamIlluminationGeometry": serialized_device}

    @staticmethod
    def deserialize(dictionary_to_deserialize):
        deserialized_device = ConeBeamIlluminationGeometry()
        for key, value in dictionary_to_deserialize.items():
            deserialized_device.__dict__[key] = value
        return deserialized_device
