import logging
import os

import numpy as np

from autocti import structures as struct
from autocti.dataset import abstract_dataset

logger = logging.getLogger(__name__)

test_data_path = "{}/files/array/".format(os.path.dirname(os.path.realpath(__file__)))


class TestSignalToNoise:
    def test__image_and_noise_are_values__signal_to_noise_is_ratio_of_each(self):
        array = struct.Array.manual_2d([[1.0, 2.0], [3.0, 4.0]])
        noise_map = struct.Array.manual_2d([[10.0, 10.0], [30.0, 4.0]])

        dataset = abstract_dataset.AbstractDataset(data=array, noise_map=noise_map)

        assert (dataset.signal_to_noise_map == np.array([[0.1, 0.2], [0.1, 1.0]])).all()
        assert dataset.signal_to_noise_max == 1.0

    def test__same_as_above__but_image_has_negative_values__replaced_with_zeros(self):
        array = struct.Array.manual_2d([[-1.0, 2.0], [3.0, -4.0]])

        noise_map = struct.Array.manual_2d([[10.0, 10.0], [30.0, 4.0]])

        dataset = abstract_dataset.AbstractDataset(data=array, noise_map=noise_map)

        assert (dataset.signal_to_noise_map == np.array([[0.0, 0.2], [0.1, 0.0]])).all()
        assert dataset.signal_to_noise_max == 0.2


class TestAbsoluteSignalToNoise:
    def test__image_and_noise_are_values__signal_to_noise_is_absolute_image_value_over_noise(
        self
    ):
        array = struct.Array.manual_2d([[-1.0, 2.0], [3.0, -4.0]])

        noise_map = struct.Array.manual_2d([[10.0, 10.0], [30.0, 4.0]])

        dataset = abstract_dataset.AbstractDataset(data=array, noise_map=noise_map)

        assert (
            dataset.absolute_signal_to_noise_map == np.array([[0.1, 0.2], [0.1, 1.0]])
        ).all()
        assert dataset.absolute_signal_to_noise_max == 1.0


class TestPotentialChiSquaredMap:
    def test__image_and_noise_are_values__signal_to_noise_is_absolute_image_value_over_noise(
        self
    ):
        array = struct.Array.manual_2d([[-1.0, 2.0], [3.0, -4.0]])
        noise_map = struct.Array.manual_2d([[10.0, 10.0], [30.0, 4.0]])

        dataset = abstract_dataset.AbstractDataset(data=array, noise_map=noise_map)

        assert (
            dataset.potential_chi_squared_map
            == np.array([[0.1 ** 2.0, 0.2 ** 2.0], [0.1 ** 2.0, 1.0 ** 2.0]])
        ).all()
        assert dataset.potential_chi_squared_max == 1.0
