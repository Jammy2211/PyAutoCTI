import os

import numpy as np
import pytest
import shutil

from autocti import structures as struct
from autocti import dataset as ds

test_data_dir = "{}/files/imaging/".format(os.path.dirname(os.path.realpath(__file__)))


def test__poisson_noise_from_data():

    data = struct.Array.zeros(shape_2d=(2, 2))
    exposure_time_map = struct.Array.ones(shape_2d=(2, 2))

    poisson_noise = ds.preprocess.poisson_noise_from_data(
        data=data, exposure_time_map=exposure_time_map, seed=1
    )

    assert (poisson_noise == np.zeros((2, 2))).all()

    data = struct.Array.manual_2d([[10.0, 0.0], [0.0, 10.0]])
    exposure_time_map = struct.Array.ones(shape_2d=(2, 2))

    poisson_noise = ds.preprocess.poisson_noise_from_data(
        data=data, exposure_time_map=exposure_time_map, seed=1
    )

    assert (poisson_noise == np.array([[(10.0 - 9.0), 0], [0, (10.0 - 6.0)]])).all()

    data = struct.Array.full(fill_value=10.0, shape_2d=(2, 2))
    exposure_time_map = struct.Array.ones(shape_2d=(2, 2))

    poisson_noise = ds.preprocess.poisson_noise_from_data(
        data=data, exposure_time_map=exposure_time_map, seed=1
    )

    # Use known noise_map_1d map for given seed.
    assert (poisson_noise == np.array([[1, 4], [3, 1]])).all()

    data = struct.Array.manual_2d([[10000000.0, 0.0], [0.0, 10000000.0]])
    exposure_time_map = struct.Array.ones(shape_2d=(2, 2))

    poisson_noise = ds.preprocess.poisson_noise_from_data(
        data=data, exposure_time_map=exposure_time_map, seed=1
    )

    assert (poisson_noise == np.array([[743, 0], [0, 3783]])).all()


def test__data_with_poisson_noised_added():

    data = struct.Array.zeros(shape_2d=(2, 2))
    exposure_time_map = struct.Array.ones(shape_2d=(2, 2))
    data_with_poisson_noise = ds.preprocess.data_with_poisson_noise_added(
        data=data, exposure_time_map=exposure_time_map, seed=1
    )

    assert (data_with_poisson_noise == np.zeros((2, 2))).all()

    data = struct.Array.manual_2d([[10.0, 0.0], [0.0, 10.0]])

    exposure_time_map = struct.Array.ones(shape_2d=(2, 2))
    data_with_poisson_noise = ds.preprocess.data_with_poisson_noise_added(
        data=data, exposure_time_map=exposure_time_map, seed=1
    )

    assert (data_with_poisson_noise == np.array([[11, 0], [0, 14]])).all()

    data = struct.Array.full(fill_value=10.0, shape_2d=(2, 2))

    exposure_time_map = struct.Array.ones(shape_2d=(2, 2))
    data_with_poisson_noise = ds.preprocess.data_with_poisson_noise_added(
        data=data, exposure_time_map=exposure_time_map, seed=1
    )

    assert (data_with_poisson_noise == np.array([[11, 14], [13, 11]])).all()

    data = struct.Array.manual_2d([[10000000.0, 0.0], [0.0, 10000000.0]])

    exposure_time_map = struct.Array.ones(shape_2d=(2, 2))

    data_with_poisson_noise = ds.preprocess.data_with_poisson_noise_added(
        data=data, exposure_time_map=exposure_time_map, seed=1
    )

    assert (
        data_with_poisson_noise == np.array([[10000743, 0.0], [0.0, 10003783.0]])
    ).all()


def test__gaussian_noise_from_shape_and_sigma():

    gaussian_noise = ds.preprocess.gaussian_noise_from_shape_and_sigma(
        shape=(9,), sigma=0.0, seed=1
    )

    assert (gaussian_noise == np.zeros((9,))).all()

    gaussian_noise = ds.preprocess.gaussian_noise_from_shape_and_sigma(
        shape=(9,), sigma=1.0, seed=1
    )

    assert gaussian_noise == pytest.approx(
        np.array([1.62, -0.61, -0.53, -1.07, 0.87, -2.30, 1.74, -0.76, 0.32]), 1e-2
    )


def test__data_with_gaussian_noise_added():

    data = struct.Array.ones(shape_2d=(3, 3))

    data_with_noise = ds.preprocess.data_with_gaussian_noise_added(
        data=data, sigma=0.0, seed=1
    )

    assert (data_with_noise == np.ones((3, 3))).all()

    data_with_noise = ds.preprocess.data_with_gaussian_noise_added(
        data=data, sigma=1.0, seed=1
    )

    assert data_with_noise == pytest.approx(
        np.array(
            [
                [1 + 1.62, 1 - 0.61, 1 - 0.53],
                [1 - 1.07, 1 + 0.87, 1 - 2.30],
                [1 + 1.74, 1 - 0.76, 1 + 0.32],
            ]
        ),
        1e-1,
    )
