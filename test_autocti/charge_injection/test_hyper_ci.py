import numpy as np

import autocti as ac

def test__noise_scaling_map_all_0s__scaled_noise_is_0s():
    noise_scaling_map = np.array([[0.0, 0.0], [0.0, 0.0]])

    hyper = ac.HyperCINoiseScalar(scale_factor=10.0)
    scaled_noise_map = hyper.scaled_noise_map_from(noise_scaling_map)

    assert (scaled_noise_map == np.array([[0.0, 0.0], [0.0, 0.0]])).all()

def test__noise_scaling_map_has_values__scaled_noise_is_baseline_noise_plus_values_times_noise_factor():
    noise_scaling_map = np.array([[1.0, 2.0], [5.0, 3.0]])

    hyper = ac.HyperCINoiseScalar(scale_factor=10.0)
    scaled_noise_map = hyper.scaled_noise_map_from(noise_scaling_map)

    assert (scaled_noise_map == np.array([[10.0, 20.0], [50.0, 30.0]])).all()

def test__collection__as_dict():

    regions_ci = ac.HyperCINoiseScalar(scale_factor=10.0)
    serial_eper = ac.HyperCINoiseScalar(scale_factor=20.0)

    hyper_collection = ac.HyperCINoiseCollection(
        regions_ci=regions_ci,
        serial_eper=serial_eper
    )

    assert hyper_collection.as_dict == {"regions_ci" : regions_ci, "serial_eper" : serial_eper}