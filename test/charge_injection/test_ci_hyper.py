import numpy as np

from autocti.charge_injection import ci_hyper


class MockChInj(np.ndarray):

    def __new__(cls, array, geometry=None, ci_pattern=None, *args, **kwargs):
        ci = np.array(array).view(cls)
        ci.frame_geometry = geometry
        ci.ci_pattern = ci_pattern
        return ci


class MockPattern(object):

    def __init__(self):
        pass


class TestScaledNoise:

    def test__noise_scaling_map_all_0s__scaled_noise_is_0s(self):
        noise_scaling_map = np.array([[0.0, 0.0],
                                      [0.0, 0.0]])

        hyper = ci_hyper.CIHyperNoise(scale_factor=10.0)
        scaled_noise_map = hyper.scaled_noise_map_from_noise_scaling(noise_scaling_map)

        assert (scaled_noise_map == np.array([[0.0, 0.0],
                                              [0.0, 0.0]])).all()

    def test__noise_scaling_map_has_values__scaled_noise_is_baseline_noise_plus_values_times_noise_factor(self):
        noise_scaling_map = np.array([[1.0, 2.0],
                                      [5.0, 3.0]])

        hyper = ci_hyper.CIHyperNoise(scale_factor=10.0)
        scaled_noise_map = hyper.scaled_noise_map_from_noise_scaling(noise_scaling_map)

        assert (scaled_noise_map == np.array([[10.0, 20.0],
                                              [50.0, 30.0]])).all()
