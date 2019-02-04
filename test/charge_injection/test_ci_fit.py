import numpy as np
import pytest

from autocti.charge_injection import ci_frame, ci_fit
from autocti.charge_injection import ci_hyper, ci_data
from autocti.data import mask
from test.mock.mock import MockGeometry, MockPattern, MockCIPreCTI, MockParams, MockSettings


class TestCIHyperFit:

    def test__image_and_ci_post_cti_the_same__noise_scaling_all_0s__likelihood_is_noise_normalization(self):
        ci_image_0 = ci_frame.CIFrame(array=10.0 * np.ones((2, 2)), frame_geometry=MockGeometry(),
                                      ci_pattern=MockPattern())
        ci_noise_map_0 = ci_frame.CIFrame(array=2.0 * np.ones((2, 2)), frame_geometry=MockGeometry(),
                                          ci_pattern=MockPattern())
        ci_mask_0 = mask.Mask(array=np.ma.zeros((2, 2)), frame_geometry=MockGeometry(),
                              ci_pattern=MockPattern())
        ci_pre_cti_0 = MockCIPreCTI(frame_geometry=MockGeometry(), ci_pattern=MockPattern(),
                                    array=10.0 * np.ones((2, 2)),
                                    value=10.0)
        ci_noise_scaling_0 = ci_frame.CIFrame(array=np.array([[0.0, 0.0], [0.0, 0.0]]), frame_geometry=MockGeometry(),
                                              ci_pattern=MockPattern())

        ci_data_fit = ci_data.CIDataFit(image=ci_image_0, noise_map=ci_noise_map_0, ci_pre_cti=ci_pre_cti_0,
                                        mask=ci_mask_0, noise_scaling=ci_noise_scaling_0)

        hyper_noise_map_0 = ci_hyper.CIHyperNoise(scale_factor=1.0)

        fit = ci_fit.CIHyperFit(ci_data_fit=ci_data_fit, cti_params=MockParams(), cti_settings=MockSettings(),
                                hyper_noises=[hyper_noise_map_0])

        chi_squared = 0
        noise_normalization = 4.0 * np.log(2 * np.pi * 4.0)

        assert fit.likelihood == -0.5 * (chi_squared + noise_normalization)

    def test__image_and_post_cti_different__noise_scaling_all_0s__likelihood_chi_squared_and_noise_normalization(self):
        ci_image_0 = ci_frame.CIFrame(array=9.0 * np.ones((2, 2)), frame_geometry=MockGeometry(),
                                      ci_pattern=MockPattern())
        ci_noise_map_0 = ci_frame.CIFrame(array=2.0 * np.ones((2, 2)), frame_geometry=MockGeometry(),
                                          ci_pattern=MockPattern())
        ci_mask_0 = mask.Mask(array=np.ma.zeros((2, 2)), frame_geometry=MockGeometry(),
                              ci_pattern=MockPattern())
        ci_pre_cti_0 = MockCIPreCTI(frame_geometry=MockGeometry(), ci_pattern=MockPattern(),
                                    array=10.0 * np.ones((2, 2)),
                                    value=10.0)
        ci_noise_scaling_0 = ci_frame.CIFrame(array=np.array([[0.0, 0.0], [0.0, 0.0]]), frame_geometry=MockGeometry(),
                                              ci_pattern=MockPattern())

        ci_data_fit = ci_data.CIDataFit(image=ci_image_0, noise_map=ci_noise_map_0, ci_pre_cti=ci_pre_cti_0,
                                        mask=ci_mask_0, noise_scaling=ci_noise_scaling_0)

        hyper_noise_map_0 = ci_hyper.CIHyperNoise(scale_factor=1.0)

        fit = ci_fit.CIHyperFit(ci_data_fit=ci_data_fit, cti_params=MockParams(), cti_settings=MockSettings(),
                                hyper_noises=[hyper_noise_map_0])

        chi_squared = 4.0 * ((1.0 / 2.0) ** 2.0)
        noise_normalization = 4.0 * np.log(2 * np.pi * 4.0)

        assert fit.likelihood == pytest.approx(-0.5 * (chi_squared + noise_normalization), 1e-4)


class TestScaledNoiseMap:

    def test__image_and_pre_cti_not_identical__noise_scaling_are_0s__no_noise_map_scaling(self):
        noise_map = 2.0 * np.ones((2, 2))
        noise_scaling = np.array([[0.0, 0.0], [0.0, 0.0]])
        hyper_noise = ci_hyper.CIHyperNoise(scale_factor=1.0)

        noise_map = ci_fit.hyper_noise_map_from_noise_map_and_noise_scaling(noise_map=noise_map,
                                                                            noise_scaling=noise_scaling,
                                                                            hyper_noises=[hyper_noise])

        assert (noise_map == (np.array([[2.0, 2.0],
                                        [2.0, 2.0]]))).all()

    def test__image_and_pre_cti_not_identical__factor_is_0__no_noise_map_scaling(self):
        noise_map = 2.0 * np.ones((2, 2))
        noise_scaling = np.array([[1.0, 2.0], [3.0, 4.0]])
        hyper_noise = ci_hyper.CIHyperNoise(scale_factor=0.0)

        noise_map = ci_fit.hyper_noise_map_from_noise_map_and_noise_scaling(noise_map=noise_map,
                                                                            noise_scaling=noise_scaling,
                                                                            hyper_noises=[hyper_noise])

        assert (noise_map == (np.array([[2.0, 2.0],
                                        [2.0, 2.0]]))).all()

    def test__image_and_pre_cti_not_identical__chi_sq_is_by_noise_map(self):
        noise_map = 2.0 * np.ones((2, 2))
        noise_scaling = np.array([[1.0, 2.0], [3.0, 4.0]])
        hyper_noise = ci_hyper.CIHyperNoise(scale_factor=1.0)

        noise_map = ci_fit.hyper_noise_map_from_noise_map_and_noise_scaling(noise_map=noise_map,
                                                                            noise_scaling=noise_scaling,
                                                                            hyper_noises=[hyper_noise])

        assert (noise_map == (np.array([[3.0, 4.0],
                                        [5.0, 6.0]]))).all()
