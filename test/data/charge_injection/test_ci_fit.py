import numpy as np
import pytest

from autofit.tools import fit_util
from autocti.charge_injection import ci_frame, ci_fit
from autocti.charge_injection import ci_hyper, ci_data


class MockGeometry(object):

    def __init__(self):
        super(MockGeometry, self).__init__()


class MockPattern(object):

    def __init__(self):
        pass


class MockCIPreCTI(np.ndarray):

    def __new__(cls, array, frame_geometry=MockGeometry(), ci_pattern=MockPattern(), value=1.0, *args, **kwargs):
        ci = np.array(array).view(cls)
        ci.frame_geometry = frame_geometry
        ci.ci_pattern = ci_pattern
        ci.value = value
        return ci

    def create_ci_post_cti(self, cti_params, cti_settings):
        return self.value * np.ones((2, 2))


class MockParams(object):

    def __init__(self):
        pass


class MockSettings(object):

    def __init__(self):
        pass


@pytest.fixture(name='ci_datas_fit')
def make_ci_datas():
    
    ci_datas_fit = []

    ci_image_0 = ci_frame.CIFrame(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=1.0 * np.ones((2, 2)))
    ci_noise_map_0 = ci_frame.CIFrame(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=2.0 * np.ones((2, 2)))

    ci_pre_cti_0 = MockCIPreCTI(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=3.0 * np.ones((2, 2)),
                                value=3.0)
    ci_mask_0 = np.ma.zeros((2, 2))
    ci_noise_scalings_0 = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[1.0, 2.0], [3.0, 4.0]])]

    ci_datas_fit.append(ci_data.CIDataFit(image=ci_image_0, noise_map=ci_noise_map_0, ci_pre_cti=ci_pre_cti_0,
                                          mask=ci_mask_0, noise_scalings=ci_noise_scalings_0))

    ci_image_1 =  ci_frame.CIFrame(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=4.0 * np.ones((2, 2)))
    ci_noise_map_1 =  ci_frame.CIFrame(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=5.0 * np.ones((2, 2)))
    ci_pre_cti_1 = MockCIPreCTI(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=6.0 * np.ones((2, 2)),
                                value=6.0)
    ci_mask_1 = np.ma.array([[True, False], [False, True]])
    ci_noise_scalings_1 = [np.array([[4.0, 3.0], [2.0, 1.0]]), np.array([[4.0, 3.0], [2.0, 1.0]])]

    ci_datas_fit.append(ci_data.CIDataFit(image=ci_image_1, noise_map=ci_noise_map_1, ci_pre_cti=ci_pre_cti_1,
                                          mask=ci_mask_1, noise_scalings=ci_noise_scalings_1))

    return ci_datas_fit


class TestCIFit:

    def test__x2_ci_datas__residuals_are_same_as_calculated_individually(self, ci_datas_fit):
        
        residual_map_0 = fit_util.residual_map_from_data_mask_and_model_data(data=ci_datas_fit[0].image, 
                                                                         mask=ci_datas_fit[0].mask,
                                                                         model_data=ci_datas_fit[0].ci_pre_cti)
        
        residual_map_1 = fit_util.residual_map_from_data_mask_and_model_data(data=ci_datas_fit[1].image, 
                                                                         mask=ci_datas_fit[1].mask,
                                                                         model_data=ci_datas_fit[1].ci_pre_cti)

        fit = ci_fit.CIFit(ci_datas_fit=ci_datas_fit, cti_params=MockParams(), cti_settings=MockSettings())

        assert (residual_map_0 == fit.residual_maps[0]).all()
        assert (residual_map_1 == fit.residual_maps[1]).all()
        assert fit.residual_maps[0].frame_geometry == ci_datas_fit[0].image.frame_geometry
        assert fit.residual_maps[1].frame_geometry == ci_datas_fit[1].image.frame_geometry
        assert fit.residual_maps[0].ci_pattern == ci_datas_fit[0].image.ci_pattern
        assert fit.residual_maps[1].ci_pattern == ci_datas_fit[1].image.ci_pattern

    def test__x2_ci_datas__chi_squareds_are_same_as_calculated_individually(self, ci_datas_fit):

        residual_map_0 = fit_util.residual_map_from_data_mask_and_model_data(data=ci_datas_fit[0].image,
                                                                             mask=ci_datas_fit[0].mask,
                                                                             model_data=ci_datas_fit[0].ci_pre_cti)


        residual_map_1 = fit_util.residual_map_from_data_mask_and_model_data(data=ci_datas_fit[1].image,
                                                                             mask=ci_datas_fit[1].mask,
                                                                             model_data=ci_datas_fit[1].ci_pre_cti)

        chi_sqaured_map_0 = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=residual_map_0, noise_map=ci_datas_fit[0].noise_map, mask=ci_datas_fit[0].mask)

        chi_sqaured_map_1 = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=residual_map_1, noise_map=ci_datas_fit[1].noise_map, mask=ci_datas_fit[0].mask)

        fit = ci_fit.CIFit(ci_datas_fit=ci_datas_fit, cti_params=MockParams(), cti_settings=MockSettings())

        assert (chi_sqaured_map_0 == fit.chi_squared_maps[0]).all()
        assert (chi_sqaured_map_1 == fit.chi_squared_maps[1]).all()
        assert fit.chi_squared_maps[0].frame_geometry == ci_datas_fit[0].image.frame_geometry
        assert fit.chi_squared_maps[1].frame_geometry == ci_datas_fit[1].image.frame_geometry
        assert fit.chi_squared_maps[0].ci_pattern == ci_datas_fit[0].image.ci_pattern
        assert fit.chi_squared_maps[1].ci_pattern == ci_datas_fit[1].image.ci_pattern

    def test__x2_ci_datas__likelihoods_are_same_as_calculated_individually(self, ci_datas_fit):

        residual_map_0 = fit_util.residual_map_from_data_mask_and_model_data(data=ci_datas_fit[0].image,
                                                                             mask=ci_datas_fit[0].mask,
                                                                             model_data=ci_datas_fit[0].ci_pre_cti)

        chi_squared_map_0 = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=residual_map_0, noise_map=ci_datas_fit[0].noise_map, mask=ci_datas_fit[0].mask)

        chi_squared_0 = fit_util.chi_squared_from_chi_squared_map_and_mask(chi_squared_map=chi_squared_map_0,
                                                                                mask=ci_datas_fit[0].mask)

        noise_normalization_0 = fit_util.noise_normalization_from_noise_map_and_mask(noise_map=ci_datas_fit[0].noise_map,
                                                                                     mask=ci_datas_fit[0].mask)

        likelihood_0 = fit_util.likelihood_from_chi_squared_and_noise_normalization(
            chi_squared=chi_squared_0, noise_normalization=noise_normalization_0)

        residual_map_1 = fit_util.residual_map_from_data_mask_and_model_data(data=ci_datas_fit[1].image,
                                                                             mask=ci_datas_fit[1].mask,
                                                                             model_data=ci_datas_fit[1].ci_pre_cti)

        chi_squared_map_1 = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=residual_map_1, noise_map=ci_datas_fit[1].noise_map, mask=ci_datas_fit[1].mask)

        chi_squared_1 = fit_util.chi_squared_from_chi_squared_map_and_mask(chi_squared_map=chi_squared_map_1,
                                                                           mask=ci_datas_fit[1].mask)

        noise_normalization_1 = fit_util.noise_normalization_from_noise_map_and_mask(
            noise_map=ci_datas_fit[1].noise_map,
            mask=ci_datas_fit[1].mask)

        likelihood_1 = fit_util.likelihood_from_chi_squared_and_noise_normalization(
            chi_squared=chi_squared_1, noise_normalization=noise_normalization_1)

        fit = ci_fit.CIFit(ci_datas_fit, cti_params=MockParams(), cti_settings=MockSettings())

        assert likelihood_0 + likelihood_1 == fit.likelihood


class TestCIHyperFit:

    def test__image_and_ci_post_cti_the_same__noise_scaling_all_0s__likelihood_is_noise_normalization(self):
        
        ci_image_0 = ci_frame.CIFrame(array=10.0 * np.ones((2, 2)), frame_geometry=MockGeometry(),
                                      ci_pattern=MockPattern())
        ci_noise_map_0 = ci_frame.CIFrame(array=2.0 * np.ones((2, 2)), frame_geometry=MockGeometry(),
                                          ci_pattern=MockPattern())
        ci_mask_0 = ci_frame.CIFrame(array=np.ma.zeros((2, 2)), frame_geometry=MockGeometry(),
                                     ci_pattern=MockPattern())
        ci_pre_cti_0 = MockCIPreCTI(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=10.0 * np.ones((2, 2)),
                                value=10.0)
        ci_noise_scalings_0 = [ci_frame.CIFrame(array=np.array([[0.0, 0.0], [0.0, 0.0]]), frame_geometry=MockGeometry(),
                                                ci_pattern=MockPattern())]

        ci_datas_fit = []
        ci_datas_fit.append(ci_data.CIDataFit(image=ci_image_0, noise_map=ci_noise_map_0, ci_pre_cti=ci_pre_cti_0,
                                              mask=ci_mask_0, noise_scalings=ci_noise_scalings_0))

        hyper_noise_map_0 = ci_hyper.HyperCINoise(scale_factor=1.0)

        fit = ci_fit.CIHyperFit(ci_datas_fit=ci_datas_fit, cti_params=MockParams(), cti_settings=MockSettings(),
                                hyper_noises=[hyper_noise_map_0])

        chi_squared = 0
        noise_normalization = 4.0 * np.log(2 * np.pi * 4.0)

        assert fit.likelihood == -0.5 * (chi_squared + noise_normalization)

    def test__image_and_post_cti_different__noise_scaling_all_0s__likelihood_chi_squared_and_noise_normalization(self):

        ci_image_0 = ci_frame.CIFrame(array=9.0 * np.ones((2, 2)), frame_geometry=MockGeometry(),
                                      ci_pattern=MockPattern())
        ci_noise_map_0 = ci_frame.CIFrame(array=2.0 * np.ones((2, 2)), frame_geometry=MockGeometry(),
                                          ci_pattern=MockPattern())
        ci_mask_0 = ci_frame.CIFrame(array=np.ma.zeros((2, 2)), frame_geometry=MockGeometry(),
                                     ci_pattern=MockPattern())
        ci_pre_cti_0 = MockCIPreCTI(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=10.0 * np.ones((2, 2)),
                                value=10.0)
        ci_noise_scalings_0 = [ci_frame.CIFrame(array=np.array([[0.0, 0.0], [0.0, 0.0]]), frame_geometry=MockGeometry(),
                                                ci_pattern=MockPattern())]

        ci_datas_fit = []
        ci_datas_fit.append(ci_data.CIDataFit(image=ci_image_0, noise_map=ci_noise_map_0, ci_pre_cti=ci_pre_cti_0,
                                              mask=ci_mask_0, noise_scalings=ci_noise_scalings_0))

        hyper_noise_map_0 = ci_hyper.HyperCINoise(scale_factor=1.0)

        fit = ci_fit.CIHyperFit(ci_datas_fit=ci_datas_fit, cti_params=MockParams(), cti_settings=MockSettings(),
                                hyper_noises=[hyper_noise_map_0])

        chi_squared = 4.0 * ((1.0 / 2.0) ** 2.0)
        noise_normalization = 4.0 * np.log(2 * np.pi * 4.0)

        assert fit.likelihood == pytest.approx(-0.5 * (chi_squared + noise_normalization), 1e-4)

    def test__image_and_ci_post_cti_the_same__noise_scaling_non_0s__likelihood_is_noise_normalization(self):
        
        ci_image_0 = ci_frame.CIFrame(array=10.0 * np.ones((2, 2)), frame_geometry=MockGeometry(),
                                      ci_pattern=MockPattern())
        ci_noise_map_0 = ci_frame.CIFrame(array=2.0 * np.ones((2, 2)), frame_geometry=MockGeometry(),
                                          ci_pattern=MockPattern())
        ci_mask_0 = ci_frame.CIFrame(array=np.ma.zeros((2, 2)), frame_geometry=MockGeometry(),
                                     ci_pattern=MockPattern())
        ci_pre_cti_0 = MockCIPreCTI(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=10.0 * np.ones((2, 2)),
                                value=10.0)

        ci_noise_scalings_0 = [ci_frame.CIFrame(array=np.array([[1.0, 2.0], [3.0, 4.0]]), frame_geometry=MockGeometry(),
                                                ci_pattern=MockPattern()),
                               ci_frame.CIFrame(array=np.array([[5.0, 6.0], [7.0, 8.0]]), frame_geometry=MockGeometry(),
                                                ci_pattern=MockPattern())]

        ci_datas_fit = []
        ci_datas_fit.append(ci_data.CIDataFit(image=ci_image_0, noise_map=ci_noise_map_0, ci_pre_cti=ci_pre_cti_0,
                                              mask=ci_mask_0, noise_scalings=ci_noise_scalings_0))

        hyper_noise_map_0 = ci_hyper.HyperCINoise(scale_factor=1.0)

        fit = ci_fit.CIHyperFit(ci_datas_fit=ci_datas_fit, cti_params=MockParams(), cti_settings=MockSettings(),
                                hyper_noises=[hyper_noise_map_0])
        
        chi_squared = 0
        noise_normalization = np.log(2 * np.pi * (2.0 + 1.0) ** 2.0) + np.log(2 * np.pi * (2.0 + 2.0) ** 2.0) + \
                     np.log(2 * np.pi * (2.0 + 3.0) ** 2.0) + np.log(2 * np.pi * (2.0 + 4.0) ** 2.0)

        assert fit.likelihood == -0.5 * (chi_squared + noise_normalization)

    def test__x2_noise_map_scaling_and_hyper_params__noise_map_term_comes_out_correct(self):

        ci_image_0 = ci_frame.CIFrame(array=10.0 * np.ones((2, 2)), frame_geometry=MockGeometry(),
                                      ci_pattern=MockPattern())
        ci_noise_map_0 = ci_frame.CIFrame(array=3.0 * np.ones((2, 2)), frame_geometry=MockGeometry(),
                                          ci_pattern=MockPattern())
        ci_mask_0 = ci_frame.CIFrame(array=np.ma.zeros((2, 2)), frame_geometry=MockGeometry(),
                                     ci_pattern=MockPattern())
        ci_pre_cti_0 = MockCIPreCTI(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=10.0 * np.ones((2, 2)),
                                value=10.0)

        ci_noise_scalings_0 = [ci_frame.CIFrame(array=np.array([[1.0, 2.0], [3.0, 4.0]]), frame_geometry=MockGeometry(),
                                                ci_pattern=MockPattern()),
                               ci_frame.CIFrame(array=np.array([[5.0, 6.0], [7.0, 8.0]]), frame_geometry=MockGeometry(),
                                                ci_pattern=MockPattern())]

        ci_datas_fit = []
        ci_datas_fit.append(ci_data.CIDataFit(image=ci_image_0, noise_map=ci_noise_map_0, ci_pre_cti=ci_pre_cti_0,
                                              mask=ci_mask_0, noise_scalings=ci_noise_scalings_0))

        hyper_noise_map_0 = ci_hyper.HyperCINoise(scale_factor=1.0)
        hyper_noise_map_1 = ci_hyper.HyperCINoise(scale_factor=2.0)

        fit = ci_fit.CIHyperFit(ci_datas_fit=ci_datas_fit, cti_params=MockParams(), cti_settings=MockSettings(),
                                hyper_noises=[hyper_noise_map_0, hyper_noise_map_1])

        chi_squared = 0
        noise_normalization = np.log(2 * np.pi * (3.0 + 1.0 + 10.0) ** 2.0) + np.log(2 * np.pi * (3.0 + 2.0 + 12.0) ** 2.0) + \
                     np.log(2 * np.pi * (3.0 + 3.0 + 14.0) ** 2.0) + np.log(2 * np.pi * (3.0 + 4.0 + 16.0) ** 2.0)

        assert fit.likelihood == -0.5 * (chi_squared + noise_normalization)

    def test__x2_ci_datas__all_quantities_are_same_as_calculated_individually(self, ci_datas_fit):

        hyper_noises = [ci_hyper.HyperCINoise(scale_factor=1.0), ci_hyper.HyperCINoise(scale_factor=2.0)]

        fit = ci_fit.CIHyperFit(ci_datas_fit, cti_params=MockParams(), cti_settings=MockSettings(),
                                hyper_noises=hyper_noises)

        hyper_noise_map_0 = ci_fit.hyper_noise_map_from_noise_map_and_noise_scalings(
            noise_map=ci_datas_fit[0].noise_map,noise_scalings=ci_datas_fit[0].noise_scalings,
            hyper_noises=hyper_noises)
        
        hyper_noise_map_1 = ci_fit.hyper_noise_map_from_noise_map_and_noise_scalings(
            noise_map=ci_datas_fit[1].noise_map, noise_scalings=ci_datas_fit[1].noise_scalings,
            hyper_noises=hyper_noises)
        
        assert (hyper_noise_map_0 == fit.noise_maps[0]).all()
        assert (hyper_noise_map_1 == fit.noise_maps[1]).all()
        assert fit.noise_maps[0].frame_geometry == ci_datas_fit[0].noise_map.frame_geometry
        assert fit.noise_maps[1].frame_geometry == ci_datas_fit[1].noise_map.frame_geometry
        assert fit.noise_maps[0].ci_pattern == ci_datas_fit[0].noise_map.ci_pattern
        assert fit.noise_maps[1].ci_pattern == ci_datas_fit[1].noise_map.ci_pattern
        
        residual_map_0 = fit_util.residual_map_from_data_mask_and_model_data(data=ci_datas_fit[0].image, 
                                                                             mask=ci_datas_fit[0].mask,
                                                                             model_data=ci_datas_fit[0].ci_pre_cti)
        
        residual_map_1 = fit_util.residual_map_from_data_mask_and_model_data(data=ci_datas_fit[1].image, 
                                                                             mask=ci_datas_fit[1].mask,
                                                                             model_data=ci_datas_fit[1].ci_pre_cti)
        
        assert (residual_map_0 == fit.residual_maps[0]).all()
        assert (residual_map_1 == fit.residual_maps[1]).all()
        assert fit.residual_maps[0].frame_geometry == ci_datas_fit[0].image.frame_geometry
        assert fit.residual_maps[1].frame_geometry == ci_datas_fit[1].image.frame_geometry
        assert fit.residual_maps[0].ci_pattern == ci_datas_fit[0].image.ci_pattern
        assert fit.residual_maps[1].ci_pattern == ci_datas_fit[1].image.ci_pattern
        
        chi_squared_map_0 = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=residual_map_0, noise_map=hyper_noise_map_0, mask=ci_datas_fit[0].mask)
        
        chi_squared_map_1 = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=residual_map_1, noise_map=hyper_noise_map_1, mask=ci_datas_fit[1].mask)


        assert (chi_squared_map_0 == fit.chi_squared_maps[0]).all()
        assert (chi_squared_map_1 == fit.chi_squared_maps[1]).all()
        assert fit.chi_squared_maps[0].frame_geometry == ci_datas_fit[0].image.frame_geometry
        assert fit.chi_squared_maps[1].frame_geometry == ci_datas_fit[1].image.frame_geometry
        assert fit.chi_squared_maps[0].ci_pattern == ci_datas_fit[0].image.ci_pattern
        assert fit.chi_squared_maps[1].ci_pattern == ci_datas_fit[1].image.ci_pattern

        chi_squared_0 = fit_util.chi_squared_from_chi_squared_map_and_mask(chi_squared_map=chi_squared_map_0,
                                                                           mask=ci_datas_fit[0].mask)

        chi_squared_1 = fit_util.chi_squared_from_chi_squared_map_and_mask(chi_squared_map=chi_squared_map_1,
                                                                           mask=ci_datas_fit[1].mask)

        noise_normalization_0 = fit_util.noise_normalization_from_noise_map_and_mask(
            noise_map=hyper_noise_map_0, mask=ci_datas_fit[0].mask)

        noise_normalization_1 = fit_util.noise_normalization_from_noise_map_and_mask(
            noise_map=hyper_noise_map_1, mask=ci_datas_fit[1].mask)
        
        likelihood_0 = fit_util.likelihood_from_chi_squared_and_noise_normalization(chi_squared=chi_squared_0,
                            noise_normalization=noise_normalization_0)

        likelihood_1 = fit_util.likelihood_from_chi_squared_and_noise_normalization(chi_squared=chi_squared_1,
                            noise_normalization=noise_normalization_1)

        assert (likelihood_0 + likelihood_1 == fit.likelihood).all()


class TestScaledNoiseMap:

    def test__image_and_pre_cti_not_identical__noise_scalings_are_0s__no_noise_map_scaling(self):
        noise_map = 2.0 * np.ones((2, 2))
        noise_scalings = [np.array([[0.0, 0.0], [0.0, 0.0]])]
        hyper_noises = [ci_hyper.HyperCINoise(scale_factor=1.0)]

        noise_map = ci_fit.hyper_noise_map_from_noise_map_and_noise_scalings(noise_map=noise_map,
                                                                             noise_scalings=noise_scalings,
                                                                             hyper_noises=hyper_noises)

        assert (noise_map == (np.array([[2.0, 2.0],
                                               [2.0, 2.0]]))).all()

    def test__image_and_pre_cti_not_identical__factor_is_0__no_noise_map_scaling(self):
        noise_map = 2.0 * np.ones((2, 2))
        noise_scalings = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        hyper_noises = [ci_hyper.HyperCINoise(scale_factor=0.0)]

        noise_map = ci_fit.hyper_noise_map_from_noise_map_and_noise_scalings(noise_map=noise_map,
                                                                             noise_scalings=noise_scalings,
                                                                             hyper_noises=hyper_noises)

        assert (noise_map == (np.array([[2.0, 2.0],
                                               [2.0, 2.0]]))).all()

    def test__image_and_pre_cti_not_identical__chi_sq_is_by_noise_map(self):
        noise_map = 2.0 * np.ones((2, 2))
        noise_scalings = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        hyper_noises = [ci_hyper.HyperCINoise(scale_factor=1.0)]

        noise_map = ci_fit.hyper_noise_map_from_noise_map_and_noise_scalings(noise_map=noise_map,
                                                                             noise_scalings=noise_scalings,
                                                                             hyper_noises=hyper_noises)

        assert (noise_map == (np.array([[3.0, 4.0],
                                               [5.0, 6.0]]))).all()

    def test__x2_noise_map_scaling_and_hyper_params__noise_map_term_comes_out_correct(self):
        noise_map = 2.0 * np.ones((2, 2))
        noise_scalings = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[1.0, 2.0], [3.0, 4.0]])]
        hyper_noises = [ci_hyper.HyperCINoise(scale_factor=1.0), ci_hyper.HyperCINoise(scale_factor=2.0)]

        noise_map = ci_fit.hyper_noise_map_from_noise_map_and_noise_scalings(noise_map=noise_map,
                                                                             noise_scalings=noise_scalings,
                                                                             hyper_noises=hyper_noises)

        assert (noise_map == (np.array([[5.0, 8.0],
                                               [11.0, 14.0]]))).all()

    def test__same_as_above__use_ci_datas_fit(self):

        ci_datas_fit = [ci_data.CIDataFit(image=None, noise_map=2.0 * np.ones((2, 2)), ci_pre_cti=None, mask=None,
                                          noise_scalings=[np.array([[1.0, 2.0], [3.0, 4.0]]),
                                                              np.array([[1.0, 2.0], [3.0, 4.0]])])]

        hyper_noises = [ci_hyper.HyperCINoise(scale_factor=1.0), ci_hyper.HyperCINoise(scale_factor=2.0)]

        noise_map = ci_fit.hyper_noise_maps_from_ci_datas_fit_and_hyper_noises(
            ci_datas_fit=ci_datas_fit, hyper_noises=hyper_noises)

        assert (noise_map == (np.array([[5.0, 8.0],
                                               [11.0, 14.0]]))).all()

    def test__same_as_above__but_use_x2__ci_datas_fit(self):

        ci_datas_fit = [ci_data.CIDataFit(image=None, noise_map=2.0 * np.ones((2, 2)), ci_pre_cti=None, mask=None,
                                          noise_scalings=[np.array([[1.0, 2.0], [3.0, 4.0]]),
                                                               np.array([[1.0, 2.0], [3.0, 4.0]])]),
                        ci_data.CIDataFit(image=None, noise_map=2.0 * np.ones((2, 2)), ci_pre_cti=None, mask=None,
                                          noise_scalings=[np.array([[2.0, 2.0], [3.0, 4.0]]),
                                                              np.array([[2.0, 2.0], [3.0, 4.0]])])]

        hyper_noises = [ci_hyper.HyperCINoise(scale_factor=1.0), ci_hyper.HyperCINoise(scale_factor=2.0)]

        noise_map = ci_fit.hyper_noise_maps_from_ci_datas_fit_and_hyper_noises(
            ci_datas_fit=ci_datas_fit, hyper_noises=hyper_noises)

        assert (noise_map[0] == (np.array([[5.0, 8.0],
                                                  [11.0, 14.0]]))).all()
        assert (noise_map[1] == (np.array([[8.0, 8.0],
                                                  [11.0, 14.0]]))).all()
