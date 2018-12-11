from autocti.data.fitting import fitting
from autocti.data.charge_injection import ci_frame, ci_hyper
from autocti.data.charge_injection import ci_data

import numpy as np
import pytest


class MockGeometry(ci_frame.CIQuadGeometry):

    def __init__(self):
        super(MockGeometry, self).__init__()


class MockPattern(object):

    def __init__(self):
        pass


class MockChInj(np.ndarray):

    def __new__(cls, array, frame_geometry=MockGeometry(), ci_pattern=MockPattern(),
                baseline_noise=None, *args, **kwargs):
        ci = np.array(array).view(cls)
        ci.frame_geometry = frame_geometry
        ci.ci_pattern = ci_pattern
        ci.baseline_noise = baseline_noise
        return ci

    def create_ci_post_cti(self, cti_params, cti_settings):
        return 3.0 * np.ones((2, 2))


class MockCIPreCTIs(np.ndarray):

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


@pytest.fixture(name='ci_datas')
def make_ci_datas():
    ci_images = [MockChInj(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=1.0 * np.ones((2, 2))),
                 MockChInj(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=4.0 * np.ones((2, 2)))]

    ci_masks = [np.ma.zeros((2, 2)), np.ma.array([[True, False], [False, True]])]

    ci_noises = [MockChInj(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=2.0 * np.ones((2, 2))),
                 MockChInj(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=5.0 * np.ones((2, 2)))]

    ci_noise_scalings = [[np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[1.0, 2.0], [3.0, 4.0]])],
                         [np.array([[4.0, 3.0], [2.0, 1.0]]), np.array([[4.0, 3.0], [2.0, 1.0]])]]

    ci_pre_ctis = [MockCIPreCTIs(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=3.0 * np.ones((2, 2)),
                                 value=3.0),
                   MockCIPreCTIs(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=6.0 * np.ones((2, 2)),
                                 value=6.0)]

    return ci_data.CIDataAnalysis(images=ci_images, masks=ci_masks, noises=ci_noises,
                                  ci_pre_ctis=ci_pre_ctis, noise_scalings=ci_noise_scalings)


# class TestFitter:
#
#     def test__x2_ci_datas__residuals_are_same_as_calculated_individually(self, ci_datas):
#         residuals0 = fitting.residuals_from_image_mask_and_model(ci_datas[0].image, ci_datas[0].mask,
#                                                                  ci_datas[0].ci_pre_cti)
#         residuals1 = fitting.residuals_from_image_mask_and_model(ci_datas[1].image, ci_datas[1].mask,
#                                                                  ci_datas[1].ci_pre_cti)
#
#         fitter = fitting.CIFitter(ci_datas, cti_params=MockParams(), cti_settings=MockSettings())
#
#         assert (residuals0 == fitter.residuals[0]).all()
#         assert (residuals1 == fitter.residuals[1]).all()
#         assert fitter.residuals[0].frame_geometry == ci_datas[0].image.frame_geometry
#         assert fitter.residuals[1].frame_geometry == ci_datas[1].image.frame_geometry
#         assert fitter.residuals[0].ci_pattern == ci_datas[0].image.ci_pattern
#         assert fitter.residuals[1].ci_pattern == ci_datas[1].image.ci_pattern
#
#     def test__x2_ci_datas__chi_squareds_are_same_as_calculated_individually(self, ci_datas):
#         residuals0 = fitting.residuals_from_image_mask_and_model(ci_datas[0].image, ci_datas[0].mask,
#                                                                  ci_datas[0].ci_pre_cti)
#         residuals1 = fitting.residuals_from_image_mask_and_model(ci_datas[1].image, ci_datas[1].mask,
#                                                                  ci_datas[1].ci_pre_cti)
#
#         chi_squareds0 = fitting.chi_squareds_from_residuals_and_noise(residuals0, ci_datas[0].noise)
#         chi_squareds1 = fitting.chi_squareds_from_residuals_and_noise(residuals1, ci_datas[1].noise)
#
#         fitter = fitting.CIFitter(ci_datas, cti_params=MockParams(), cti_settings=MockSettings())
#
#         assert (chi_squareds0 == fitter.chi_squareds[0]).all()
#         assert (chi_squareds1 == fitter.chi_squareds[1]).all()
#         assert fitter.chi_squareds[0].frame_geometry == ci_datas[0].image.frame_geometry
#         assert fitter.chi_squareds[1].frame_geometry == ci_datas[1].image.frame_geometry
#         assert fitter.chi_squareds[0].ci_pattern == ci_datas[0].image.ci_pattern
#         assert fitter.chi_squareds[1].ci_pattern == ci_datas[1].image.ci_pattern
#
#     def test__x2_ci_datas__likelihoods_are_same_as_calculated_individually(self, ci_datas):
#         residuals = fitting.residuals_from_image_mask_and_model(ci_datas[0].image, ci_datas[0].mask,
#                                                                 ci_datas[0].ci_pre_cti)
#         chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, ci_datas[0].noise)
#         chi_squared_term = fitting.chi_squared_term_from_chi_squareds(chi_squareds)
#         noise_term = fitting.noise_term_from_mask_and_noise(ci_datas[0].mask, ci_datas[0].noise)
#         likelihood0 = fitting.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)
#
#         residuals = fitting.residuals_from_image_mask_and_model(ci_datas[1].image, ci_datas[1].mask,
#                                                                 ci_datas[1].ci_pre_cti)
#         chi_squareds = fitting.chi_squareds_from_residuals_and_noise(residuals, ci_datas[1].noise)
#         chi_squared_term = fitting.chi_squared_term_from_chi_squareds(chi_squareds)
#         noise_term = fitting.noise_term_from_mask_and_noise(ci_datas[1].mask, ci_datas[1].noise)
#         likelihood1 = fitting.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)
#
#         fitter = fitting.CIFitter(ci_datas, cti_params=MockParams(), cti_settings=MockSettings())
#
#         assert likelihood0 + likelihood1 == fitter.likelihood
#
#
# class TestHyperFitter:
#     class TestScaledLikelihood:
#
#         def test__noise_scaling_all_0s__identical_to_test_in_fitting(self):
#             images = [MockChInj(array=10.0 * np.ones((2, 2)), frame_geometry=MockGeometry(), ci_pattern=MockPattern())]
#             masks = [MockChInj(array=np.ma.zeros((2, 2)), frame_geometry=MockGeometry(), ci_pattern=MockPattern())]
#             noises = [MockChInj(array=2.0 * np.ones((2, 2)), frame_geometry=MockGeometry(), ci_pattern=MockPattern())]
#             models = [MockChInj(array=10.0 * np.ones((2, 2)), frame_geometry=MockGeometry(), ci_pattern=MockPattern())]
#             noise_scalings = [MockChInj(array=np.array([[0.0, 0.0], [0.0, 0.0]]), frame_geometry=MockGeometry(),
#                                         ci_pattern=MockPattern())]
#             hyper_noise = [ci_hyper.HyperCINoise(scale_factor=1.0)]
#
#             ci_pre_ctis = [MockCIPreCTIs(array=np.zeros((2, 2)), frame_geometry=MockGeometry(), ci_pattern=MockPattern(),
#                                          value=10.0)]
#
#             ci_datas = ci_data.CIDataAnalysis(images=images, masks=masks, noises=noises, ci_pre_ctis=ci_pre_ctis,
#                                               noise_scalings=noise_scalings)
#
#             fitter = fitting.HyperCIFitter(ci_datas, cti_params=MockParams(), cti_settings=MockSettings(),
#                                            hyper_noises=hyper_noise)
#
#             chi_sq_term = 0
#             noise_term = 4.0 * np.log(2 * np.pi * 4.0)
#
#             assert fitter.scaled_likelihood == -0.5 * (chi_sq_term + noise_term)
#
#         def test__image_and_pre_cti_not_identical__likelihood_is_chi_sq_plus_noise_term(self):
#             images = [MockChInj(array=9.0 * np.ones((2, 2)), frame_geometry=MockGeometry(), ci_pattern=MockPattern())]
#             masks = [MockChInj(array=np.ma.zeros((2, 2)), frame_geometry=MockGeometry(), ci_pattern=MockPattern())]
#             noises = [MockChInj(array=2.0 * np.ones((2, 2)), frame_geometry=MockGeometry(), ci_pattern=MockPattern())]
#             models = [MockChInj(array=10.0 * np.ones((2, 2)), frame_geometry=MockGeometry(), ci_pattern=MockPattern())]
#             noise_scalings = [MockChInj(array=np.array([[0.0, 0.0], [0.0, 0.0]]), frame_geometry=MockGeometry(),
#                                         ci_pattern=MockPattern())]
#             hyper_noise = [ci_hyper.HyperCINoise(scale_factor=1.0)]
#
#             ci_pre_ctis = [MockCIPreCTIs(array=np.zeros((2, 2)), frame_geometry=MockGeometry(), ci_pattern=MockPattern(),
#                                          value=10.0)]
#
#             ci_datas = ci_data.CIDataAnalysis(images=images, masks=masks, noises=noises, ci_pre_ctis=ci_pre_ctis,
#                                               noise_scalings=noise_scalings)
#
#             fitter = fitting.HyperCIFitter(ci_datas, cti_params=MockParams(), cti_settings=MockSettings(),
#                                            hyper_noises=hyper_noise)
#
#             chi_sq_term = 4.0 * ((1.0 / 2.0) ** 2.0)
#             noise_term = 4.0 * np.log(2 * np.pi * 4.0)
#
#             assert fitter.scaled_likelihood == pytest.approx(-0.5 * (chi_sq_term + noise_term), 1e-4)
#
#         def test__noise_scaling_different_values__noise_term_comes_out_correct(self):
#             images = [MockChInj(array=10.0 * np.ones((2, 2)), frame_geometry=MockGeometry(), ci_pattern=MockPattern())]
#             masks = [MockChInj(array=np.ma.zeros((2, 2)), frame_geometry=MockGeometry(), ci_pattern=MockPattern())]
#             noises = [MockChInj(array=2.0 * np.ones((2, 2)), frame_geometry=MockGeometry(), ci_pattern=MockPattern())]
#             models = [MockChInj(array=10.0 * np.ones((2, 2)), frame_geometry=MockGeometry(), ci_pattern=MockPattern())]
#             noise_scalings = [[MockChInj(array=np.array([[1.0, 2.0], [3.0, 4.0]]), frame_geometry=MockGeometry(),
#                                          ci_pattern=MockPattern())]]
#             hyper_noise = [ci_hyper.HyperCINoise(scale_factor=1.0)]
#
#             ci_pre_ctis = [MockCIPreCTIs(array=np.zeros((2, 2)), frame_geometry=MockGeometry(), ci_pattern=MockPattern(),
#                                          value=10.0)]
#
#             ci_datas = ci_data.CIDataAnalysis(images=images, masks=masks, noises=noises, ci_pre_ctis=ci_pre_ctis,
#                                               noise_scalings=noise_scalings)
#
#             fitter = fitting.HyperCIFitter(ci_datas, cti_params=MockParams(), cti_settings=MockSettings(),
#                                            hyper_noises=hyper_noise)
#
#             chi_sq_term = 0
#             noise_term = np.log(2 * np.pi * (2.0 + 1.0) ** 2.0) + np.log(2 * np.pi * (2.0 + 2.0) ** 2.0) + \
#                          np.log(2 * np.pi * (2.0 + 3.0) ** 2.0) + np.log(2 * np.pi * (2.0 + 4.0) ** 2.0)
#
#             assert fitter.scaled_likelihood == -0.5 * (chi_sq_term + noise_term)
#
#         def test__x2_noise_scaling_and_hyper_params__noise_term_comes_out_correct(self):
#             images = [MockChInj(array=10.0 * np.ones((2, 2)), frame_geometry=MockGeometry(), ci_pattern=MockPattern())]
#             masks = [MockChInj(array=np.ma.zeros((2, 2)), frame_geometry=MockGeometry(), ci_pattern=MockPattern())]
#             noises = [MockChInj(array=3.0 * np.ones((2, 2)), frame_geometry=MockGeometry(), ci_pattern=MockPattern())]
#             models = [MockChInj(array=10.0 * np.ones((2, 2)), frame_geometry=MockGeometry(), ci_pattern=MockPattern())]
#             noise_scalings = [[MockChInj(array=np.array([[1.0, 2.0], [3.0, 4.0]]), frame_geometry=MockGeometry(),
#                                          ci_pattern=MockPattern()),
#                                MockChInj(array=np.array([[5.0, 6.0], [7.0, 8.0]]), frame_geometry=MockGeometry(),
#                                          ci_pattern=MockPattern())]]
#
#             hyper_noise = [ci_hyper.HyperCINoise(scale_factor=1.0), ci_hyper.HyperCINoise(scale_factor=2.0)]
#
#             ci_pre_ctis = [MockCIPreCTIs(array=np.zeros((2, 2)), frame_geometry=MockGeometry(), ci_pattern=MockPattern(),
#                                          value=10.0)]
#
#             ci_datas = ci_data.CIDataAnalysis(images=images, masks=masks, noises=noises, ci_pre_ctis=ci_pre_ctis,
#                                               noise_scalings=noise_scalings)
#
#             fitter = fitting.HyperCIFitter(ci_datas, cti_params=MockParams(), cti_settings=MockSettings(),
#                                            hyper_noises=hyper_noise)
#
#             chi_sq_term = 0
#             noise_term = np.log(2 * np.pi * (3.0 + 1.0 + 10.0) ** 2.0) + np.log(2 * np.pi * (3.0 + 2.0 + 12.0) ** 2.0) + \
#                          np.log(2 * np.pi * (3.0 + 3.0 + 14.0) ** 2.0) + np.log(2 * np.pi * (3.0 + 4.0 + 16.0) ** 2.0)
#
#             assert fitter.scaled_likelihood == -0.5 * (chi_sq_term + noise_term)
#
#     class TestFittingImages:
#
#         def test__x2_ci_datas__scaled_noises_are_same_as_calculated_individually(self, ci_datas):
#             hyper_noise = [ci_hyper.HyperCINoise(scale_factor=1.0), ci_hyper.HyperCINoise(scale_factor=2.0)]
#
#             scaled_noises0 = fitting.scaled_noise_from_noise_and_noise_scalings(ci_datas[0].noise,
#                                                                                 ci_datas[0].noise_scalings, hyper_noise)
#             scaled_noises1 = fitting.scaled_noise_from_noise_and_noise_scalings(ci_datas[1].noise,
#                                                                                 ci_datas[1].noise_scalings, hyper_noise)
#
#             fitter = fitting.HyperCIFitter(ci_datas, cti_params=MockParams(), cti_settings=MockSettings(),
#                                            hyper_noises=hyper_noise)
#
#             assert (scaled_noises0 == fitter.scaled_noises[0]).all()
#             assert (scaled_noises1 == fitter.scaled_noises[1]).all()
#             assert fitter.scaled_noises[0].frame_geometry == ci_datas[0].image.frame_geometry
#             assert fitter.scaled_noises[1].frame_geometry == ci_datas[1].image.frame_geometry
#             assert fitter.scaled_noises[0].ci_pattern == ci_datas[0].image.ci_pattern
#             assert fitter.scaled_noises[1].ci_pattern == ci_datas[1].image.ci_pattern
#
#         def test__x2_ci_datas__scaled_chi_squareds_are_same_as_calculated_individually(self, ci_datas):
#             hyper_noise = [ci_hyper.HyperCINoise(scale_factor=1.0), ci_hyper.HyperCINoise(scale_factor=2.0)]
#
#             scaled_noise = fitting.scaled_noise_from_noise_and_noise_scalings(ci_datas[0].noise,
#                                                                               ci_datas[0].noise_scalings, hyper_noise)
#             residuals0 = fitting.residuals_from_image_mask_and_model(ci_datas[0].image, ci_datas[0].mask,
#                                                                      ci_datas[0].ci_pre_cti)
#             scaled_chi_squareds0 = fitting.chi_squareds_from_residuals_and_noise(residuals0, scaled_noise)
#
#             scaled_noise = fitting.scaled_noise_from_noise_and_noise_scalings(ci_datas[1].noise,
#                                                                               ci_datas[1].noise_scalings, hyper_noise)
#             residuals1 = fitting.residuals_from_image_mask_and_model(ci_datas[1].image, ci_datas[1].mask,
#                                                                      ci_datas[1].ci_pre_cti)
#             scaled_chi_squareds1 = fitting.chi_squareds_from_residuals_and_noise(residuals1, scaled_noise)
#
#             fitter = fitting.HyperCIFitter(ci_datas, cti_params=MockParams(), cti_settings=MockSettings(),
#                                            hyper_noises=hyper_noise)
#
#             assert (scaled_chi_squareds0 == fitter.scaled_chi_squareds[0]).all()
#             assert (scaled_chi_squareds1 == fitter.scaled_chi_squareds[1]).all()
#             assert fitter.scaled_chi_squareds[0].frame_geometry == ci_datas[0].image.frame_geometry
#             assert fitter.scaled_chi_squareds[1].frame_geometry == ci_datas[1].image.frame_geometry
#             assert fitter.scaled_chi_squareds[0].ci_pattern == ci_datas[0].image.ci_pattern
#             assert fitter.scaled_chi_squareds[1].ci_pattern == ci_datas[1].image.ci_pattern
#
#         def test__x2_ci_datas__likelihoods_are_same_as_calculated_individually(self, ci_datas):
#             hyper_noise = [ci_hyper.HyperCINoise(scale_factor=1.0), ci_hyper.HyperCINoise(scale_factor=2.0)]
#
#             scaled_noise0 = fitting.scaled_noise_from_noise_and_noise_scalings(ci_datas[0].noise,
#                                                                                ci_datas[0].noise_scalings,
#                                                                                hyper_noise)
#             residuals0 = fitting.residuals_from_image_mask_and_model(ci_datas[0].image, ci_datas[0].mask,
#                                                                      ci_datas[0].ci_pre_cti)
#             scaled_chi_squareds0 = fitting.chi_squareds_from_residuals_and_noise(residuals0, scaled_noise0)
#             scaled_chi_squared_term0 = fitting.chi_squared_term_from_chi_squareds(scaled_chi_squareds0)
#             scaled_noise_term0 = fitting.noise_term_from_mask_and_noise(ci_datas[0].mask, scaled_noise0)
#             scaled_likelihood0 = fitting.likelihood_from_chi_squared_and_noise_terms(scaled_chi_squared_term0,
#                                                                                      scaled_noise_term0)
#
#             scaled_noise1 = fitting.scaled_noise_from_noise_and_noise_scalings(ci_datas[1].noise,
#                                                                                ci_datas[1].noise_scalings, hyper_noise)
#             residuals1 = fitting.residuals_from_image_mask_and_model(ci_datas[1].image, ci_datas[1].mask,
#                                                                      ci_datas[1].ci_pre_cti)
#             scaled_chi_squareds1 = fitting.chi_squareds_from_residuals_and_noise(residuals1, scaled_noise1)
#             scaled_chi_squared_term1 = fitting.chi_squared_term_from_chi_squareds(scaled_chi_squareds1)
#             scaled_noise_term1 = fitting.noise_term_from_mask_and_noise(ci_datas[1].mask, scaled_noise1)
#             scaled_likelihood1 = fitting.likelihood_from_chi_squared_and_noise_terms(scaled_chi_squared_term1,
#                                                                                      scaled_noise_term1)
#
#             fitter = fitting.HyperCIFitter(ci_datas, cti_params=MockParams(), cti_settings=MockSettings(),
#                                            hyper_noises=hyper_noise)
#
#             assert (scaled_likelihood0 + scaled_likelihood1 == fitter.scaled_likelihood).all()

