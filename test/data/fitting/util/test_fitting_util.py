from autocti.data.fitting.util import fitting_util
from autocti.data.fitting import fitting_data
from autocti.data.charge_injection import ci_hyper

import numpy as np
import pytest


class TestResiduals:

    def test__model_mathces_data__residuals_all_0s(self):

        image = [10.0 * np.ones((2, 2))]
        mask = [np.zeros((2, 2))]
        model_image = [10.0 * np.ones((2, 2))]

        residual = fitting_util.residuals_from_datas_masks_and_model_datas(image, mask, model_image)

        assert (residual == np.zeros((2, 2))).all()

    def test__model_data_mismatch__residuals_non_0(self):

        image = [10.0 * np.ones((2, 2))]
        mask = [np.zeros((2, 2))]
        model_image = [np.array([[11, 10],
                          [9, 8]])]

        residual = fitting_util.residuals_from_datas_masks_and_model_datas(image, mask, model_image)

        assert (residual == np.array([[-1, 0],
                                       [1, 2]])).all()

    def test__model_data_mismatch__masked_residuals_set_to_0(self):

        image = [10.0 * np.ones((2, 2))]
        mask = [np.array([[True, False],
                            [False, True]])]
        model_image = [np.array([[11, 10],
                          [9, 8]])]

        residual = fitting_util.residuals_from_datas_masks_and_model_datas(image, mask, model_image)

        assert (residual == np.array([[0, 0],
                                       [1, 0]])).all()

    def test__same_as_above__but_lists_of_images(self):

        image = [10.0 * np.ones((2, 2)), 10.0 * np.ones((2, 2))]
        mask = [np.array([[True, False], [False, True]]), np.ma.zeros((2, 2))]
        model_image = [np.array([[11, 10], [9, 8]]), np.array([[11, 10], [9, 8]])]

        residuals = fitting_util.residuals_from_datas_masks_and_model_datas(image, mask, model_image)

        assert (residuals[0] == np.array([[0, 0], [1, 0]])).all()
        assert (residuals[1] == np.array([[-1, 0], [1, 2]])).all()


class TestChiSquareds:

    def test__model_mathces_data__chi_sq_all_0s(self):

        image = [10.0 * np.ones((2, 2))]
        mask = [np.zeros((2, 2))]
        noise_map = [4.0 * np.ones((2, 2))]
        model_image = [10.0 * np.ones((2, 2))]

        residual = fitting_util.residuals_from_datas_masks_and_model_datas(image, mask, model_image)
        chi_squared = fitting_util.chi_squareds_from_residuals_and_noise(residual, noise_map)

        assert (chi_squared == np.zeros((2, 2))).all()

    def test__model_data_mismatch__chi_sq_non_0(self):
        image = 10.0 * np.ones((2, 2))
        mask = np.zeros((2, 2))
        noise_map = 2.0 * np.ones((2, 2))
        model_image = np.array([[11, 10],
                          [9, 8]])

        residual = fitting_util.residuals_from_datas_masks_and_model_datas(image, mask, model_image)
        chi_squared = fitting_util.chi_squareds_from_residuals_and_noise(residual, noise_map)

        assert (chi_squared == (np.array([[1 / 4, 0],
                                          [1 / 4, 1]]))).all()

    def test__model_data_mismatch__masked_chi_sqs_set_to_0(self):

        image = 10.0 * np.ones((2, 2))
        mask = np.array([[True, False],
                            [False, True]])
        noise_map = 2.0 * np.ones((2, 2))
        model_image = np.array([[11, 10],
                          [9, 8]])

        residual = fitting_util.residuals_from_datas_masks_and_model_datas(image, mask, model_image)
        chi_squared = fitting_util.chi_squareds_from_residuals_and_noise(residual, noise_map)

        assert (chi_squared == (np.array([[0, 0],
                                          [1 / 4, 0]]))).all()

    def test__same_as_above__but_lists_of_images(self):

        image = [10.0 * np.ones((2, 2)), 10.0 * np.ones((2, 2))]
        mask = [np.array([[True, False], [False, True]]), np.ma.zeros((2, 2))]
        noise_map = [2.0 * np.ones((2, 2)), 2.0 * np.ones((2, 2))]
        model_image = [np.array([[11, 10], [9, 8]]), np.array([[11, 10], [9, 8]])]

        residuals = fitting_util.residuals_from_datas_masks_and_model_datas(image, mask, model_image)
        chi_squareds = fitting_util.chi_squareds_from_residuals_and_noise(residuals, noise_map)

        assert (chi_squareds[0] == (np.array([[0, 0], [1 / 4, 0]]))).all()
        assert (chi_squareds[1] == (np.array([[1 / 4, 0], [1 / 4, 1]]))).all()


class TestLikelihood:

    def test__model_mathces_data__no_mask__noise_1s__lh_is_noise_term(self):

        image = [10.0 * np.ones((2, 2))]
        mask = [np.zeros((2, 2))]
        noise_map = [2.0 * np.ones((2, 2))]
        model_image = [10.0 * np.ones((2, 2))]

        residual = fitting_util.residuals_from_datas_masks_and_model_datas(image, mask, model_image)
        chi_squared = fitting_util.chi_squareds_from_residuals_and_noise(residual, noise_map)
        chi_squared_term = fitting_util.chi_squared_term_from_chi_squareds(chi_squared)
        noise_term = fitting_util.noise_term_from_mask_and_noise(mask, noise_map)
        likelihood = fitting_util.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)

        chi_sq_term = 0
        noise_term = 4.0 * np.log(2 * np.pi * 4.0)

        assert likelihood == -0.5 * (chi_sq_term + noise_term)

    def test__model_data_mismatch__no_mask__chi_sq_term_contributes_to_lh(self):

        image = [10.0 * np.ones((2, 2))]
        mask = [np.zeros((2, 2))]
        noise_map = [2.0 * np.ones((2, 2))]
        model_image = [np.array([[11, 10], [9, 8]])]

        residuals = fitting_util.residuals_from_datas_masks_and_model_datas(image, mask, model_image)
        chi_squareds = fitting_util.chi_squareds_from_residuals_and_noise(residuals, noise_map)
        chi_squared_term = fitting_util.chi_squared_term_from_chi_squareds(chi_squareds)
        noise_term = fitting_util.noise_term_from_mask_and_noise(mask, noise_map)
        likelihood = fitting_util.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)
        # chi squared = 0.25, 0, 0.25, 1.0
        # likelihood = -0.5*(0.25+0+0.25+1.0)

        chi_sq_term = 1.5
        noise_term = 4.0 * np.log(2 * np.pi * 4.0)

        assert likelihood == -0.5 * (chi_sq_term + noise_term)

    def test__same_as_above_but_different_noise_in_each_pixel(self):

        image = [10.0 * np.ones((2, 2))]
        mask = [np.zeros((2, 2))]
        noise_map = [np.array([[1.0, 2.0],
                          [3.0, 4.0]])]
        model_image = [np.array([[11, 10],
                          [9, 8]])]

        residuals = fitting_util.residuals_from_datas_masks_and_model_datas(image, mask, model_image)
        chi_squareds = fitting_util.chi_squareds_from_residuals_and_noise(residuals, noise_map)
        chi_squared_term = fitting_util.chi_squared_term_from_chi_squareds(chi_squareds)
        noise_term = fitting_util.noise_term_from_mask_and_noise(mask, noise_map)
        likelihood = fitting_util.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)

        # chi squared = (1.0/1.0)**2, (0.0), (-1.0/3.0)**2.0, (2.0/4.0)**2.0

        chi_sq_term = 1.0 + (1.0 / 9.0) + 0.25
        noise_term = np.log(2 * np.pi * 1.0) + np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 9.0) + np.log(
            2 * np.pi * 16.0)

        assert likelihood == pytest.approx(-0.5 * (chi_sq_term + noise_term), 1e-4)

    def test__model_data_mismatch__mask_certain_pixels__lh_non_0(self):

        image = [10.0 * np.ones((2, 2))]
        mask = [np.array([[True, False],
                            [False, True]])]
        noise_map = [2.0 * np.ones((2, 2))]
        model_image = [np.array([[11, 10],
                          [9, 8]])]

        residuals = fitting_util.residuals_from_datas_masks_and_model_datas(image, mask, model_image)
        chi_squareds = fitting_util.chi_squareds_from_residuals_and_noise(residuals, noise_map)
        chi_squared_term = fitting_util.chi_squared_term_from_chi_squareds(chi_squareds)
        noise_term = fitting_util.noise_term_from_mask_and_noise(mask, noise_map)
        likelihood = fitting_util.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)

        # chi squared = 0, 0.25, (0.25 and 1.0 are masked)

        chi_sq_term = 0.25
        noise_term = np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0)

        assert likelihood == pytest.approx(-0.5 * (chi_sq_term + noise_term), 1e-4)

    def test__same_as_above__but_x2_lists_of_data(self):

        images = [10.0 * np.ones((2, 2)), 10.0 * np.ones((2, 2))]
        masks = [np.array([[True, False],
                            [False, True]]), np.zeros((2, 2))]
        noise_maps = [2.0 * np.ones((2, 2)), np.array([[1.0, 2.0],
                          [3.0, 4.0]])]
        models = [np.array([[11, 10],
                          [9, 8]]), np.array([[11, 10],
                          [9, 8]])]

        residuals = fitting_util.residuals_from_datas_masks_and_model_datas(images, masks, model_images)
        chi_squareds = fitting_util.chi_squareds_from_residuals_and_noise(residuals, noise_maps)
        chi_squared_terms = fitting_util.chi_squared_term_from_chi_squareds(chi_squareds)
        noise_terms = fitting_util.noise_term_from_mask_and_noise(masks, noise_maps)
        likelihoods = fitting_util.likelihood_from_chi_squared_and_noise_terms(chi_squared_terms, noise_terms)

        # chi squared = 0, 0.25, (0.25 and 1.0 are masked)

        chi_sq_term = 0.25
        noise_term = np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0)

        assert chi_squared_terms[0] == chi_sq_term
        assert noise_terms[0] == noise_term
        assert likelihoods[0] == pytest.approx(-0.5 * (chi_sq_term + noise_term), 1e-4)

        chi_sq_term = 1.0 + (1.0 / 9.0) + 0.25
        noise_term = np.log(2 * np.pi * 1.0) + np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 9.0) + np.log(
            2 * np.pi * 16.0)

        assert chi_squared_terms[1] == chi_sq_term
        assert noise_terms[1] == noise_term
        assert likelihoods[1] == pytest.approx(-0.5 * (chi_sq_term + noise_term), 1e-4)


class TestScaledNoiseMap:

    def test__image_and_pre_cti_not_identical__noise_scalings_are_0s__no_noise_scaling(self):

        noise_map = 2.0 * np.ones((2, 2))
        noise_scalings = [np.array([[0.0, 0.0], [0.0, 0.0]])]
        hyper_noise = [ci_hyper.HyperCINoise(scale_factor=1.0)]

        scaled_noise_map = fitting_util.scaled_noise_map_from_noise_map_and_noise_scalings(noise_map=noise_map,
                                                       noise_scalings=noise_scalings, hyper_noises=hyper_noise)

        assert (scaled_noise_map == (np.array([[2.0, 2.0],
                                           [2.0, 2.0]]))).all()

    def test__image_and_pre_cti_not_identical__scaled_factor_is_0__no_noise_scaling(self):

        noise_map = 2.0 * np.ones((2, 2))
        noise_scalings = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        hyper_noise = [ci_hyper.HyperCINoise(scale_factor=0.0)]

        scaled_noise_map = fitting_util.scaled_noise_map_from_noise_map_and_noise_scalings(noise_map=noise_map,
                                                       noise_scalings=noise_scalings, hyper_noises=hyper_noise)

        assert (scaled_noise_map == (np.array([[2.0, 2.0],
                                           [2.0, 2.0]]))).all()

    def test__image_and_pre_cti_not_identical__chi_sq_is_scaled_by_noise(self):

        noise_map = 2.0 * np.ones((2, 2))
        noise_scalings = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        hyper_noise = [ci_hyper.HyperCINoise(scale_factor=1.0)]

        scaled_noise_map = fitting_util.scaled_noise_map_from_noise_map_and_noise_scalings(noise_map=noise_map,
                                                       noise_scalings=noise_scalings, hyper_noises=hyper_noise)

        assert (scaled_noise_map == (np.array([[3.0, 4.0],
                                           [5.0, 6.0]]))).all()

    def test__x2_noise_scaling_and_hyper_params__noise_term_comes_out_correct(self):

        noise_map = 2.0 * np.ones((2, 2))
        noise_scalings = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[1.0, 2.0], [3.0, 4.0]])]
        hyper_noise = [ci_hyper.HyperCINoise(scale_factor=1.0), ci_hyper.HyperCINoise(scale_factor=2.0)]

        scaled_noise_map = fitting_util.scaled_noise_map_from_noise_map_and_noise_scalings(noise_map=noise_map,
                                                       noise_scalings=noise_scalings, hyper_noises=hyper_noise)

        assert (scaled_noise_map == (np.array([[5.0, 8.0],
                                              [11.0, 14.0]]))).all()

    def test__same_as_above__but_use_fitting_hyper_image(self):

        fitting_hyper_images = [fitting_data.FittingHyperImage(image=None, noise_map=2.0*np.ones((2,2)), mask=None,
                                                             noise_scalings=[np.array([[1.0, 2.0], [3.0, 4.0]]),
                                                                             np.array([[1.0, 2.0], [3.0, 4.0]])])]

        hyper_noise = [ci_hyper.HyperCINoise(scale_factor=1.0), ci_hyper.HyperCINoise(scale_factor=2.0)]

        scaled_noise_map = fitting_util.scaled_noise_maps_from_fitting_hyper_images_and_noise_scalings(
            fitting_hyper_images=fitting_hyper_images, hyper_noises=hyper_noise)

        assert (scaled_noise_map == (np.array([[5.0, 8.0],
                                              [11.0, 14.0]]))).all()

    def test__same_as_above__but_use_x2__fitting_hyper_images(self):

        fitting_hyper_images = [fitting_data.FittingHyperImage(image=None, noise_map=2.0 * np.ones((2, 2)), mask=None,
                                           noise_scalings=[np.array([[1.0, 2.0], [3.0, 4.0]]),
                                                           np.array([[1.0, 2.0], [3.0, 4.0]])]),
                                fitting_data.FittingHyperImage(image=None, noise_map=2.0 * np.ones((2, 2)), mask=None,
                                           noise_scalings=[np.array([[2.0, 2.0], [3.0, 4.0]]),
                                                           np.array([[2.0, 2.0], [3.0, 4.0]])])]

        hyper_noise = [ci_hyper.HyperCINoise(scale_factor=1.0), ci_hyper.HyperCINoise(scale_factor=2.0)]

        scaled_noise_map = fitting_util.scaled_noise_maps_from_fitting_hyper_images_and_noise_scalings(
            fitting_hyper_images=fitting_hyper_images, hyper_noises=hyper_noise)

        assert (scaled_noise_map[0] == (np.array([[5.0, 8.0],
                                                 [11.0, 14.0]]))).all()
        assert (scaled_noise_map[1] == (np.array([[8.0, 8.0],
                                                 [11.0, 14.0]]))).all()