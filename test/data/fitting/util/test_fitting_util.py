from autocti.data.fitting.util import fitting_util
from autocti.data.charge_injection import ci_hyper

import numpy as np
import pytest


class TestResiduals:

    def test__model_mathces_data__residuals_all_0s(self):

        image = [10.0 * np.ones((2, 2))]
        mask = [np.zeros((2, 2))]
        model = [10.0 * np.ones((2, 2))]

        residual = fitting_util.residuals_from_datas_masks_and_model_datas(image, mask, model)

        assert (residual == np.zeros((2, 2))).all()

    def test__model_data_mismatch__residuals_non_0(self):

        image = [10.0 * np.ones((2, 2))]
        mask = [np.zeros((2, 2))]
        model = [np.array([[11, 10],
                          [9, 8]])]

        residual = fitting_util.residuals_from_datas_masks_and_model_datas(image, mask, model)

        assert (residual == np.array([[-1, 0],
                                       [1, 2]])).all()

    def test__model_data_mismatch__masked_residuals_set_to_0(self):

        image = [10.0 * np.ones((2, 2))]
        mask = [np.array([[True, False],
                            [False, True]])]
        model = [np.array([[11, 10],
                          [9, 8]])]

        residual = fitting_util.residuals_from_datas_masks_and_model_datas(image, mask, model)

        assert (residual == np.array([[0, 0],
                                       [1, 0]])).all()

    def test__same_as_above__but_lists_of_images(self):

        image = [10.0 * np.ones((2, 2)), 10.0 * np.ones((2, 2))]
        mask = [np.array([[True, False], [False, True]]), np.ma.zeros((2, 2))]
        model = [np.array([[11, 10], [9, 8]]), np.array([[11, 10], [9, 8]])]

        residuals = fitting_util.residuals_from_datas_masks_and_model_datas(image, mask, model)

        assert (residuals[0] == np.array([[0, 0], [1, 0]])).all()
        assert (residuals[1] == np.array([[-1, 0], [1, 2]])).all()


class TestChiSquareds:

    def test__model_mathces_data__chi_sq_all_0s(self):

        image = [10.0 * np.ones((2, 2))]
        mask = [np.zeros((2, 2))]
        noise = [4.0 * np.ones((2, 2))]
        model = [10.0 * np.ones((2, 2))]

        residual = fitting_util.residuals_from_datas_masks_and_model_datas(image, mask, model)
        chi_squared = fitting_util.chi_squareds_from_residuals_and_noise(residual, noise)

        assert (chi_squared == np.zeros((2, 2))).all()

    def test__model_data_mismatch__chi_sq_non_0(self):
        image = 10.0 * np.ones((2, 2))
        mask = np.zeros((2, 2))
        noise = 2.0 * np.ones((2, 2))
        model = np.array([[11, 10],
                          [9, 8]])

        residual = fitting_util.residuals_from_datas_masks_and_model_datas(image, mask, model)
        chi_squared = fitting_util.chi_squareds_from_residuals_and_noise(residual, noise)

        assert (chi_squared == (np.array([[1 / 4, 0],
                                          [1 / 4, 1]]))).all()

    def test__model_data_mismatch__masked_chi_sqs_set_to_0(self):

        image = 10.0 * np.ones((2, 2))
        mask = np.array([[True, False],
                            [False, True]])
        noise = 2.0 * np.ones((2, 2))
        model = np.array([[11, 10],
                          [9, 8]])

        residual = fitting_util.residuals_from_datas_masks_and_model_datas(image, mask, model)
        chi_squared = fitting_util.chi_squareds_from_residuals_and_noise(residual, noise)

        assert (chi_squared == (np.array([[0, 0],
                                          [1 / 4, 0]]))).all()

    def test__same_as_above__but_lists_of_images(self):

        image = [10.0 * np.ones((2, 2)), 10.0 * np.ones((2, 2))]
        mask = [np.array([[True, False], [False, True]]), np.ma.zeros((2, 2))]
        noise = [2.0 * np.ones((2, 2)), 2.0 * np.ones((2, 2))]
        model = [np.array([[11, 10], [9, 8]]), np.array([[11, 10], [9, 8]])]

        residuals = fitting_util.residuals_from_datas_masks_and_model_datas(image, mask, model)
        chi_squareds = fitting_util.chi_squareds_from_residuals_and_noise(residuals, noise)

        assert (chi_squareds[0] == (np.array([[0, 0], [1 / 4, 0]]))).all()
        assert (chi_squareds[1] == (np.array([[1 / 4, 0], [1 / 4, 1]]))).all()


class TestLikelihood:

    def test__model_mathces_data__no_mask__noise_1s__lh_is_noise_term(self):

        image = [10.0 * np.ones((2, 2))]
        mask = [np.zeros((2, 2))]
        noise = [2.0 * np.ones((2, 2))]
        model = [10.0 * np.ones((2, 2))]

        residual = fitting_util.residuals_from_datas_masks_and_model_datas(image, mask, model)
        chi_squared = fitting_util.chi_squareds_from_residuals_and_noise(residual, noise)
        chi_squared_term = fitting_util.chi_squared_term_from_chi_squareds(chi_squared)
        noise_term = fitting_util.noise_term_from_mask_and_noise(mask, noise)
        likelihood = fitting_util.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)

        chi_sq_term = 0
        noise_term = 4.0 * np.log(2 * np.pi * 4.0)

        assert likelihood == -0.5 * (chi_sq_term + noise_term)

    def test__model_data_mismatch__no_mask__chi_sq_term_contributes_to_lh(self):

        image = [10.0 * np.ones((2, 2))]
        mask = [np.zeros((2, 2))]
        noise = [2.0 * np.ones((2, 2))]
        model = [np.array([[11, 10], [9, 8]])]

        residuals = fitting_util.residuals_from_datas_masks_and_model_datas(image, mask, model)
        chi_squareds = fitting_util.chi_squareds_from_residuals_and_noise(residuals, noise)
        chi_squared_term = fitting_util.chi_squared_term_from_chi_squareds(chi_squareds)
        noise_term = fitting_util.noise_term_from_mask_and_noise(mask, noise)
        likelihood = fitting_util.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)
        # chi squared = 0.25, 0, 0.25, 1.0
        # likelihood = -0.5*(0.25+0+0.25+1.0)

        chi_sq_term = 1.5
        noise_term = 4.0 * np.log(2 * np.pi * 4.0)

        assert likelihood == -0.5 * (chi_sq_term + noise_term)

    def test__same_as_above_but_different_noise_in_each_pixel(self):

        image = [10.0 * np.ones((2, 2))]
        mask = [np.zeros((2, 2))]
        noise = [np.array([[1.0, 2.0],
                          [3.0, 4.0]])]
        model = [np.array([[11, 10],
                          [9, 8]])]

        residuals = fitting_util.residuals_from_datas_masks_and_model_datas(image, mask, model)
        chi_squareds = fitting_util.chi_squareds_from_residuals_and_noise(residuals, noise)
        chi_squared_term = fitting_util.chi_squared_term_from_chi_squareds(chi_squareds)
        noise_term = fitting_util.noise_term_from_mask_and_noise(mask, noise)
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
        noise = [2.0 * np.ones((2, 2))]
        model = [np.array([[11, 10],
                          [9, 8]])]

        residuals = fitting_util.residuals_from_datas_masks_and_model_datas(image, mask, model)
        chi_squareds = fitting_util.chi_squareds_from_residuals_and_noise(residuals, noise)
        chi_squared_term = fitting_util.chi_squared_term_from_chi_squareds(chi_squareds)
        noise_term = fitting_util.noise_term_from_mask_and_noise(mask, noise)
        likelihood = fitting_util.likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term)

        # chi squared = 0, 0.25, (0.25 and 1.0 are masked)

        chi_sq_term = 0.25
        noise_term = np.log(2 * np.pi * 4.0) + np.log(2 * np.pi * 4.0)

        assert likelihood == pytest.approx(-0.5 * (chi_sq_term + noise_term), 1e-4)

    def test__same_as_above__but_x2_lists_of_data(self):

        images = [10.0 * np.ones((2, 2)), 10.0 * np.ones((2, 2))]
        masks = [np.array([[True, False],
                            [False, True]]), np.zeros((2, 2))]
        noises = [2.0 * np.ones((2, 2)), np.array([[1.0, 2.0],
                          [3.0, 4.0]])]
        models = [np.array([[11, 10],
                          [9, 8]]), np.array([[11, 10],
                          [9, 8]])]

        residuals = fitting_util.residuals_from_datas_masks_and_model_datas(images, masks, models)
        chi_squareds = fitting_util.chi_squareds_from_residuals_and_noise(residuals, noises)
        chi_squared_terms = fitting_util.chi_squared_term_from_chi_squareds(chi_squareds)
        noise_terms = fitting_util.noise_term_from_mask_and_noise(masks, noises)
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


class TestScaledNoises:

    def test__image_and_pre_cti_not_identical__noise_scalings_are_0s__no_noise_scaling(self):

        noise = [2.0 * np.ones((2, 2))]
        noise_scalings = [np.array([[0.0, 0.0], [0.0, 0.0]])]
        hyper_noise = [ci_hyper.HyperCINoise(scale_factor=1.0)]

        scaled_noise = fitting_util.scaled_noise_maps_from_noise_maps_and_noise_scalings(noise, noise_scalings,
                                                                                         hyper_noise)

        assert (scaled_noise == (np.array([[2.0, 2.0],
                                           [2.0, 2.0]]))).all()

    def test__image_and_pre_cti_not_identical__scaled_factor_is_0__no_noise_scaling(self):

        noise = [2.0 * np.ones((2, 2))]
        noise_scalings = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        hyper_noise = [ci_hyper.HyperCINoise(scale_factor=0.0)]

        scaled_noise = fitting_util.scaled_noise_maps_from_noise_maps_and_noise_scalings(noise, noise_scalings,
                                                                                         hyper_noise)

        assert (scaled_noise == (np.array([[2.0, 2.0],
                                           [2.0, 2.0]]))).all()

    def test__image_and_pre_cti_not_identical__chi_sq_is_scaled_by_noise(self):

        noise = [2.0 * np.ones((2, 2))]
        noise_scalings = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        hyper_noise = [ci_hyper.HyperCINoise(scale_factor=1.0)]

        scaled_noise = fitting_util.scaled_noise_maps_from_noise_maps_and_noise_scalings(noise, noise_scalings,
                                                                                         hyper_noise)

        assert (scaled_noise == (np.array([[3.0, 4.0],
                                           [5.0, 6.0]]))).all()

    def test__x2_noise_scaling_and_hyper_params__noise_term_comes_out_correct(self):

        noise = [2.0 * np.ones((2, 2))]
        noise_scalings = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[1.0, 2.0], [3.0, 4.0]])]
        hyper_noise = [ci_hyper.HyperCINoise(scale_factor=1.0), ci_hyper.HyperCINoise(scale_factor=2.0)]

        scaled_noise = fitting_util.scaled_noise_maps_from_noise_maps_and_noise_scalings(noise, noise_scalings, hyper_noise)

        assert (scaled_noise == (np.array([[5.0, 8.0],
                                           [11.0, 14.0]]))).all()
