import numpy as np
import pytest
import autocti as ac


class TestFitImaging:
    def test__image_and_model_are_identical__no_masking__check_values_are_correct(self):

        mask = ac.Mask.manual(
            mask_2d=np.array([[False, False], [False, False]]), pixel_scales=(1.0, 1.0)
        )

        data = ac.MaskedArray.manual_2d(
            array=np.array([[1.0, 2.0], [3.0, 4.0]]), mask=mask
        )
        noise_map = ac.MaskedArray.manual_2d(
            array=np.array([[2.0, 2.0], [2.0, 2.0]]), mask=mask
        )

        imaging = ac.Imaging(image=data, noise_map=noise_map)

        masked_imaging = ac.MaskedImaging(imaging=imaging, mask=mask)

        model_image = ac.MaskedArray.manual_2d(
            array=np.array([[1.0, 2.0], [3.0, 4.0]]), mask=mask
        )

        fit = ac.FitImaging(masked_imaging=masked_imaging, model_image=model_image)

        assert (fit.mask == np.array([[False, False], [False, False]])).all()

        assert (fit.image == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert (fit.noise_map == np.array([[2.0, 2.0], [2.0, 2.0]])).all()

        assert fit.signal_to_noise_map == pytest.approx(
            np.array([[0.5, 1.0], [1.5, 2.0]]), 1.0e-4
        )

        assert (fit.model_image == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert (fit.residual_map == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
        assert (fit.normalized_residual_map == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
        assert (fit.chi_squared_map == np.array([[0.0, 0.0], [0.0, 0.0]])).all()

        assert fit.chi_squared == 0.0
        assert fit.reduced_chi_squared == 0.0
        assert fit.noise_normalization == np.sum(np.log(2 * np.pi * noise_map ** 2.0))
        assert fit.log_likelihood == -0.5 * (fit.chi_squared + fit.noise_normalization)

    def test__image_and_model_are_different__inclue_masking__check_values_are_correct(
        self
    ):

        mask = ac.Mask.manual(
            mask_2d=np.array([[False, False], [True, False]]), pixel_scales=(1.0, 1.0)
        )

        data = ac.MaskedArray.manual_2d(
            array=np.array([[1.0, 2.0], [10.0, 4.0]]), mask=mask
        )
        noise_map = ac.MaskedArray.manual_2d(
            array=np.array([[2.0, 2.0], [2.0, 2.0]]), mask=mask
        )

        imaging = ac.Imaging(image=data, noise_map=noise_map)

        masked_imaging = ac.MaskedImaging(imaging=imaging, mask=mask)

        model_image = ac.MaskedArray.manual_2d(
            array=np.array([[1.0, 2.0], [100.0, 3.0]]), mask=mask
        )

        fit = ac.FitImaging(masked_imaging=masked_imaging, model_image=model_image)

        assert (fit.mask == np.array([[False, False], [True, False]])).all()

        assert (fit.image == np.array([[1.0, 2.0], [0.0, 4.0]])).all()

        assert (fit.noise_map == np.array([[2.0, 2.0], [0.0, 2.0]])).all()

        assert fit.signal_to_noise_map == pytest.approx(
            np.array([[0.5, 1.0], [0.0, 2.0]]), 1.0e-4
        )

        assert (fit.model_image == np.array([[1.0, 2.0], [0.0, 3.0]])).all()

        assert (fit.residual_map == np.array([[0.0, 0.0], [0.0, 1.0]])).all()

        assert (fit.normalized_residual_map == np.array([[0.0, 0.0], [0.0, 0.5]])).all()

        assert (fit.chi_squared_map == np.array([[0.0, 0.0], [0.0, 0.25]])).all()

        assert fit.chi_squared == 0.25
        assert fit.reduced_chi_squared == 0.25 / 3.0
        assert fit.noise_normalization == 3.0 * (np.log(2 * np.pi * 2.0 ** 2.0))
        assert fit.log_likelihood == -0.5 * (fit.chi_squared + fit.noise_normalization)
