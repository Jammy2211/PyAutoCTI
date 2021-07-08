import autocti as ac


class TestFitDatasetLine:
    def test__fit_quantities_same_as_calculated_individually(
        self, dataset_line_7, mask_1d_7_unmasked
    ):

        masked_dataset_line_7 = dataset_line_7.apply_mask(mask=mask_1d_7_unmasked)

        post_cti_line = ac.Array1D.full(
            fill_value=1.0,
            shape_native=masked_dataset_line_7.data.shape_native,
            pixel_scales=1.0,
        ).native

        fit = ac.FitDatasetLine(
            dataset_line=masked_dataset_line_7, post_cti_line=post_cti_line
        )

        residual_map = ac.util.fit.residual_map_with_mask_from(
            data=masked_dataset_line_7.data,
            mask=mask_1d_7_unmasked,
            model_data=post_cti_line,
        )

        assert (fit.residual_map == residual_map).all()

        chi_squared_map = ac.util.fit.chi_squared_map_with_mask_from(
            residual_map=residual_map,
            noise_map=masked_dataset_line_7.noise_map,
            mask=mask_1d_7_unmasked,
        )

        assert (fit.chi_squared_map == chi_squared_map).all()

        chi_squared = ac.util.fit.chi_squared_with_mask_from(
            chi_squared_map=chi_squared_map, mask=mask_1d_7_unmasked
        )

        noise_normalization = ac.util.fit.noise_normalization_with_mask_from(
            noise_map=masked_dataset_line_7.noise_map, mask=mask_1d_7_unmasked
        )

        log_likelihood = ac.util.fit.log_likelihood_from(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert fit.log_likelihood == log_likelihood
