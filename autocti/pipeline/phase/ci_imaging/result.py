from autocti.pipeline.phase.dataset.result import Result as DatasetResult


class Result(DatasetResult):
    @property
    def max_log_likelihood_full_fits(self):
        return self.analysis.fits_full_dataset_from_instance(
            instance=self.instance, hyper_noise_scale=True
        )

    @property
    def max_log_likelihood_full_fits_no_hyper_scaling(self):
        return self.analysis.fits_full_dataset_from_instance(
            instance=self.instance, hyper_noise_scale=False
        )

    @property
    def noise_scaling_maps_list_of_ci_regions(self):

        return list(
            map(
                lambda fit: fit.chi_squared_map_of_ci_regions,
                self.max_log_likelihood_full_fits_no_hyper_scaling,
            )
        )

    @property
    def noise_scaling_maps_list_of_parallel_trails(self):

        return list(
            map(
                lambda max_log_likelihood_full_fit: max_log_likelihood_full_fit.chi_squared_map_of_parallel_trails,
                self.max_log_likelihood_full_fits_no_hyper_scaling,
            )
        )

    @property
    def noise_scaling_maps_list_of_serial_trails(self):

        return list(
            map(
                lambda max_log_likelihood_full_fit: max_log_likelihood_full_fit.chi_squared_map_of_serial_trails,
                self.max_log_likelihood_full_fits_no_hyper_scaling,
            )
        )

    @property
    def noise_scaling_maps_list_of_serial_overscan_no_trails(self):

        return list(
            map(
                lambda max_log_likelihood_full_fit: max_log_likelihood_full_fit.chi_squared_map_of_serial_overscan_no_trails,
                self.max_log_likelihood_full_fits_no_hyper_scaling,
            )
        )
