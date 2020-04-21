import autofit as af


class Result(af.Result):
    def __init__(
        self,
        instance,
        log_likelihood,
        previous_model,
        gaussian_tuples,
        analysis,
        optimizer,
    ):
        """
        The result of a phase
        """
        super().__init__(
            instance=instance,
            log_likelihood=log_likelihood,
            previous_model=previous_model,
            gaussian_tuples=gaussian_tuples,
        )

        self.analysis = analysis
        self.optimizer = optimizer

    @property
    def most_likely_extracted_fits(self):
        return self.analysis.fits_of_ci_data_extracted_for_instance(
            instance=self.instance
        )

    @property
    def most_likely_full_fits(self):
        return self.analysis.fits_of_ci_data_full_for_instance(instance=self.instance)

    @property
    def most_likely_full_fits_no_hyper_scaling(self):
        return self.analysis.fits_of_ci_data_full_for_instance(instance=self.instance)

    @property
    def noise_scaling_maps_list_of_ci_regions(self):

        return list(
            map(
                lambda fit: fit.chi_squared_map_of_ci_regions,
                self.most_likely_full_fits_no_hyper_scaling,
            )
        )

    @property
    def noise_scaling_maps_list_of_parallel_trails(self):

        return list(
            map(
                lambda most_likely_full_fit: most_likely_full_fit.chi_squared_map_of_parallel_trails,
                self.most_likely_full_fits_no_hyper_scaling,
            )
        )

    @property
    def noise_scaling_maps_list_of_serial_trails(self):

        return list(
            map(
                lambda most_likely_full_fit: most_likely_full_fit.chi_squared_map_of_serial_trails,
                self.most_likely_full_fits_no_hyper_scaling,
            )
        )

    @property
    def noise_scaling_maps_list_of_serial_overscan_no_trails(self):

        return list(
            map(
                lambda most_likely_full_fit: most_likely_full_fit.chi_squared_map_of_serial_overscan_no_trails,
                self.most_likely_full_fits_no_hyper_scaling,
            )
        )
