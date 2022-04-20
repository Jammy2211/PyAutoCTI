from autocti.model.result import ResultDataset


class ResultImagingCI(ResultDataset):
    @property
    def max_log_likelihood_full_fit(self):
        return self.analysis.fit_full_dataset_via_instance_from(
            instance=self.instance, hyper_noise_scale=True
        )

    @property
    def max_log_likelihood_full_fit_no_hyper_scaling(self):
        return self.analysis.fit_full_dataset_via_instance_from(
            instance=self.instance, hyper_noise_scale=False
        )

    @property
    def noise_scaling_map_of_regions_ci(self):
        return (
            self.max_log_likelihood_full_fit_no_hyper_scaling.chi_squared_map_of_regions_ci
        )

    @property
    def noise_scaling_map_of_parallel_epers(self):
        return (
            self.max_log_likelihood_full_fit_no_hyper_scaling.chi_squared_map_of_parallel_epers
        )

    @property
    def noise_scaling_map_of_serial_epers(self):
        return (
            self.max_log_likelihood_full_fit_no_hyper_scaling.chi_squared_map_of_serial_epers
        )

    @property
    def noise_scaling_map_of_serial_overscan_no_trails(self):
        return (
            self.max_log_likelihood_full_fit_no_hyper_scaling.chi_squared_map_of_serial_overscan_no_trails
        )

    @property
    def noise_scaling_map_list(self):

        return [
            self.noise_scaling_map_of_regions_ci,
            self.noise_scaling_map_of_parallel_epers,
            self.noise_scaling_map_of_serial_epers,
            self.noise_scaling_map_of_serial_overscan_no_trails,
        ]
