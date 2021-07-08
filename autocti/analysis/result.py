from autofit.non_linear import result as res


class Result(res.Result):
    def __init__(self, samples, model, analysis, search):
        """
        The result of a phase
        """
        super().__init__(samples=samples, model=model, search=search)

        self.analysis = analysis
        self.search = search

    @property
    def clocker(self):
        return self.analysis.clocker


class ResultDataset(Result):
    @property
    def max_log_likelihood_fit(self):
        return self.analysis.fit_from_instance(instance=self.instance)

    @property
    def mask(self):
        return self.max_log_likelihood_fit.mask


class ResultDatasetLine(ResultDataset):

    pass


class ResultImagingCI(ResultDataset):
    @property
    def max_log_likelihood_full_fit(self):
        return self.analysis.fit_full_dataset_from_instance(
            instance=self.instance, hyper_noise_scale=True
        )

    @property
    def max_log_likelihood_full_fit_no_hyper_scaling(self):
        return self.analysis.fit_full_dataset_from_instance(
            instance=self.instance, hyper_noise_scale=False
        )

    @property
    def noise_scaling_map_of_regions_ci(self):
        return (
            self.max_log_likelihood_full_fit_no_hyper_scaling.chi_squared_map_of_regions_ci
        )

    @property
    def noise_scaling_map_of_parallel_trails(self):
        return (
            self.max_log_likelihood_full_fit_no_hyper_scaling.chi_squared_map_of_parallel_trails
        )

    @property
    def noise_scaling_map_of_serial_trails(self):
        return (
            self.max_log_likelihood_full_fit_no_hyper_scaling.chi_squared_map_of_serial_trails
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
            self.noise_scaling_map_of_parallel_trails,
            self.noise_scaling_map_of_serial_trails,
            self.noise_scaling_map_of_serial_overscan_no_trails,
        ]
