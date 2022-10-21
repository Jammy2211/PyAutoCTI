from autocti.charge_injection.fit import FitImagingCI
from autocti.model.result import ResultDataset


class ResultImagingCI(ResultDataset):
    @property
    def max_log_likelihood_full_fit(self) -> FitImagingCI:
        return self.analysis.fit_via_instance_and_dataset_from(
            instance=self.instance,
            imaging_ci=self.analysis.dataset.imaging_full,
            hyper_noise_scale=True,
        )

    @property
    def max_log_likelihood_full_fit_no_hyper_scaling(self):
        return self.analysis.fit_via_instance_and_dataset_from(
            instance=self.instance,
            imaging_ci=self.analysis.dataset.imaging_full,
            hyper_noise_scale=False,
        )

    @property
    def noise_scaling_map_dict(self):

        fit = self.max_log_likelihood_full_fit_no_hyper_scaling

        return {
            "regions_ci": fit.chi_squared_map_of_regions_ci,
            "parallel_eper": fit.chi_squared_map_of_parallel_eper,
            "serial_eper": fit.chi_squared_map_of_serial_eper,
            "serial_overscan_no_eper": fit.chi_squared_map_of_serial_overscan_no_eper,
        }
