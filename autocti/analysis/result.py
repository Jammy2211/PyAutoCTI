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
    def max_log_likelihood_fits(self):
        print(self.instance.cti)
        return self.analysis.fits_from_instance(instance=self.instance)

    @property
    def masks(self):
        return [fit.mask for fit in self.max_log_likelihood_fits]


class ResultCIImaging(ResultDataset):
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

    @property
    def noise_scaling_maps_list(self,):

        total_images = len(self.analysis.imaging_cis)

        noise_scaling_maps_list = []

        for image_index in range(total_images):
            noise_scaling_maps_list.append(
                [
                    self.noise_scaling_maps_list_of_ci_regions[image_index],
                    self.noise_scaling_maps_list_of_parallel_trails[image_index],
                    self.noise_scaling_maps_list_of_serial_trails[image_index],
                    self.noise_scaling_maps_list_of_serial_overscan_no_trails[
                        image_index
                    ],
                ]
            )

        for image_index in range(total_images):
            noise_scaling_maps_list[image_index] = [
                noise_scaling_map
                for noise_scaling_map in noise_scaling_maps_list[image_index]
            ]

        return noise_scaling_maps_list
