from autocti.pipeline.phase import dataset


class Result(dataset.Result):
    @property
    def most_likely_fit(self):

        hyper_image_sky = self.analysis.hyper_image_sky_for_instance(
            instance=self._instance
        )

        hyper_background_noise = self.analysis.hyper_background_noise_for_instance(
            instance=self._instance
        )

        return self.analysis.masked_imaging_fit_for_tracer(
            tracer=self.most_likely_tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

    @property
    def most_likely_extracted_fits(self):
        return self.analysis.fits_of_ci_data_extracted_for_instance(
            instance=self._instance
        )

    @property
    def most_likely_full_fits(self):
        return self.analysis.fits_of_ci_data_full_for_instance(instance=self._instance)

    @property
    def most_likely_full_fits_no_hyper_scaling(self):
        return self.analysis.fits_of_ci_data_full_for_instance(instance=self._instance)

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
