import numpy as np

import autoarray as aa

from autocti.charge_injection.imaging.imaging import ImagingCI


class FitImagingCI(aa.FitImaging):
    def __init__(self, dataset: ImagingCI, post_cti_data, hyper_noise_scalars=None):
        """
        Fit a charge injection ci_data-set with a model cti image, also scalng the noises within a Bayesian
        framework.

        Parameters
        -----------
        dataset
            The charge injection image that is fitted.
        post_cti_data
            The `pre_cti_data` with cti added to it via the clocker and a CTI model.
        hyper_noise_scalars :
            The hyper_ci-parameter(s) which the noise_scaling_map_list_list is multiplied by to scale the noise-map.
        """

        super().__init__(dataset=dataset, use_mask_in_fit=True)

        self.post_cti_data = post_cti_data
        self.hyper_noise_scalars = hyper_noise_scalars
        self.noise_scaling_map_list = dataset.noise_scaling_map_list

    @property
    def imaging_ci(self) -> ImagingCI:
        return self.dataset

    @property
    def noise_map(self) -> aa.Array2D:

        if self.hyper_noise_scalars is not None and len(self.hyper_noise_scalars) > 0:

            return hyper_noise_map_from_noise_map_and_noise_scalings(
                hyper_noise_scalars=self.hyper_noise_scalars,
                noise_scaling_map_list=self.dataset.noise_scaling_map_list,
                noise_map=self.dataset.noise_map,
            )

        return self.dataset.noise_map

    @property
    def model_data(self) -> aa.Array2D:
        return self.post_cti_data

    @property
    def layout(self):
        return self.imaging_ci.layout

    @property
    def pre_cti_data(self):
        return self.dataset.pre_cti_data

    @property
    def chi_squared_map_of_regions_ci(self):
        return self.layout.extract.regions_array_2d_from(array=self.chi_squared_map)

    @property
    def chi_squared_map_of_parallel_epers(self):
        return self.layout.extract.parallel_eper.array_2d_from(
            array=self.chi_squared_map
        )

    @property
    def chi_squared_map_of_serial_trails(self):
        return self.layout.extract.serial_eper.array_2d_from(array=self.chi_squared_map)

    @property
    def chi_squared_map_of_serial_overscan_no_trails(self):
        return self.layout.extract.serial_overscan_above_epers_array_2d_from(
            array=self.chi_squared_map
        )


def hyper_noise_map_from_noise_map_and_noise_scalings(
    hyper_noise_scalars, noise_scaling_map_list, noise_map
):
    """For a noise-map, use the model hyper noise and noise-scaling maps to compute a scaled noise-map.

    Parameters
    -----------
    hyper_noise_scalars
    noise_scaling_map_list
    noise_map : imaging.NoiseMap or ndarray
        An arrays describing the RMS standard deviation error in each pixel, preferably in units of electrons per
        second.
    """
    hyper_noise_maps = list(
        map(
            lambda hyper_noise_scalar, noise_scaling_map: hyper_noise_scalar.scaled_noise_map_from_noise_scaling(
                noise_scaling_map
            ),
            hyper_noise_scalars,
            noise_scaling_map_list,
        )
    )

    return np.add(noise_map, sum(hyper_noise_maps))
