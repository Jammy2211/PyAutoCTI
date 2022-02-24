import numpy as np

from autoarray.fit.fit_data import FitData
from autoarray.fit.fit_dataset import FitImaging

from autocti.charge_injection.imaging import ImagingCI


class FitImagingCI(FitImaging):
    def __init__(self, dataset: ImagingCI, post_cti_data, hyper_noise_scalars=None):
        """Fit a charge injection ci_data-set with a model cti image, also scalng the noises within a Bayesian
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

        self.hyper_noise_scalars = hyper_noise_scalars

        if hyper_noise_scalars is not None and len(hyper_noise_scalars) > 0:

            self.noise_scaling_map_list = dataset.noise_scaling_map_list

            noise_map = hyper_noise_map_from_noise_map_and_noise_scalings(
                hyper_noise_scalars=hyper_noise_scalars,
                noise_scaling_map_list=dataset.noise_scaling_map_list,
                noise_map=dataset.noise_map,
            )

        else:

            noise_map = dataset.noise_map

        fit = FitData(
            data=dataset.data,
            noise_map=noise_map,
            model_data=post_cti_data,
            mask=dataset.mask,
            use_mask_in_fit=True,
        )

        super().__init__(dataset=dataset, fit=fit)

    @property
    def imaging_ci(self):
        return self.dataset

    @property
    def layout(self):
        return self.imaging_ci.layout

    @property
    def post_cti_data(self):
        return self.model_data

    @property
    def pre_cti_data(self):
        return self.dataset.pre_cti_data

    @property
    def chi_squared_map_of_regions_ci(self):
        return self.layout.extract_misc.regions_array_2d_from(
            array=self.chi_squared_map
        )

    @property
    def chi_squared_map_of_parallel_epers(self):
        return self.layout.parallel_epers_array_2d_from(array=self.chi_squared_map)

    @property
    def chi_squared_map_of_serial_trails(self):
        return self.layout.serial_epers_array_2d_from(array=self.chi_squared_map)

    @property
    def chi_squared_map_of_serial_overscan_no_trails(self):
        return self.layout.serial_overscan_above_epers_array_2d_from(
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
