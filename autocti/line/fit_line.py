import numpy as np

from autocti.charge_injection import imaging_ci
from autoarray.fit import fit


class FitImagingCI(fit.FitImaging):
    def __init__(
        self, imaging_ci: imaging_ci.ImagingCI, post_cti_ci, hyper_noise_scalars=None
    ):
        """Fit a charge injection ci_data-set with a model cti image, also scalng the noises within a Bayesian
        framework.

        Parameters
        -----------
        imaging_ci
            The charge injection image that is fitted.
        post_cti_ci
            The `pre_cti_ci` with cti added to it via the clocker and a CTI model.
        hyper_noise_scalars :
            The hyper_ci-parameter(s) which the noise_scaling_maps_list is multiplied by to scale the noise-map.
        """

        self.hyper_noise_scalars = hyper_noise_scalars

        if hyper_noise_scalars is not None and len(hyper_noise_scalars) > 0:

            self.noise_scaling_maps = imaging_ci.noise_scaling_maps

            noise_map = hyper_noise_map_from_noise_map_and_noise_scalings(
                hyper_noise_scalars=hyper_noise_scalars,
                noise_scaling_maps=imaging_ci.noise_scaling_maps,
                noise_map=imaging_ci.noise_map,
            )

            imaging_ci = imaging_ci.modify_noise_map(noise_map=noise_map)

        super().__init__(imaging=imaging_ci, model_image=post_cti_ci)

    @property
    def imaging_ci(self):
        return self.dataset

    @property
    def post_cti_ci(self):
        return self.model_data

    @property
    def pre_cti_ci(self):
        return self.dataset.pre_cti_ci

    @property
    def chi_squared_map_of_regions_ci(self):
        return self.chi_squared_map.frame_with_extracted_regions_ci_from

    @property
    def chi_squared_map_of_parallel_trails(self):
        return self.chi_squared_map.parallel_non_frame_with_extracted_regions_ci_from

    @property
    def chi_squared_map_of_serial_trails(self):
        return self.chi_squared_map.serial_trails_frame_from

    @property
    def chi_squared_map_of_serial_overscan_no_trails(self):
        return self.chi_squared_map.serial_overscan_no_trails_frame_from


def hyper_noise_map_from_noise_map_and_noise_scalings(
    hyper_noise_scalars, noise_scaling_maps, noise_map
):
    """For a noise-map, use the model hyper noise and noise-scaling maps to compute a scaled noise-map.

    Parameters
    -----------
    hyper_noise_scalars
    noise_scaling_maps
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
            noise_scaling_maps,
        )
    )

    return np.add(noise_map, sum(hyper_noise_maps))
