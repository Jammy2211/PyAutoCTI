import numpy as np

from autocti.charge_injection import ci_imaging
from autocti.fit import fit


class CIFitImaging(fit.FitImaging):
    def __init__(
        self,
        masked_ci_imaging: ci_imaging.MaskedCIImaging,
        ci_post_cti,
        hyper_noise_scalars=None,
    ):
        """Fit a charge injection ci_data-set with a model cti image, also scalng the noises within a Bayesian
        framework.

        Parameters
        -----------
        masked_ci_imaging
            The charge injection image that is fitted.
        cti_params : arctic_params.ArcticParams
            The cti model parameters which describe how CTI during clocking.
        clocker : arctic_settings.ArcticSettings
            The settings that control how arctic models CTI.
        hyper_noise_scalars :
            The ci_hyper-parameter(s) which the noise_scaling_maps_list is multiplied by to scale the noise map.
        """

        self.ci_masked_data = masked_ci_imaging
        self.hyper_noise_scalars = hyper_noise_scalars

        if hyper_noise_scalars is not None and len(hyper_noise_scalars) > 0:
            self.noise_scaling_maps = masked_ci_imaging.noise_scaling_maps

            noise_map = hyper_noise_map_from_noise_map_and_noise_scalings(
                hyper_noise_scalars=hyper_noise_scalars,
                noise_scaling_maps=masked_ci_imaging.noise_scaling_maps,
                noise_map=masked_ci_imaging.noise_map,
            )

            masked_ci_imaging = masked_ci_imaging.modify_noise_map(noise_map=noise_map)

        super().__init__(masked_imaging=masked_ci_imaging, model_image=ci_post_cti)

    @property
    def masked_ci_imaging(self):
        return self.ci_masked_data

    @property
    def ci_post_cti(self):
        return self.model_data

    @property
    def ci_pre_cti(self):
        return self.ci_masked_data.ci_pre_cti

    @property
    def chi_squared_map_of_ci_regions(self):
        return self.chi_squared_map.ci_regions_frame

    @property
    def chi_squared_map_of_parallel_trails(self):
        return self.chi_squared_map.parallel_non_ci_regions_frame

    @property
    def chi_squared_map_of_serial_trails(self):
        return self.chi_squared_map.serial_trails_frame

    @property
    def chi_squared_map_of_serial_overscan_no_trails(self):
        return self.chi_squared_map.serial_overscan_no_trails_frame


def hyper_noise_map_from_noise_map_and_noise_scalings(
    hyper_noise_scalars, noise_scaling_maps, noise_map
):
    """For a noise map, use the model hyper noise and noise-scaling maps to compute a scaled noise map.

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
