import numpy as np
from autoarray.fit import fit as aa_fit
from autocti.charge_injection import ci_data


class CIImagingFit(aa_fit.ImagingFit):
    def __init__(
        self,
        ci_masked_imaging: ci_data.CIMaskedImaging,
        cti_params,
        cti_settings,
        hyper_noise_scalars=None,
    ):
        """Fit a charge injection ci_data-set with a model cti image, also scalng the noises within a Bayesian
        framework.

        Parameters
        -----------
        ci_masked_imaging
            The charge injection image that is fitted.
        cti_params : arctic_params.ArcticParams
            The cti model parameters which describe how CTI during clocking.
        cti_settings : arctic_settings.ArcticSettings
            The settings that control how arctic models CTI.
        hyper_noise_scalars :
            The ci_hyper-parameter(s) which the noise_scaling_maps is multiplied by to scale the noise-map.
        """

        self.ci_masked_data = ci_masked_imaging
        self.cti_params = cti_params
        self.cti_settings = cti_settings

        model_image = ci_masked_imaging.ci_frame.frame_geometry.add_cti(
            image=ci_masked_imaging.ci_pre_cti,
            cti_params=cti_params,
            cti_settings=cti_settings,
        )

        self.hyper_noise_scalars = hyper_noise_scalars

        if hyper_noise_scalars is not None and hyper_noise_scalars is not []:

            self.noise_scaling_maps = ci_masked_imaging.noise_scaling_maps
            noise_map = hyper_noise_map_from_noise_map_and_noise_scalings(
                hyper_noise_scalars=hyper_noise_scalars,
                noise_scaling_maps=ci_masked_imaging.noise_scaling_maps,
                noise_map=ci_masked_imaging.noise_map,
            )

        else:

            noise_map = ci_masked_imaging.noise_map

        super().__init__(
            mask=ci_masked_imaging.mask,
            image=ci_masked_imaging.image,
            noise_map=noise_map,
            model_image=model_image,
        )

    @property
    def ci_masked_imaging(self):
        return self.ci_masked_data

    @property
    def ci_post_cti(self):
        return self.model_data

    @property
    def ci_pre_cti(self):
        return self.ci_data_masked.ci_pre_cti

    @property
    def chi_squared_map_of_ci_regions(self):
        return self.ci_data_masked.chinj.ci_regions_array(
            array=self.chi_squared_map.copy()
        )

    @property
    def chi_squared_map_of_parallel_trails(self):
        return self.ci_data_masked.chinj.parallel_non_ci_regions_frame_from_frame(
            array=self.chi_squared_map.copy()
        )

    @property
    def chi_squared_map_of_serial_trails(self):
        return self.ci_data_masked.chinj.serial_all_trails_frame_from_frame(
            array=self.chi_squared_map.copy()
        )

    @property
    def chi_squared_map_of_serial_overscan_above_trails(self):
        return self.ci_data_masked.chinj.serial_overscan_above_trails_frame_from_frame(
            array=self.chi_squared_map.copy()
        )


def hyper_noise_map_from_noise_map_and_noise_scalings(
    hyper_noise_scalars, noise_scaling_maps, noise_map
):
    """For a noise-map, use the model hyper noise and noise-scaling maps to compute a scaled noise-map.

    Parameters
    -----------
    hyper_noise_scalars
    noise_scaling_maps
    noise_map : imaging.NoiseMap or ndarray
        An arrays describing the RMS standard deviation error in each pixel, preferably in unit_label of electrons per
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
