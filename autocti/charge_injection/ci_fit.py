import numpy as np
import autofit as af
from autocti.charge_injection import ci_data


class CIDataFit(af.DataFit):
    def __init__(self, ci_data_masked: ci_data.CIDataMasked, noise_map, model_data):
        """Abstract fit of a charge injection dataset with a model cti image.

        Parameters
        -----------
        ci_data_masked : ci_data.CIDataMasked
            The charge injection image that is fitted.
        cti_params : arctic_params.ArcticParams
            The cti model parameters which describe how CTI during clocking.
        cti_settings : arctic_settings.ArcticSettings
            The settings that control how arctic models CTI.
        """

        super(CIDataFit, self).__init__(
            data=ci_data_masked.image,
            noise_map=noise_map,
            mask=ci_data_masked.mask,
            model_data=model_data,
        )

        self.ci_data_masked = ci_data_masked

    @property
    def ci_post_cti(self):
        return self.model_data

    @property
    def ci_pre_cti(self):
        return self.ci_data_masked.ci_pre_cti

    @property
    def image(self):
        return self.data

    @property
    def model_image(self):
        return self.model_data

    @property
    def figure_of_merit(self):
        return self.likelihood

    @property
    def chi_squared_map_of_ci_regions(self):
        return self.ci_data_masked.chinj.ci_regions_from_array(
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


class CIFit(CIDataFit):
    def __init__(
        self,
        ci_data_masked: ci_data.CIDataMasked,
        cti_params,
        cti_settings,
        hyper_noise_scalars=None,
    ):
        """Fit a charge injection ci_data-set with a model cti image, also scalng the noises within a Bayesian
        framework.

        Parameters
        -----------
        ci_data_masked
            The charge injection image that is fitted.
        cti_params : arctic_params.ArcticParams
            The cti model parameters which describe how CTI during clocking.
        cti_settings : arctic_settings.ArcticSettings
            The settings that control how arctic models CTI.
        hyper_noise_scalars :
            The ci_hyper-parameter(s) which the noise_scaling_maps is multiplied by to scale the noise-map.
        """

        self.cti_params = cti_params
        self.cti_settings = cti_settings

        model_data = ci_data_masked.ci_frame.frame_geometry.add_cti(
            image=ci_data_masked.ci_pre_cti,
            cti_params=cti_params,
            cti_settings=cti_settings,
        )

        self.hyper_noise_scalars = hyper_noise_scalars

        if hyper_noise_scalars is not None and hyper_noise_scalars is not []:

            self.noise_scaling_maps = ci_data_masked.noise_scaling_maps
            noise_map = hyper_noise_map_from_noise_map_and_noise_scalings(
                hyper_noise_scalars=hyper_noise_scalars,
                noise_scaling_maps=ci_data_masked.noise_scaling_maps,
                noise_map=ci_data_masked.noise_map,
            )

        else:

            noise_map = ci_data_masked.noise_map

        super().__init__(
            ci_data_masked=ci_data_masked, noise_map=noise_map, model_data=model_data
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
        An array describing the RMS standard deviation error in each pixel, preferably in units of electrons per
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
