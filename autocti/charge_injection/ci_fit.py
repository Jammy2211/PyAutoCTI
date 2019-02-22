#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
#
# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3.0 of the License, or (at your option)
# any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
#
import numpy as np
from autofit.tools import fit
from autocti.charge_injection import ci_data

from autocti import exc


def fit_ci_data_fit_with_cti_params_and_settings(ci_data_fit, cti_params, cti_settings):
    """Fit ci data with a model of cti, using the cti params and settings, automatically determining the type of fit
    based on the properties of the data.

    Parameters
    -----------
    ci_data_fit : ci_data.CIDataFit or ci_data.CIDataHyperFit
        The charge injection image that is fitted.
    cti_params : arctic_params.ArcticParams
        The cti model parameters which describe how CTI during clocking.
    cti_settings : arctic_settings.ArcticSettings
        The settings that control how arctic models CTI.
    """

    if ci_data_fit:
        return CIFit(masked_ci_data=ci_data_fit, cti_params=cti_params, cti_settings=cti_settings)
    else:
        raise exc.FittingException('The fit routine did not call a Fit class - check the '
                                   'properties of the tracer')


def hyper_fit_ci_data_fit_with_cti_params_and_settings(ci_data_fit, cti_params, cti_settings, hyper_noise_scalers):
    """Fit ci data with a model of cti, using the cti params and settings, automatically determining the type of fit
    based on the properties of the data.

    Parameters
    -----------
    ci_data_fit : ci_data.CIDataFit or ci_data.CIDataHyperFit
        The charge injection image that is fitted.
    cti_params : arctic_params.ArcticParams
        The cti model parameters which describe how CTI during clocking.
    cti_settings : arctic_settings.ArcticSettings
        The settings that control how arctic models CTI.
    hyper_noise_scalers :
        The ci_hyper-parameter(s) which the noise_scaling_maps is multiplied by to scale the noise-map.
    """

    if ci_data_fit.is_hyper_data:
        return CIHyperFit(ci_data_hyper_fit=ci_data_fit, cti_params=cti_params, cti_settings=cti_settings,
                          hyper_noise_scalers=hyper_noise_scalers)
    else:
        raise exc.FittingException('The fit routine did not call a Fit class - check the '
                                   'properties of the tracer')


class AbstractCIFit(fit.DataFit):

    def __init__(self, masked_ci_data: ci_data.MaskedCIData, noise_map, cti_params, cti_settings):
        """Abstract fit of a charge injection dataset with a model cti image.

        Parameters
        -----------
        masked_ci_data : ci_data.CIDataFit
            The charge injection image that is fitted.
        cti_params : arctic_params.ArcticParams
            The cti model parameters which describe how CTI during clocking.
        cti_settings : arctic_settings.ArcticSettings
            The settings that control how arctic models CTI.
        """
        model_data = masked_ci_data.ci_frame.add_cti(image=masked_ci_data.ci_pre_cti,
                                                     cti_params=cti_params,
                                                     cti_settings=cti_settings)

        super().__init__(data=masked_ci_data.image, noise_map=noise_map,
                         mask=masked_ci_data.mask, model_data=model_data)
        self.ci_data_fit = masked_ci_data
        self.cti_params = cti_params
        self.cti_settings = cti_settings

    @property
    def ci_post_cti(self):
        return self.model_data

    @property
    def ci_pre_cti(self):
        return self.ci_data_fit.ci_pre_cti

    @property
    def image(self):
        return self.data

    @property
    def model_image(self):
        return self.model_data

    @property
    def figure_of_merit(self):
        return self.likelihood


class CIFit(AbstractCIFit):
    def __init__(self, masked_ci_data: ci_data.MaskedCIData, cti_params, cti_settings):
        super().__init__(masked_ci_data, masked_ci_data.noise_map, cti_params, cti_settings)

    @property
    def noise_scaling_map_of_ci_regions(self):
        return self.ci_data_fit.chinj.ci_regions_from_array(array=self.chi_squared_map)

    @property
    def noise_scaling_map_of_parallel_trails(self):
        return self.ci_data_fit.chinj.parallel_non_ci_regions_frame_from_frame(array=self.chi_squared_map)

    @property
    def noise_scaling_map_of_serial_trails(self):
        return self.ci_data_fit.chinj.serial_all_trails_frame_from_frame(array=self.chi_squared_map)

    @property
    def noise_scaling_map_of_serial_overscan_above_trails(self):
        return self.ci_data_fit.chinj.serial_overscan_above_trails_frame_from_frame(array=self.chi_squared_map)


class CIHyperFit(AbstractCIFit):

    def __init__(self, ci_data_hyper_fit, cti_params, cti_settings, hyper_noise_scalers):
        """Fit a charge injection ci_data-set with a model cti image, also scalng the noises within a Bayesian framework.

        Parameters
        -----------
        ci_data_hyper_fit : ci_data.CIDataHyperFit
            The charge injection image that is fitted.
        cti_params : arctic_params.ArcticParams
            The cti model parameters which describe how CTI during clocking.
        cti_settings : arctic_settings.ArcticSettings
            The settings that control how arctic models CTI.
        hyper_noise_scalers :
            The ci_hyper-parameter(s) which the noise_scaling_maps is multiplied by to scale the noise-map.
        """
        self.hyper_noises = hyper_noise_scalers
        self.hyper_noise_map = hyper_noise_map_from_noise_map_and_noise_scalings(
            noise_scaling_maps=ci_data_hyper_fit.noise_scaling_maps, hyper_noise_scalers=hyper_noise_scalers,
            noise_map=ci_data_hyper_fit.noise_map)
        super().__init__(masked_ci_data=ci_data_hyper_fit, noise_map=self.hyper_noise_map, cti_params=cti_params,
                         cti_settings=cti_settings)


def hyper_noise_map_from_noise_map_and_noise_scalings(noise_scaling_maps, hyper_noise_scalers, noise_map):
    """For a noise-map, use the model hyper noise and noise-scaling maps to compute a scaled noise-map.

    Parameters
    -----------
    hyper_noise_scalers
    noise_scaling_maps
    noise_map : imaging.NoiseMap or ndarray
        An array describing the RMS standard deviation error in each pixel, preferably in units of electrons per
        second.
    """
    hyper_noise_maps = list(map(lambda hyper_noise_scaler, noise_scaling_map:
                                hyper_noise_scaler.scaled_noise_map_from_noise_scaling(noise_scaling_map),
                                hyper_noise_scalers, noise_scaling_maps))

    return np.add(noise_map, sum(hyper_noise_maps))
