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

from autocti import exc
from autofit.tools import fit

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
    hyper_noise_scalers :
        The ci_hyper-parameter(s) which the noise_scaling_maps is multiplied by to scale the noise-map.
    """

    if not ci_data_fit.is_hyper_data:
        return CIFit(ci_data_fit=ci_data_fit, cti_params=cti_params, cti_settings=cti_settings)
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


class AbstractCIFit(object):

    def __init__(self, ci_data_fit, cti_params, cti_settings):
        """Abstract fit of a charge injection dataset with a model cti image.

        Parameters
        -----------
        ci_data_fit : ci_data.CIDataFit
            The charge injection image that is fitted.
        cti_params : arctic_params.ArcticParams
            The cti model parameters which describe how CTI during clocking.
        cti_settings : arctic_settings.ArcticSettings
            The settings that control how arctic models CTI.
        """
        self.ci_data_fit = ci_data_fit
        self.cti_params = cti_params
        self.cti_settings = cti_settings

        self.ci_post_cti = self.ci_data_fit.ci_frame.add_cti(image=self.ci_data_fit.ci_pre_cti,
                                                             cti_params=self.cti_params,
                                                             cti_settings=self.cti_settings)


class CIDataFit(fit.DataFit):

    def __init__(self, image, noise_map, mask, ci_post_cti):
        """Class to fit charge injection fit data with a model image.

        Parameters
        -----------
        image : ndarray
            The observed image that is fitted.
        noise_map : ndarray
            The noise-map of the observed image.
        mask: msk.Mask
            The mask that is applied to the image.
        model_data : ndarray
            The model image the oberved image is fitted with.
        """
        super(CIDataFit, self).__init__(data=image, noise_map=noise_map, mask=mask, model_data=ci_post_cti)

    @property
    def image(self):
        return self.data

    @property
    def model_image(self):
        return self.model_data

    @property
    def figure_of_merit(self):
        return self.likelihood


class CIFit(CIDataFit, AbstractCIFit):

    def __init__(self, ci_data_fit, cti_params, cti_settings):
        """Fit a charge injection ci_data-set with a model cti image.

        Parameters
        -----------
        ci_data_fit : ci_data.CIDataFit
            The charge injection image that is fitted.
        cti_params : arctic_params.ArcticParams
            The cti model parameters which describe how CTI during clocking.
        cti_settings : arctic_settings.ArcticSettings
            The settings that control how arctic models CTI.
        """
        AbstractCIFit.__init__(self=self, ci_data_fit=ci_data_fit, cti_params=cti_params, cti_settings=cti_settings)

        super(CIFit, self).__init__(image=ci_data_fit.image, noise_map=ci_data_fit.noise_map,
                                    mask=ci_data_fit.mask, ci_post_cti=self.ci_post_cti)

    @property
    def noise_scaling_map_of_ci_regions(self):
        return self.ci_data_fit.ci_frame.ci_regions_from_array(array=self.chi_squared_map)

    @property
    def noise_scaling_map_of_parallel_trails(self):
        return self.ci_data_fit.ci_frame.parallel_non_ci_regions_frame_from_frame(array=self.chi_squared_map)

    @property
    def noise_scaling_map_of_serial_trails(self):
        return self.ci_data_fit.ci_frame.serial_all_trails_frame_from_frame(array=self.chi_squared_map)

    @property
    def noise_scaling_map_of_serial_overscan_above_trails(self):
        return self.ci_data_fit.ci_frame.serial_overscan_above_trails_frame_from_frame(array=self.chi_squared_map)

class CIHyperFit(CIDataFit, AbstractCIFit):

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

        AbstractCIFit.__init__(self=self, ci_data_fit=ci_data_hyper_fit, cti_params=cti_params,
                               cti_settings=cti_settings)

        super(CIHyperFit, self).__init__(image=ci_data_hyper_fit.image, noise_map=self.hyper_noise_map,
                                         mask=ci_data_hyper_fit.mask, ci_post_cti=self.ci_post_cti)


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