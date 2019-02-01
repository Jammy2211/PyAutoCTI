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


class AbstractCIFit(object):

    def __init__(self, ci_data_fit, cti_params, cti_settings):
        self.ci_data_fit = ci_data_fit
        self.cti_params = cti_params
        self.cti_settings = cti_settings

        self.ci_post_cti = ci_data_fit.ci_pre_cti.ci_post_cti_from_cti_params_and_settings(
            cti_params=self.cti_params,
            cti_settings=self.cti_settings)


class CIDataFit(fit.DataFit):

    def __init__(self, image, noise_map, mask, ci_post_cti):
        super(CIDataFit, self).__init__(data=image, noise_map=noise_map, mask=mask, model_data=ci_post_cti)

    @property
    def image(self):
        return self.data

    @property
    def model_images(self):
        return self.model_data

    @property
    def figure_of_merit(self):
        return self.likelihood


class CIFit(CIDataFit, AbstractCIFit):

    def __init__(self, ci_data_fit, cti_params, cti_settings):
        AbstractCIFit.__init__(self=self, ci_data_fit=ci_data_fit, cti_params=cti_params, cti_settings=cti_settings)

        super(CIFit, self).__init__(image=ci_data_fit.image,
                                    noise_map=ci_data_fit.noise_map,
                                    mask=ci_data_fit.mask,
                                    ci_post_cti=self.ci_post_cti)


class CIHyperFit(CIDataFit, AbstractCIFit):

    def __init__(self, ci_data_fit, cti_params, cti_settings, hyper_noise):
        """Fit a charge injection ci_data-set with a model cti image, also scalng the noises within a Bayesian framework.
        Params
        -----------
        ci_data : CIImage.CIImage
            The charge injection ci_data (image, mask, noises)
        ci_post_cti : CIImage.CIPreCTI
            The post-cti model image of the charge injection ci_data.
        hyper_noises :
            The ci_hyper-parameter(s) which the noise_scalings is multiplied by to scale the noises.
        """

        self.hyper_noises = hyper_noise

        self.hyper_noise_map = hyper_noise_map_from_ci_data_fit_and_hyper_noise(ci_data_fit=ci_data_fit,
                                                                                hyper_noise=hyper_noise)

        AbstractCIFit.__init__(self=self, ci_data_fit=ci_data_fit, cti_params=cti_params, cti_settings=cti_settings)

        super(CIHyperFit, self).__init__(image=ci_data_fit.image,
                                         noise_map=self.hyper_noise_map,
                                         mask=ci_data_fit.mask,
                                         ci_post_cti=self.ci_post_cti)


def hyper_noise_map_from_ci_data_fit_and_hyper_noise(ci_data_fit, hyper_noise):
    """For a list of fitting hyper-images (which includes the image's noise-scaling maps) and model hyper noises,
    compute their scaled noise-maps.

    This is performed by using each hyper-noise's *noise_factor* and *noise_power* parameter in conjunction with the \
    unscaled noise-map and noise-scaling map.

    Parameters
    ----------
    ci_data_fit : fitting.fitting_data.FittingHyperImage
       The fitting hyper-image.
    hyper_noise : galaxy.Galaxy
        The hyper-noises which represent the model components used to scale the noise, generated from the chi-squared \
        image of a previous phase's fit.
    """
    return hyper_noise_map_from_noise_map_and_noise_scaling(
        noise_scaling=ci_data_fit.noise_scaling,
        hyper_noise=hyper_noise,
        noise_map=ci_data_fit.noise_map)


def hyper_noise_map_from_noise_map_and_noise_scaling(noise_scaling, hyper_noise, noise_map):
    """For a noise-map, use the model hyper noise and noise-scaling maps to compute a scaled noise-map.

    Parameters
    -----------
    hyper_noise
    noise_scaling
    noise_map : imaging.NoiseMap or ndarray
        An array describing the RMS standard deviation error in each pixel, preferably in units of electrons per
        second.
    """
    scaled_noise_map = hyper_noise.scaled_noise_map_from_noise_scaling(noise_scaling)
    return np.add(noise_map, scaled_noise_map)
