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

    def __init__(self, ci_datas_fit, cti_params, cti_settings):

        self.ci_datas_fit = ci_datas_fit
        self.cti_params = cti_params
        self.cti_settings = cti_settings

        self.ci_post_ctis = list(map(lambda ci_data_fit :
                                            ci_data_fit.ci_pre_cti.ci_post_cti_from_cti_params_and_settings(cti_params=self.cti_params,
                                                                                                            cti_settings=self.cti_settings),
                                     self.ci_datas_fit))


class CIDataFit(fit.DataFitStack):

    def __init__(self, images, noise_maps, masks, ci_post_ctis):

        super(CIDataFit, self).__init__(datas=images, noise_maps=noise_maps, masks=masks, model_datas=ci_post_ctis)

    @property
    def images(self):
        return self.datas

    @property
    def model_images(self):
        return self.model_datas

    @property
    def figure_of_merit(self):
        return self.likelihood

class CIFit(CIDataFit, AbstractCIFit):

    def __init__(self, ci_datas_fit, cti_params, cti_settings):

        AbstractCIFit.__init__(self=self, ci_datas_fit=ci_datas_fit, cti_params=cti_params, cti_settings=cti_settings)

        super(CIFit, self).__init__(images=[ci_data_fit.image for ci_data_fit in self.ci_datas_fit],
                                    noise_maps=[ci_data_fit.noise_map for ci_data_fit in self.ci_datas_fit],
                                    masks=[ci_data_fit.mask for ci_data_fit in self.ci_datas_fit],
                                    ci_post_ctis=[ci_post_cti for ci_post_cti in self.ci_post_ctis])


class CIHyperFit(CIDataFit, AbstractCIFit):

    def __init__(self, ci_datas_fit, cti_params, cti_settings, hyper_noises):
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

        self.hyper_noises = hyper_noises

        self.hyper_noise_maps = hyper_noise_maps_from_ci_datas_fit_and_hyper_noises(ci_datas_fit=ci_datas_fit,
                                                                                    hyper_noises=hyper_noises)

        AbstractCIFit.__init__(self=self, ci_datas_fit=ci_datas_fit, cti_params=cti_params, cti_settings=cti_settings)

        super(CIHyperFit, self).__init__(images=[ci_data_fit.image for ci_data_fit in self.ci_datas_fit],
                                    noise_maps=[hyper_noise_map for hyper_noise_map in self.hyper_noise_maps],
                                    masks=[ci_data_fit.mask for ci_data_fit in self.ci_datas_fit],
                                    ci_post_ctis=[ci_post_cti for ci_post_cti in self.ci_post_ctis])


def hyper_noise_maps_from_ci_datas_fit_and_hyper_noises(ci_datas_fit, hyper_noises):
    """For a list of fitting hyper-images (which includes the image's noise-scaling maps) and model hyper noises,
    compute their scaled noise-maps.

    This is performed by using each hyper-noise's *noise_factor* and *noise_power* parameter in conjunction with the \
    unscaled noise-map and noise-scaling map.

    Parameters
    ----------
    ci_datas_fit : [fitting.fitting_data.FittingHyperImage]
        List of the fitting hyper-images.
    hyper_noises : [galaxy.Galaxy]
        The hyper-noises which represent the model components used to scale the noise, generated from the chi-squared \
        image of a previous phase's fit.
    """
    return list(map(lambda ci_data_fit:
                    hyper_noise_map_from_noise_map_and_noise_scalings(
                        noise_scalings=ci_data_fit.noise_scalings,
                        hyper_noises=hyper_noises,
                        noise_map=ci_data_fit.noise_map),
                    ci_datas_fit))


def hyper_noise_map_from_noise_map_and_noise_scalings(noise_scalings, hyper_noises, noise_map):
    """For a noise-map, use the model hyper noise and noise-scaling maps to compute a scaled noise-map.

    Parameters
    -----------
    contributions_ : ndarray
        The regular's list of 1D masked contribution maps (e.g. one for each hyper-galaxy)
    hyper_galaxies : [galaxy.Galaxy]
        The hyper-galaxies which represent the model components used to scale the noise, which correspond to
        individual galaxies in the regular.
    noise_map : imaging.NoiseMap or ndarray
        An array describing the RMS standard deviation error in each pixel, preferably in units of electrons per
        second.
    """
    scaled_noise_maps = list(map(lambda hyper_noise, noise_scaling:
                                 hyper_noise.scaled_noise_map_from_noise_scaling(noise_scaling),
                                 hyper_noises, noise_scalings))

    return np.add(noise_map, sum(scaled_noise_maps))
