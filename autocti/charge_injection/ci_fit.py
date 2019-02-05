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


class CIFit(fit.DataFit):

    def __init__(self, ci_data_fit, cti_params, cti_settings):

        self.ci_data_fit = ci_data_fit
        self.cti_params = cti_params
        self.cti_settings = cti_settings

        self.ci_post_cti = self.ci_data_fit.ci_frame.add_cti(image=self.ci_data_fit.ci_pre_cti,
                                                             cti_params=self.cti_params,
                                                             cti_settings=self.cti_settings)

        super(CIFit, self).__init__(data=ci_data_fit.image,
                                    noise_map=ci_data_fit.noise_map,
                                    mask=ci_data_fit.mask,
                                    model_data=self.ci_post_cti)

    @property
    def image(self):
        return self.data

    @property
    def model_images(self):
        return self.model_data

    @property
    def figure_of_merit(self):
        return self.likelihood


class CIHyperFit(CIFit):

    def __init__(self, ci_data_fit, cti_params, cti_settings, hyper_noises):
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

        self.hyper_noise_map = hyper_noise_map_from_noise_map_and_noise_scaling(noise_scaling=ci_data_fit.noise_scaling,
                                                                                hyper_noises=hyper_noises,
                                                                                noise_map=ci_data_fit.noise_map)

        super().__init__(ci_data_fit, cti_params, cti_settings)


def hyper_noise_map_from_noise_map_and_noise_scaling(noise_scaling, hyper_noises, noise_map):
    """For a noise-map, use the model hyper noise and noise-scaling maps to compute a scaled noise-map.

    Parameters
    -----------
    hyper_noises
    noise_scaling
    noise_map : imaging.NoiseMap or ndarray
        An array describing the RMS standard deviation error in each pixel, preferably in units of electrons per
        second.
    """
    scaled_noise_maps = [hyper_noise.scaled_noise_map_from_noise_scaling(noise_scaling) for hyper_noise in hyper_noises]
    return np.add(noise_map, sum(scaled_noise_maps))
