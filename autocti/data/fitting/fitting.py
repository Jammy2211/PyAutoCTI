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

"""
File: python/VIS_CTI_ChargeInjection/CIImage.pyy

Created on: 02/14/18
Author: James Nightingale
"""

import numpy as np

from autocti.data.charge_injection import ci_frame
from autocti.data.fitting.util import fitting_util


class CIFitter(object):

    def __init__(self, ci_datas, cti_params, cti_settings):
        self.ci_datas = ci_datas
        self.cti_params = cti_params
        self.cti_settings = cti_settings

    def as_ci_frames(self, arrays):
        return list(map(lambda array, ci_data:
                        ci_frame.CIFrame(frame_geometry=ci_data.image.frame_geometry,
                                         ci_pattern=ci_data.image.ci_pattern,
                                         array=array),
                        arrays, self.ci_datas))

    @property
    def ci_post_ctis(self):
        ci_post_ctis = list(map(lambda ci_data:
                                ci_data.ci_pre_cti.create_ci_post_cti(cti_params=self.cti_params,
                                                                      cti_settings=self.cti_settings), self.ci_datas))
        return self.as_ci_frames(ci_post_ctis)

    @property
    def noise_term(self):
        return np.sum(list(map(lambda ci_data: fitting_util.noise_term_from_mask_and_noise(ci_data.mask, ci_data.noise),
                               self.ci_datas)))

    @property
    def residuals(self):
        """Fit a charge injection ci_data-set with a model cti image.

        Params
        -----------
        ci_image : ChInj.CIImage
            The charge injection ci_data (image, mask, noises)
        ci_post_cti : ChInj.CIPreCTI
            The post-cti model image of the charge injection ci_data.
        """
        residuals = list(map(lambda ci_data, ci_post_cti:
                             fitting_util.residuals_from_image_mask_and_model(ci_data.image, ci_data.mask, ci_post_cti),
                             self.ci_datas, self.ci_post_ctis))

        return self.as_ci_frames(residuals)

    @property
    def chi_squareds(self):
        """Fit a charge injection ci_data-set with a model cti image.

        Params
        -----------
        ci_image : ChInj.CIImage
            The charge injection ci_data (image, mask, noises)
        ci_post_cti : ChInj.CIPreCTI
            The post-cti model image of the charge injection ci_data.
        """
        chi_squareds = list(map(lambda ci_data, residuals:
                                fitting_util.chi_squareds_from_residuals_and_noise(residuals, ci_data.noise),
                                self.ci_datas, self.residuals))

        return self.as_ci_frames(chi_squareds)

    @property
    def chi_squared_term(self):
        return fitting_util.chi_squared_term_from_chi_squareds(self.chi_squareds)

    @property
    def likelihood(self):
        """Fit a charge injection ci_data-set with a model cti image.

        Params
        -----------
        ci_image : ChInj.CIImage
            The charge injection ci_data (image, mask, noises)
        ci_post_cti : ChInj.CIPreCTI
            The post-cti model image of the charge injection ci_data.
        """
        return fitting_util.likelihood_from_chi_squared_and_noise_terms(self.chi_squared_term, self.noise_term)


class HyperCIFitter(CIFitter):

    def __init__(self, ci_datas, cti_params, cti_settings, hyper_noises):
        """Fit a charge injection ci_data-set with a model cti image, also scalng the noises within a Bayesian framework.
        Params
        -----------
        ci_data : CIImage.CIImage
            The charge injection ci_data (image, mask, noises)
        ci_post_cti : CIImage.CIPreCTI
            The post-cti model image of the charge injection ci_data.
        noise_scalings : CIHyper.CINoiseScaling
            The images used to scale the noises in certain regions of the image.
        noise_scalings:
            The ci_hyper-parameter(s) which the noise_scalings is multiplied by to scale the noises.
        """
        super(HyperCIFitter, self).__init__(ci_datas, cti_params, cti_settings)
        self.hyper_noises = hyper_noises

    @property
    def scaled_noises(self):
        scaled_noises = list(map(lambda ci_data:
                                 scaled_noise_from_noise_and_noise_scalings(ci_data.noise, ci_data.noise_scalings,
                                                                            self.hyper_noises), self.ci_datas))
        return self.as_ci_frames(scaled_noises)

    @property
    def scaled_noise_term(self):
        return np.sum(list(map(lambda ci_data, scaled_noise: noise_term_from_mask_and_noise(ci_data.mask, scaled_noise),
                               self.ci_datas, self.scaled_noises)))

    @property
    def scaled_chi_squareds(self):
        scaled_chi_squareds = list(map(lambda residuals, scaled_noise: chi_squareds_from_residuals_and_noise(residuals,
                                                                                                             scaled_noise),
                                       self.residuals, self.scaled_noises))

        return self.as_ci_frames(scaled_chi_squareds)

    @property
    def scaled_chi_squared_term(self):
        return chi_squared_term_from_chi_squareds(self.scaled_chi_squareds)

    @property
    def scaled_likelihood(self):
        return likelihood_from_chi_squared_and_noise_terms(self.scaled_chi_squared_term, self.scaled_noise_term)
