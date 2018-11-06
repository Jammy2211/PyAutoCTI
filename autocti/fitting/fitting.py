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

from __future__ import division, print_function
import sys

from autocti.charge_injection import ci_frame

import numpy as np

if sys.version_info[0] < 3:
    from future_builtins import *

class CIFitter(object):

    def __init__(self, ci_datas, cti_params, cti_settings):

        self.ci_datas = ci_datas
        self.cti_params = cti_params
        self.cti_settings = cti_settings

    def as_ci_frames(self, arrays):
        return list(map(lambda  array, ci_data :
                        ci_frame.CIFrame(frame_geometry=ci_data.image.frame_geometry, ci_pattern=ci_data.image.ci_pattern,
                                         array=array),
                        arrays, self.ci_datas))

    @property
    def ci_post_ctis(self):
        ci_post_ctis = list(map(lambda ci_data :
                                ci_data.ci_pre_cti.create_ci_post_cti(cti_params=self.cti_params,
                                                                      cti_settings=self.cti_settings), self.ci_datas))
        return self.as_ci_frames(ci_post_ctis)

    @property
    def noise_term(self):
        return np.sum(list(map(lambda ci_data : noise_term_from_mask_and_noise(ci_data.mask, ci_data.noise),
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
                         residuals_from_image_mask_and_model(ci_data.image, ci_data.mask, ci_post_cti),
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
        chi_squareds = list(map(lambda ci_data, residuals :
                        chi_squareds_from_residuals_and_noise(residuals, ci_data.noise), self.ci_datas, self.residuals))

        return self.as_ci_frames(chi_squareds)

    @property
    def chi_squared_term(self):
        return chi_squared_term_from_chi_squareds(self.chi_squareds)

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
        return likelihood_from_chi_squared_and_noise_terms(self.chi_squared_term, self.noise_term)


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
        hyper_noises:
            The ci_hyper-parameter(s) which the noise_scalings is multiplied by to scale the noises.
        """
        super(HyperCIFitter, self).__init__(ci_datas, cti_params, cti_settings)
        self.hyper_noises = hyper_noises

    @property
    def scaled_noises(self):
        scaled_noises = list(map(lambda ci_data :
                        scaled_noise_from_noise_and_noise_scalings(ci_data.noise, ci_data.noise_scalings,
                                                                   self.hyper_noises), self.ci_datas))
        return self.as_ci_frames(scaled_noises)

    @property
    def scaled_noise_term(self):
        return np.sum(list(map(lambda ci_data, scaled_noise : noise_term_from_mask_and_noise(ci_data.mask, scaled_noise),
                               self.ci_datas, self.scaled_noises)))

    @property
    def scaled_chi_squareds(self):

        scaled_chi_squareds = list(map(lambda residuals, scaled_noise : chi_squareds_from_residuals_and_noise(residuals,
                                   scaled_noise), self.residuals, self.scaled_noises))

        return self.as_ci_frames(scaled_chi_squareds)

    @property
    def scaled_chi_squared_term(self):
        return chi_squared_term_from_chi_squareds(self.scaled_chi_squareds)

    @property
    def scaled_likelihood(self):
        return likelihood_from_chi_squared_and_noise_terms(self.scaled_chi_squared_term, self.scaled_noise_term)


def residuals_from_image_mask_and_model(image, mask, model):
    """Compute the residuals between an observed charge injection image and post-cti model image.

    NOTE : If a pixel is masked a 0.0 is returned.

    Residuals = (Data - Model).

    Parameters
    -----------
    image : ChInj.CIImage
        The observed charge injection image ci_data.
    mask : ChInj.CIMask
        The mask of the charge injection image ci_data.
    model : np.ndarray
        The model image.
    """
    residuals = image - model
    return residuals - residuals*mask

def chi_squareds_from_residuals_and_noise(residuals, noise):
    """Computes a chi-squared image, by calculating the squared residuals between an observed charge injection \
    images and post-cti model_image image and dividing by the variance (noises**2.0) in each pixel.

    Chi_Sq = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)

    This gives the residuals, which are divided by the variance of each pixel and squared to give their chi sq.

    Parameters
    -----------
    image : ChInj.CIImage
        The observed charge injection image ci_data (includes the mask).
    mask : ChInj.CIMask
        The mask of the charge injection image ci_data.
    noise : np.ndarray
        The noises in the image.
    model : np.ndarray
        The model_image image.
    """
    return (residuals / noise) ** 2.0

def chi_squared_term_from_chi_squareds(chi_squareds):
    """Compute the chi-squared of a model image's fit to the weighted_data, by taking the difference between the
    observed image and model ray-tracing image, dividing by the noise in each pixel and squaring:

    [Chi_Squared] = sum(([Data - Model] / [Noise]) ** 2.0)

    Parameters
    ----------
    image : grids.GridData
        The image weighted_data.
    noise : grids.GridData
        The noise in each pixel.
    model : grids.GridData
        The model image of the weighted_data.
    """
    return np.sum(chi_squareds)

def noise_term_from_mask_and_noise(mask, noise):
    """Compute the noise normalization term of an image, which is computed by summing the noise in every pixel:

    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    Parameters
    ----------
    noise : grids.GridData
        The noise in each pixel.
    """
    masked_noise = np.ma.masked_array(noise, mask)
    return np.sum(np.log(2 * np.pi * masked_noise ** 2.0))

def likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term):
    """Computes the likelihood of a charge injection line image, by taking the difference between an observed \
     charge injection line image and model_image post-cti image. The likelihood consists of two terms:

    Chi-squared term - The residuals (model_image - ci_data) of every pixel divided by the noises in each pixel, all squared.
    [Chi_Squared_Term] = sum(([Residuals] / [Noise]) ** 2.0)

    The overall normalization of the noises is also included, by summing the log noises value in each pixel:
    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    These are summed and multiplied by -0.5 to give the likelihood:

    Likelihood = -0.5*[Chi_Squared_Term + Noise_Term]

    A mask may also be included, and pixels not included in the mask are omitted from both the chi-squared and \
    noises terms.

    Parameters
    -----------
    image : ChInj.CIImage
        The observed charge injection image ci_data (includes the mask).
    mask : ChInj.CIMask
        The mask of the charge injection image ci_data.
    noise : np.ndarray
        The noises in the image.
    model : np.ndarray
        The model_image image.

    Returns
    ----------
    likelihood : float
        The likelihood computed from comparing this *CIImage* instance's observed and model_mapper images.
    """
    return -0.5 * (chi_squared_term + noise_term)

def scaled_noise_from_noise_and_noise_scalings(noise, noise_scalings, hyper_ci_noises):

    scaled_noises = list(map(lambda hyper, noise_scaling :
                                   hyper.compute_scaled_noise(noise_scaling),
                                   hyper_ci_noises, noise_scalings))

    return noise + sum(scaled_noises)