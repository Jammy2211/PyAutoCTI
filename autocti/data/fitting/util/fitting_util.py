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


def residuals_from_datas_masks_and_model_datas(datas, masks, model_datas):
    """Compute the residuals between an observed charge injection image and post-cti model image.

    NOTE : If a pixel is masked a 0.0 is returned.

    Residuals = (Data - Model).

    Parameters
    -----------
    datas : [np.ndarray]
        List of the observed data-sets.
    masks : [ChInj.CIMask]
        List of the masks of the observed data-sets.
    model_datas_ : [np.ndarray]
        List of the model data-sets.
    """
    residuals = list(map(lambda data, model_data: np.subtract(data, model_data), datas, model_datas))
    return list(map(lambda residual, mask: residual - residual * mask, residuals, masks))


def chi_squareds_from_residuals_and_noise(residuals, noise_maps):
    """Computes a chi-squared image, by calculating the squared residuals between an observed charge injection \
    images and post-cti model_image image and dividing by the variance (noises**2.0) in each pixel.

    Chi_Sq = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)

    This gives the residuals, which are divided by the variance of each pixel and squared to give their chi sq.

    Parameters
    -----------
    residuals : [np.ndarray]
        List of the residuals of the model-data's fit to the observed data-sets.
    noise_maps : [np.ndarray]
        List of the noise-maps of the observed datas.
    """
    return list(map(lambda residual, noise_map: np.square(np.divide(residual, noise_map)), residuals, noise_maps))


def chi_squared_term_from_chi_squareds(chi_squareds):
    """Compute the chi-squared of a model image's fit to the weighted_data, by taking the difference between the
    observed image and model ray-tracing image, dividing by the noise in each pixel and squaring:

    [Chi_Squared] = sum(([Data - Model] / [Noise]) ** 2.0)

    Parameters
    ----------
    chi_squareds : [np.ndarray]
        List of the chi-squareds values of the model-datas fit to the observed data.
    """
    return list(map(lambda chi_squared: np.sum(chi_squared), chi_squareds))


def noise_term_from_mask_and_noise(masks, noise_maps):
    """Compute the noise normalization term of an image, which is computed by summing the noise in every pixel:

    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    Parameters
    ----------
    masks : [ChInj.CIMask]
        List of the masks of the observed data-sets.
    noise_maps : [np.ndarray]
        List of the noise-maps of the observed datas.
    """
    masked_noise_maps = list(map(lambda noise_map, mask: np.ma.masked_array(noise_map, mask), noise_maps, masks))
    return list(map(lambda masked_noise_map: np.sum(np.log(2 * np.pi * masked_noise_map ** 2.0)), masked_noise_maps))


def likelihood_from_chi_squared_and_noise_terms(chi_squared_terms, noise_terms):
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
    ----------
    chi_squared_terms : [float]
        List of the chi-squared terms for each model-datas fit to the observed data.
    noise_terms : [float]
        List of the normalization noise-terms for each observed data's noise-map.
    """
    return list(map(lambda chi_squared_term, noise_term: -0.5 * (chi_squared_term + noise_term),
                    chi_squared_terms, noise_terms))


def scaled_noise_maps_from_fitting_hyper_images_and_noise_scalings(fitting_hyper_images, hyper_noises):
    """For a list of fitting hyper-images (which includes the image's noise-scaling maps) and model hyper noises,
    compute their scaled noise-maps.

    This is performed by using each hyper-noise's *noise_factor* and *noise_power* parameter in conjunction with the \
    unscaled noise-map and noise-scaling map.

    Parameters
    ----------
    fitting_hyper_images : [fitting.fitting_data.FittingHyperImage]
        List of the fitting hyper-images.
    hyper_noises : [galaxy.Galaxy]
        The hyper-noises which represent the model components used to scale the noise, generated from the chi-squared \
        image of a previous phase's fit.
    """
    return list(map(lambda fitting_hyper_image  :
                    scaled_noise_map_from_noise_map_and_noise_scalings(noise_scalings=fitting_hyper_image.noise_scalings,
                                                                       hyper_noises=hyper_noises,
                                                                       noise_map=fitting_hyper_image.noise_map),
                    fitting_hyper_images))

def scaled_noise_map_from_noise_map_and_noise_scalings(noise_scalings, hyper_noises, noise_map):
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
