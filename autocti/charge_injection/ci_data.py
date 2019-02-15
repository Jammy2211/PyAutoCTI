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

from autocti.charge_injection import ci_frame as frame
from autocti.charge_injection import ci_pattern as pattern
from autocti.data import cti_image
from autocti.data import mask as msk
from autocti.data import util


class CIData(object):

    def __init__(self, image, noise_map, ci_pre_cti, ci_pattern, ci_frame):

        self.image = image
        self.noise_map = noise_map
        self.ci_pre_cti = ci_pre_cti
        self.ci_pattern = ci_pattern
        self.ci_frame = ci_frame
        self.is_hyper_data = False

    @property
    def chinj(self):
        return frame.ChInj(frame_geometry=self.ci_frame, ci_pattern=self.ci_pattern)

    @property
    def shape(self):
        return self.image.shape

    def map_to_ci_data_fit(self, func, mask):
        return CIDataFit(image=func(self.image),
                         noise_map=func(self.noise_map),
                         ci_pre_cti=func(self.ci_pre_cti),
                         mask=func(mask),
                         ci_pattern=self.ci_pattern,
                         ci_frame=self.ci_frame)

    def map_to_ci_data_hyper_fit(self, func, mask, noise_scaling_maps=None):
        return CIDataFit(image=func(self.image),
                         noise_map=func(self.noise_map),
                         ci_pre_cti=func(self.ci_pre_cti),
                         mask=func(mask),
                         ci_pattern=self.ci_pattern,
                         ci_frame=self.ci_frame,
                         noise_scaling_maps=func(
                                  noise_scaling_maps) if noise_scaling_maps is not None else noise_scaling_maps)

    def parallel_calibration_data(self, columns, mask):
        return self.map_to_ci_data_fit(lambda obj: self.chinj.parallel_calibration_section_for_columns(array=obj,
                                                                                                       columns=columns),
                                       mask)

    def serial_calibration_data(self, rows, mask):
        return self.map_to_ci_data_fit(
            lambda obj: self.chinj.serial_calibration_section_for_rows(array=obj, rows=rows), mask)

    def parallel_serial_calibration_data(self, mask):
        return self.map_to_ci_data_fit(lambda obj: self.chinj.parallel_serial_calibration_section(obj, ), mask)

    @property
    def signal_to_noise_map(self):
        """The estimated signal-to-noise_maps mappers of the image."""
        signal_to_noise_map = np.divide(self.image, self.noise_map)
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return signal_to_noise_map

    @property
    def signal_to_noise_max(self):
        """The maximum value of signal-to-noise_maps in an image pixel in the image's signal-to-noise_maps mappers"""
        return np.max(self.signal_to_noise_map)


class CIDataFit(object):

    def __init__(self, image, noise_map, ci_pre_cti, mask, ci_pattern, ci_frame, noise_scaling_maps=None):
        """A fitting image is the collection of data components (e.g. the image, noise-maps, PSF, etc.) which are used \
        to generate and fit it with a model image.

        The fitting image is in 2D and masked, primarily to removoe cosmic rays.

        The fitting image also includes a number of attributes which are used to performt the fit, including (y,x) \
        grids of coordinates, convolvers and other utilities.

        Parameters
        ----------
        image : im.Image
            The 2D observed image and other observed quantities (noise-map, PSF, exposure-time map, etc.)
        mask: msk.Mask | None
            The 2D mask that is applied to image data.

        Attributes
        ----------
        image : ScaledSquarePixelArray
            The 2D observed image data (not an instance of im.Image, so does not include the other data attributes,
            which are explicitly made as new attributes of the fitting image).
        noise_map : NoiseMap
            An array describing the RMS standard deviation error in each pixel, preferably in units of electrons per
            second.
        mask: msk.Mask
            The 2D mask that is applied to image data.
        """
        self.image = image
        self.noise_map = noise_map
        self.ci_pre_cti = ci_pre_cti
        self.mask = mask
        self.ci_pattern = ci_pattern
        self.ci_frame = ci_frame
        self.noise_scaling_maps = noise_scaling_maps

    @property
    def is_hyper_data(self):
        if hasattr(self, 'noise_scaling_maps'):
            return True
        else:
            return False

    @property
    def chinj(self):
        return frame.ChInj(frame_geometry=self.ci_frame, ci_pattern=self.ci_pattern)

    @property
    def signal_to_noise_map(self):
        """The estimated signal-to-noise_maps mappers of the image."""
        signal_to_noise_map = np.divide(self.image, self.noise_map)
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return signal_to_noise_map

    @property
    def signal_to_noise_max(self):
        """The maximum value of signal-to-noise_maps in an image pixel in the image's signal-to-noise_maps mappers"""
        return np.max(self.signal_to_noise_map)


def simulate(ci_pre_cti, frame_geometry, ci_pattern, cti_params, cti_settings, read_noise=None, cosmics=None,
             noise_seed=-1):
    """Simulate a charge injection image, including effects like noises.

    Parameters
    -----------
    cosmics
    shape : (int, int)
        The dimensions of the output simulated charge injection image.
    frame_geometry : ci_frame.CIQuadGeometry
        The quadrant geometry of the simulated image, defining where the parallel / serial overscans are and \
        therefore the direction of clocking and rotations before input into the cti algorithm.
    ci_pattern : ci_pattern.CIPatternSimulate
        The charge injection ci_pattern (regions, normalization, etc.) of the charge injection image.
    cti_params : ArcticParams.ArcticParams
        The CTI model parameters (trap density, trap lifetimes etc.).
    cti_settings : ArcticSettings.ArcticSettings
        The settings that control the cti clocking algorithm (e.g. ccd well_depth express option).
    read_noise : None or float
        The FWHM of the Gaussian read-noises added to the image.
    noise_seed : int
        Seed for the read-noises added to the image.
    """

    ci_frame = frame.ChInj(frame_geometry=frame_geometry, ci_pattern=ci_pattern)

    if cosmics is not None:
        ci_pre_cti += cosmics

    ci_pre_cti = cti_image.ImageFrame(frame_geometry=frame_geometry, array=ci_pre_cti)

    ci_post_cti = ci_pre_cti.add_cti_to_image(cti_params, cti_settings)

    if read_noise is not None:
        ci_image = ci_post_cti + read_noise_map_from_shape_and_sigma(shape=ci_post_cti.shape, sigma=read_noise,
                                                                     noise_seed=noise_seed)
        ci_noise_map = read_noise*np.ones(ci_post_cti.shape)
    else:
        ci_image = ci_post_cti
        ci_noise_map = None

    return CIData(ci_frame=ci_frame, image=ci_image, noise_map=ci_noise_map, ci_pre_cti=ci_pre_cti, ci_pattern=ci_pattern)


def ci_pre_cti_from_ci_pattern_geometry_image_and_mask(ci_pattern, image, mask=None):
    """Setup a pre-cti image from this charge injection ci_data, using the charge injection ci_pattern.

    The pre-cti image is computed depending on whether the charge injection ci_pattern is uniform, non-uniform or \
    'fast' (see ChargeInjectPattern).
    """
    if isinstance(ci_pattern, pattern.CIPatternUniform):
        return ci_pattern.ci_pre_cti_from_shape(image.shape)
    return ci_pattern.ci_pre_cti_from_ci_image_and_mask(image, mask)


def load_ci_data_from_fits(frame_geometry, ci_pattern,
                           ci_image_path, ci_image_hdu=0,
                           ci_noise_map_path=None, ci_noise_map_hdu=0,
                           ci_noise_map_from_single_value=None,
                           ci_pre_cti_path=None, ci_pre_cti_hdu=0,
                           mask=None):
    ci_image = util.numpy_array_from_fits(file_path=ci_image_path, hdu=ci_image_hdu)

    if ci_noise_map_path is not None:
        ci_noise_map = util.numpy_array_from_fits(file_path=ci_noise_map_path, hdu=ci_noise_map_hdu)
    else:
        ci_noise_map = np.ones(ci_image.shape) * ci_noise_map_from_single_value

    if ci_pre_cti_path is not None:
        ci_pre_cti = util.numpy_array_from_fits(file_path=ci_pre_cti_path, hdu=ci_pre_cti_hdu)
    else:
        ci_pre_cti = ci_pre_cti_from_ci_pattern_geometry_image_and_mask(ci_pattern, ci_image, mask=mask)

    return CIData(image=ci_image, noise_map=ci_noise_map, ci_pre_cti=ci_pre_cti, ci_pattern=ci_pattern,
                  ci_frame=frame_geometry)


def read_noise_map_from_shape_and_sigma(shape, sigma, noise_seed=-1):
    """Generate a two-dimensional read noises-map, generating values from a Gaussian distribution with mean 0.0.

    Params
    ----------
    shape : (int, int)
        The (x,y) image_shape of the generated Gaussian noises map.
    read_noise : float
        Standard deviation of the 1D Gaussian that each noises value is drawn from
    seed : int
        The seed of the random number generator, used for the random noises maps.
    """
    if noise_seed == -1:
        noise_seed = np.random.randint(0,
                                       int(1e9))  # Use one seed, so all regions have identical column non-uniformity.
    np.random.seed(noise_seed)
    read_noise_map = np.random.normal(loc=0.0, scale=sigma, size=shape)
    return read_noise_map
