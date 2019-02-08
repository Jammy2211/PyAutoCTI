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

from autocti import exc
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
        return CIDataHyperFit(image=func(self.image),
                              noise_map=func(self.noise_map),
                              ci_pre_cti=func(self.ci_pre_cti),
                              mask=func(mask),
                              ci_pattern=self.ci_pattern,
                              ci_frame=self.ci_frame,
                              noise_scaling_maps=func(
                                  noise_scaling_maps) if noise_scaling_maps is not None else noise_scaling_maps)

    def parallel_calibration_data(self, columns, mask):
        return self.map_to_ci_data_fit(lambda obj: self.chinj.parallel_calibration_section_for_columns(obj, columns),
                                       mask)

    def serial_calibration_data(self, column, rows, mask):
        return self.map_to_ci_data_fit(
            lambda obj: self.chinj.serial_calibration_section_for_column_and_rows(obj, column=column, rows=rows), mask)

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

    def __init__(self, image, noise_map, ci_pre_cti, mask, ci_pattern, ci_frame):
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
        self.is_hyper_data = False

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

class CIDataHyperFit(CIDataFit):

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
        super().__init__(image=image, noise_map=noise_map, ci_pre_cti=ci_pre_cti, mask=mask, ci_pattern=ci_pattern,
                         ci_frame=ci_frame)

        self.noise_scaling_maps = noise_scaling_maps
        self.is_hyper_data = True


def simulate(shape, frame_geometry, ci_pattern, cti_params, cti_settings, read_noise=None, cosmics=None,
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

    ci_pre_cti = ci_pattern.simulate_ci_pre_cti(frame_geometry=frame_geometry, shape=shape)
    if cosmics is not None:
        ci_pre_cti += cosmics

    ci_pre_cti = cti_image.ImageFrame(frame_geometry=frame_geometry, array=ci_pre_cti)

    ci_post_cti = ci_pre_cti.add_cti_to_image(cti_params, cti_settings)

    if read_noise is not None:
        ci_post_cti += read_noise_map_from_shape_and_sigma(shape=shape, sigma=read_noise, noise_seed=noise_seed)

    return ci_post_cti[:, :]


def ci_pre_cti_from_ci_pattern_geometry_image_and_mask(ci_pattern, frame_geometry, image, mask=None):
    """Setup a pre-cti image from this charge injection ci_data, using the charge injection ci_pattern.

    The pre-cti image is computed depending on whether the charge injection ci_pattern is uniform, non-uniform or \
    'fast' (see ChargeInjectPattern).
    """

    if type(ci_pattern) == pattern.CIPatternUniform:

        ci_pre_cti = ci_pattern.ci_pre_cti_from_shape(shape=image.shape)
        return CIPreCTI(frame_geometry=frame_geometry, array=ci_pre_cti)

    elif type(ci_pattern) == pattern.CIPatternNonUniform:

        ci_pre_cti = ci_pattern.ci_pre_cti_from_ci_image_and_mask(ci_image=image, mask=mask)
        return CIPreCTI(frame_geometry=frame_geometry, array=ci_pre_cti)
    else:
        raise exc.CIPatternException('the CIPattern of the CIImage is not an instance of '
                                     'a known ci_pattern class')


class CIPreCTI(np.ndarray):
    # noinspection PyMissingConstructor,PyUnusedLocal
    def __init__(self, frame_geometry, array):
        self.frame_geometry = frame_geometry

    def __new__(cls, frame_geometry, array, *args, **kwargs):
        return array.view(cls)

    def output_as_fits(self, file_path, overwrite=False):
        """Output the image ci_data as a fits file.

        Params
        ----------
        path : str
            The output nlo path of the ci_data
        filename : str
            The file phase_name of the output image.
        """
        util.numpy_array_to_fits(array=self, file_path=file_path, overwrite=overwrite)

    def add_cti_to_image(self, cti_params, cti_settings):
        self.frame_geometry.add_cti(self, cti_params, cti_settings)


def load_ci_data_list_from_fits(frame_geometries, ci_patterns,
                                ci_image_paths, ci_image_hdus=None,
                                ci_noise_map_paths=None, ci_noise_map_hdus=None,
                                ci_pre_cti_paths=None, ci_pre_cti_hdus=None,
                                masks=None):
    list_size = len(ci_image_paths)

    ci_datas = []

    if ci_pre_cti_paths is None:
        ci_pre_cti_paths = list_size * [None]

    if masks is None:
        masks = list_size * [None]

    if ci_image_hdus is None:
        ci_image_hdus = list_size * [0]

    if ci_noise_map_hdus is None:
        ci_noise_map_hdus = list_size * [0]

    if ci_pre_cti_hdus is None:
        ci_pre_cti_hdus = list_size * [0]

    for data_index in range(list_size):
        ci_data = load_ci_data_from_fits(frame_geometry=frame_geometries[data_index],
                                         ci_pattern=ci_patterns[data_index],
                                         ci_image_path=ci_image_paths[data_index],
                                         ci_image_hdu=ci_image_hdus[data_index],
                                         ci_noise_map_path=ci_noise_map_paths[data_index],
                                         ci_noise_map_hdu=ci_noise_map_hdus[data_index],
                                         ci_pre_cti_path=ci_pre_cti_paths[data_index],
                                         ci_pre_cti_hdu=ci_pre_cti_hdus[data_index],
                                         mask=masks[data_index])

        ci_datas.append(ci_data)

    return ci_datas


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
        ci_pre_cti = ci_pre_cti_from_ci_pattern_geometry_image_and_mask(ci_pattern, frame_geometry, ci_image, mask=mask)

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
    setup_random_seed(noise_seed)
    read_noise_map = np.random.normal(loc=0.0, scale=sigma, size=shape)
    return read_noise_map


def setup_random_seed(seed):
    """Setup the random seed. If the input seed is -1, the code will use a random seed for every run. If it is positive,
    that seed is used for all runs, thereby giving reproducible results

    Params
    ----------
    seed : int
        The seed of the random number generator, used for the random noises maps.
    """
    if seed == -1:
        seed = np.random.randint(0, int(1e9))  # Use one seed, so all regions have identical column non-uniformity.
    np.random.seed(seed)
