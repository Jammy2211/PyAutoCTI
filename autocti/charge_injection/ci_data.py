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
from autocti.model import pyarctic


class CIData(object):

    def __init__(self, image, noise_map, ci_pre_cti, ci_pattern, ci_frame, noise_scaling=None):
        self.image = image
        self.noise_map = noise_map
        self.ci_pre_cti = ci_pre_cti
        self.noise_scaling = noise_scaling
        self.ci_pattern = ci_pattern
        self.ci_frame = ci_frame

    @property
    def chinj(self):
        return frame.ChInj(self.ci_frame, self.ci_pattern)

    @property
    def shape(self):
        return self.image.shape

    def map(self, func, mask):
        return CIDataFit(image=func(self.image),
                         noise_map=func(self.noise_map),
                         ci_pre_cti=func(self.ci_pre_cti),
                         mask=func(mask),
                         ci_pattern=self.ci_pattern,
                         ci_frame=self.ci_frame,
                         noise_scaling=func(
                             self.noise_scaling) if self.noise_scaling is not None else self.noise_scaling)

    def parallel_calibration_data(self, columns, mask):
        return self.map(lambda obj: self.chinj.parallel_calibration_section_for_columns(obj, columns), mask)

    def serial_calibration_data(self, column, rows, mask):
        return self.map(
            lambda obj: self.chinj.serial_calibration_section_for_column_and_rows(obj, column=column, rows=rows), mask)

    def parallel_serial_calibration_data(self, mask):
        return self.map(lambda obj: self.chinj.parallel_serial_calibration_section(obj, ), mask)

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

    def __init__(self, image, noise_map, ci_pre_cti, mask, ci_pattern, ci_frame, noise_scaling=None):
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
        self.noise_scaling = noise_scaling
        self.ci_pattern = ci_pattern
        self.ci_frame = ci_frame


class CIImage(np.ndarray):
    def __new__(cls, array, *args, **kwargs):
        return array.view(CIImage)

    @classmethod
    def simulate(cls, shape, frame_geometry, ci_pattern, cti_params, cti_settings, read_noise=None, cosmics=None,
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

        ci_pre_cti = ci_pattern.simulate_ci_pre_cti(shape)
        if cosmics is not None:
            ci_pre_cti += cosmics

        ci_pre_cti = cti_image.ImageFrame(frame_geometry=frame_geometry, array=ci_pre_cti)

        ci_post_cti = ci_pre_cti.add_cti_to_image(cti_params, cti_settings)

        if read_noise is not None:
            ci_post_cti += read_noise_map_from_shape_and_sigma(shape=shape, sigma=read_noise, noise_seed=noise_seed)

        # TODO : This is ugly... fix in future
        sim_image = CIImage(ci_post_cti[:, :])
        sim_image.ci_pre_cti = ci_pre_cti
        return sim_image

    def ci_pre_cti_from_ci_pattern_and_mask(self, ci_pattern, frame_geometry, mask=None):
        """Setup a pre-cti image from this charge injection ci_data, using the charge injection ci_pattern.

        The pre-cti image is computed depending on whether the charge injection ci_pattern is uniform, non-uniform or \
        'fast' (see ChargeInjectPattern).
        """

        if type(ci_pattern) == pattern.CIPatternUniform:

            ci_pre_cti = ci_pattern.ci_pre_cti_from_shape(shape=self.shape)
            return CIPreCTI(frame_geometry=frame_geometry, array=ci_pre_cti, ci_pattern=ci_pattern)

        elif type(ci_pattern) == pattern.CIPatternNonUniform:

            ci_pre_cti = ci_pattern.ci_pre_cti_from_ci_image_and_mask(ci_image=self, mask=mask)
            return CIPreCTI(frame_geometry=frame_geometry, array=ci_pre_cti, ci_pattern=ci_pattern)

        elif type(ci_pattern) == pattern.CIPatternUniformFast:

            ci_pre_cti = ci_pattern.ci_pre_cti_from_shape(shape=self.shape)
            return CIPreCTIFast(frame_geometry=frame_geometry, array=ci_pre_cti, ci_pattern=ci_pattern)

        else:
            raise exc.CIPatternException('the CIPattern of the CIImage is not an instance of '
                                         'a known ci_pattern class')


class CIPreCTI(frame.CIFrame):

    def __init__(self, frame_geometry, array, ci_pattern):
        """The pre-cti image of a charge injection dataset. This image has a corresponding *ChargeInjectPattern*, \
        which describes whether the pre-cti image is uniform or non-uniform injection ci_pattern.

        Params
        ----------
        frame_geometry : ci_frame.CIQuadGeometry
            The quadrant geometry of the image, defining where the parallel / serial overscans are and \
            therefore the direction of clocking and rotations before input into the cti algorithm.
        array : ndarray
            2D Array of pre-cti image ci_data.
        ci_pattern : ci_pattern.CIPattern
            The charge injection ci_pattern (regions, normalization, etc.) of the pre-cti image.
        """
        super(CIPreCTI, self).__init__(frame_geometry, ci_pattern, array)

    def ci_post_cti_from_cti_params_and_settings(self, cti_params, cti_settings):
        """Setup a post-cti image from this pre-cti image, by passing the pre-cti image through a cti clocking \
        algorithm.

        Params
        -----------
        cti_params : ArcticParams.ArcticParams
            The CTI model parameters (trap density, trap lifetimes etc.).
        cti_settings : ArcticSettings.ArcticSettings
            The settings that control the cti clocking algorithm (e.g. ccd well_depth express option).
        """
        ci_post_cti = self.frame_geometry.add_cti(self, cti_params=cti_params, cti_settings=cti_settings)
        return frame.CIFrame(frame_geometry=self.frame_geometry, ci_pattern=self.ci_pattern, array=ci_post_cti)


class CIPreCTIFast(CIPreCTI):

    def __init__(self, frame_geometry, array, ci_pattern):
        """A fast pre-cti image of a charge injection dataset, used for CTI calibration modeling.

        A fast pre-cti image serves the same purpose as a *CIPreCTI* image, but it exploits the fact that for a \
        uniform charge injection image one can add cti to just one column / row of ci_data and copy it across the entire
        image, thus skipping the majority of clocking calls.

        The fast column / row of the pre-cti image setup in the constructor represent the 1 column / row of the \
        uniform charge injection image that is passed to the clocking algorithm.

        See https://euclid.roe.ac.uk/issues/7058

        Parameters
        ----------
        frame_geometry : ci_frame.CIQuadGeometry
            The quadrant geometry of the image, defining where the parallel / serial overscans are and \
            therefore the direction of clocking and rotations before input into the cti algorithm.
        array : ndarray
            2D Array of pre-cti image ci_data.
        ci_pattern : ci_pattern.CIPattern
            The charge injection ci_pattern (regions, normalization, etc.) of the pre-cti image.
        """
        super(CIPreCTIFast, self).__init__(frame_geometry, array, ci_pattern)
        fast_column_pre_cti = self.ci_pattern.compute_fast_column(self.shape[0])
        self.fast_column_pre_cti = self.frame_geometry.rotate_for_parallel_cti(fast_column_pre_cti)
        fast_row_pre_cti = self.ci_pattern.compute_fast_row(self.shape[1])
        self.fast_row_pre_cti = self.frame_geometry.rotate_before_serial_cti(fast_row_pre_cti)

    def fast_column_post_cti_from_cti_params_and_settings(self, cti_params, cti_settings):
        """Add cti to the fast-column, using cti_settings.

        Parameters
        ----------
        cti_params : ArcticParams.ArcticParams
            The CTI model parameters (trap density, trap lifetimes etc.).
        cti_settings : ArcticSettings.ArcticSettings
            The settings that control the cti clocking algorithm (e.g. ccd well_depth express option).
        """
        return pyarctic.call_arctic(self.fast_column_pre_cti, cti_params.parallel_species, cti_params.parallel_ccd,
                                    cti_settings.parallel)

    def map_fast_column_post_cti_to_image(self, fast_column_post_cti):
        """Map the post-cti fast column to the image, thus making the complete post-cti image after parallel clocking.
        """

        ci_post_cti = np.zeros(self.shape)

        fast_column_post_cti = fast_column_post_cti[:, 0][:, np.newaxis]

        ci_post_cti += fast_column_post_cti

        return self.frame_geometry.rotate_for_parallel_cti(ci_post_cti)

    def fast_row_post_cti_from_cti_params_and_settings(self, cti_params, cti_settings):
        """Add cti to the fast-row, using cti_settings.

        Parameters
        ----------
        cti_params : ArcticParams.ArcticParams
            The CTI model parameters (trap density, trap lifetimes etc.).
        cti_settings : ArcticSettings.ArcticSettings
            The settings that control the cti clocking algorithm (e.g. ccd well_depth express option).
        """
        return pyarctic.call_arctic(self.fast_row_pre_cti, cti_params.serial_species, cti_params.serial_ccd,
                                    cti_settings.serial)

    def map_fast_row_post_cti_to_image(self, fast_row_post_cti):
        """Map the post-cti fast row to the image, thus making the complete post-cti image after serial clocking.
        """
        ci_post_cti = np.zeros(self.shape).T

        fast_row_post_cti = fast_row_post_cti[:, 0][:, np.newaxis]

        ci_post_cti += fast_row_post_cti

        return self.frame_geometry.rotate_after_serial_cti(ci_post_cti)

    def add_cti_to_image(self, cti_params, cti_settings):
        """Create the post-cti image of this fast pre-cti image, using the speed up described in the constructor.

        Parameters
        -----------
        cti_params : ArcticParams.ArcticParams
            The CTI model parameters (trap density, trap lifetimes etc.).
        cti_settings : ArcticSettings.ArcticSettings
            The settings that control the cti clocking algorithm (e.g. ccd well_depth express option).
        """

        if cti_settings.parallel is not None and cti_settings.serial is None:

            fast_column_post_cti = self.fast_column_post_cti_from_cti_params_and_settings(cti_params, cti_settings)
            post_cti_image = self.map_fast_column_post_cti_to_image(fast_column_post_cti)

        elif cti_settings.serial is not None and cti_settings.parallel is None:

            fast_row_post_cti = self.fast_row_post_cti_from_cti_params_and_settings(cti_params, cti_settings)
            post_cti_image = self.map_fast_row_post_cti_to_image(fast_row_post_cti)

        else:

            raise exc.CIPreCTIException(' Cannot use CIPostCTIFast in both parallel and serial directions')

        return post_cti_image


def load_ci_data_list(frame_geometries, ci_patterns,
                      ci_image_paths, ci_image_hdus=None,
                      ci_noise_map_paths=None, ci_noise_map_hdus=None,
                      ci_pre_cti_paths=None, ci_pre_cti_hdus=None, ci_pre_cti_from_image=False,
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
        ci_data = load_ci_data(frame_geometry=frame_geometries[data_index], ci_pattern=ci_patterns[data_index],
                               ci_image_path=ci_image_paths[data_index], ci_image_hdu=ci_image_hdus[data_index],
                               ci_noise_map_path=ci_noise_map_paths[data_index],
                               ci_noise_map_hdu=ci_noise_map_hdus[data_index],
                               ci_pre_cti_path=ci_pre_cti_paths[data_index],
                               ci_pre_cti_hdu=ci_pre_cti_hdus[data_index],
                               ci_pre_cti_from_image=ci_pre_cti_from_image, mask=masks[data_index])

        ci_datas.append(ci_data)

    return ci_datas


def load_ci_data(frame_geometry, ci_pattern,
                 ci_image_path, ci_image_hdu=0,
                 ci_noise_map_path=None, ci_noise_map_hdu=0,
                 ci_noise_map_from_single_value=None,
                 ci_pre_cti_path=None, ci_pre_cti_hdu=0, ci_pre_cti_from_image=False,
                 mask=None):
    ci_image = load_ci_image(ci_image_path=ci_image_path, ci_image_hdu=ci_image_hdu)

    ci_noise_map = load_ci_noise_map(frame_geometry=frame_geometry, ci_pattern=ci_pattern,
                                     ci_noise_map_path=ci_noise_map_path, ci_noise_map_hdu=ci_noise_map_hdu,
                                     ci_noise_map_from_single_value=ci_noise_map_from_single_value,
                                     shape=ci_image.shape)

    ci_pre_cti = load_ci_pre_cti(frame_geometry, ci_pattern, ci_pre_cti_path=ci_pre_cti_path,
                                 ci_pre_cti_hdu=ci_pre_cti_hdu,
                                 ci_image=ci_image, ci_pre_cti_from_image=ci_pre_cti_from_image, mask=mask)

    return CIData(image=ci_image, noise_map=ci_noise_map, ci_pre_cti=ci_pre_cti, ci_pattern=ci_pattern,
                  ci_frame=frame_geometry)


def load_ci_image(ci_image_path, ci_image_hdu):
    return CIImage(array=util.numpy_array_from_fits(file_path=ci_image_path, hdu=ci_image_hdu))


def load_ci_noise_map(frame_geometry, ci_pattern, ci_noise_map_path, ci_noise_map_hdu, ci_noise_map_from_single_value,
                      shape):
    if ci_noise_map_path is not None and ci_noise_map_from_single_value is None:
        return frame.CIFrame(frame_geometry=frame_geometry, ci_pattern=ci_pattern,
                             array=util.numpy_array_from_fits(file_path=ci_noise_map_path, hdu=ci_noise_map_hdu))
    elif ci_noise_map_path is None and ci_noise_map_from_single_value is not None:
        return frame.CIFrame.from_single_value(value=ci_noise_map_from_single_value, shape=shape,
                                               frame_geometry=frame_geometry, ci_pattern=ci_pattern)
    else:
        raise exc.CIDataException(
            'You have supplied both a ci_noise_map_path and a ci_noise_map_from_single_value value. Only one quantity '
            'may be supplied.')


def load_ci_pre_cti(frame_geometry, ci_pattern, ci_pre_cti_path, ci_pre_cti_hdu,
                    ci_image=None, ci_pre_cti_from_image=False, mask=None):
    if not ci_pre_cti_from_image:
        return CIPreCTI(frame_geometry=frame_geometry, ci_pattern=ci_pattern,
                        array=util.numpy_array_from_fits(file_path=ci_pre_cti_path, hdu=ci_pre_cti_hdu))

    return ci_image.ci_pre_cti_from_ci_pattern_and_mask(ci_pattern, frame_geometry, mask=mask)


def baseline_noise_map_from_shape_and_sigma(shape, sigma):
    """
    Create the noises used for CTI Calibration, where each value represents the standard deviation of the \
    pixel's assumed Gaussian noises.

    The only source of noises considered for charge injection imaging is read-noises.

    Params
    ----------
    image_shape : (int, int)
        The pixel image_shape of the 2D mask.
    read_noise : float
        The read-noises level, defined as the standard deviation of a Gaussian with mean 0.0.
    """
    return np.ones(shape) * sigma


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


def compute_variances_from_noise(noise):
    """The variances are the noises (standard deviations) squared."""
    return np.square(noise)
