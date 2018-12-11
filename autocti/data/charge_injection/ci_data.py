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

if sys.version_info[0] < 3:
    from future_builtins import *

import numpy as np

from autocti.data.charge_injection import ci_frame
from autocti.data.charge_injection import ci_pattern
from autocti.data import cti_image
from autocti.model import pyarctic
from autocti.tools import infoio
from autocti import exc

class CIData(list):

    def __init__(self, images, masks, noises, ci_pre_ctis):

        super(CIData, self).__init__()

        class DataSet(object):

            def __init__(self, image, mask, noise, ci_pre_cti):

                self.image = image
                self.mask = mask
                self.noise = noise
                self.ci_pre_cti = ci_pre_cti
                self.noise_scalings = None

        for i in range(len(images)):
            self.append(DataSet(images[i], masks[i], noises[i], ci_pre_ctis[i]))


class CIDataAnalysis(CIData):

    def __init__(self, images, masks, noises, ci_pre_ctis, noise_scalings=None):

        super(CIDataAnalysis, self).__init__(images, masks, noises, ci_pre_ctis)

        if noise_scalings is not None:
            for i in range(len(noise_scalings)):
                self[i].noise_scalings = noise_scalings[i]


class CIImage(ci_frame.CIFrameCTI):

    def __init__(self, frame_geometry, ci_pattern, array):
        """The observed charge injection imaging ci_data.

        Parameters
        ----------
        frame_geometry : ci_frame.CIQuadGeometry
            The quadrant geometry of the image, defining where the parallel / serial overscans are and \
            therefore the direction of clocking and rotations before input into the cti algorithm.
        ci_pattern : ci_pattern.CIPattern
            The charge injection ci_pattern (regions, normalization, etc.) of the charge injection image.
        array : ndarray
            2D Array of array charge injection image ci_data.
        """

        super(CIImage, self).__init__(frame_geometry, ci_pattern, array=array)

    @classmethod
    def simulate(cls, shape, frame_geometry, ci_pattern, cti_params, cti_settings, read_noise=None, cosmics=None,
                 noise_seed=-1):
        """Simulate a charge injection image, including effects like noises.

        Parameters
        -----------
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

        ci_pattern = ci_pattern.create_pattern()

        ci_pre_cti = cti_image.CTIImage(frame_geometry=frame_geometry, array=ci_pre_cti)

        ci_post_cti = ci_pre_cti.add_cti_to_image(cti_params, cti_settings)

        if read_noise is not None:
            ci_post_cti += create_read_noise_map(shape=shape, read_noise=read_noise, noise_seed=noise_seed)

        # TODO : This is ugly... fix in future
        sim_image = CIImage(frame_geometry=frame_geometry, ci_pattern=ci_pattern, array=ci_post_cti[:, :])
        sim_image.ci_pre_cti = ci_pre_cti
        return sim_image

    def create_ci_pre_cti(self, mask=None):
        """Setup a pre-cti image from this charge injection ci_data, using the charge injection ci_pattern.

        The pre-cti image is computed depending on whether the charge injection ci_pattern is uniform, non-uniform or \
        'fast' (see ChargeInjectPattern).
        """

        if type(self.ci_pattern) == ci_pattern.CIPatternUniform:

            ci_pre_cti = self.ci_pattern.compute_ci_pre_cti(self.shape)
            return CIPreCTI(frame_geometry=self.frame_geometry, array=ci_pre_cti, ci_pattern=self.ci_pattern)

        elif type(self.ci_pattern) == ci_pattern.CIPatternNonUniform:

            ci_pre_cti = self.ci_pattern.compute_ci_pre_cti(self, mask)
            return CIPreCTI(frame_geometry=self.frame_geometry, array=ci_pre_cti, ci_pattern=self.ci_pattern)

        elif type(self.ci_pattern) == ci_pattern.CIPatternUniformFast:

            ci_pre_cti = self.ci_pattern.compute_ci_pre_cti(self.shape)
            return CIPreCTIFast(frame_geometry=self.frame_geometry, array=ci_pre_cti, ci_pattern=self.ci_pattern)

        else:
            raise exc.CIPatternException('the CIPattern of the CIImage is not an instance of '
                                                  'a known ci_pattern class')

    def generate_info(self):
        """Generate string containing information on the charge injection image (and its ci_pattern)."""
        info = infoio.generate_class_info(self, prefix='ci_data_', include_types=[float])
        return info


class CIMask(ci_frame.CIFrame):

    @classmethod
    def empty_for_shape(cls, shape, frame_geometry, ci_pattern):
        """
        Create the mask used for CTI Calibration as all False's (e.g. no masking).

        Parameters
        ----------
        image_shape : (int, int)
            The dimensions of the 2D mask.
        frame_geometry : ci_frame.CIQuadGeometry
            The quadrant geometry of the simulated image, defining where the parallel / serial overscans are and \
            therefore the direction of clocking and rotations before input into the cti algorithm.
        """
        return cls(frame_geometry=frame_geometry, ci_pattern=ci_pattern, array=np.full(shape, False))

    @classmethod
    def create(cls, shape, frame_geometry, ci_pattern, regions=None, cosmic_rays=None, cr_parallel=0, cr_serial=0,
               cr_diagonal=0):
        """
        Create the mask used for CTI Calibration, which is all False unless spsecific regions are input for masking.

        Parameters
        ----------
        image_shape : (int, int)
            The dimensions of the 2D mask.
        frame_geometry : ci_frame.CIQuadGeometry
            The quadrant geometry of the simulated image, defining where the parallel / serial overscans are and \
            therefore the direction of clocking and rotations before input into the cti algorithm.
        regionss : [(int, int, int, int)]
            A list of the regions on the image to mask.
        cosmic_rays : ndarray.ma
            2D array flagging where cosmic rays on the image.
        cr_parallel : int
            If a cosmic-ray mask is supplied, the number of pixels from each ray pixels are masked in the parallel \
            direction.
        cr_serial : int
            If a cosmic-ray mask is supplied, the number of pixels from each ray pixels are masked in the serial \
            direction.
        """
        mask = CIMask.empty_for_shape(shape, frame_geometry, ci_pattern)
        
        if regions is not None:
            mask.regions = list(map(lambda region: cti_image.Region(region), regions))
            for region in mask.regions:
                mask[region.y0:region.y1, region.x0:region.x1] = True
        elif regions is None:
            mask.regions = None

        if cosmic_rays is not None:
            for y in range(mask.shape[0]):
                for x in range(mask.shape[1]):
                    if cosmic_rays[y, x]:
                        y0, y1 = mask.frame_geometry.parallel_trail_from_y(y, cr_parallel)
                        mask[y0:y1, x] = True
                        x0, x1 = mask.frame_geometry.serial_trail_from_x(x, cr_serial)
                        mask[y, x0:x1] = True
                        y0, y1 = mask.frame_geometry.parallel_trail_from_y(y, cr_diagonal)
                        x0, x1 = mask.frame_geometry.serial_trail_from_x(x, cr_diagonal)
                        mask[y0:y1, x0:x1] = True

        elif cosmic_rays is None:
            mask.cosmic_rays = None

        return mask


class CIPreCTI(ci_frame.CIFrameCTI):

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

    def create_ci_post_cti(self, cti_params, cti_settings):
        """Setup a post-cti image from this pre-cti image, by passing the pre-cti image through a cti clocking \
        algorithm.

        Params
        -----------
        cti_params : ArcticParams.ArcticParams
            The CTI model parameters (trap density, trap lifetimes etc.).
        cti_settings : ArcticSettings.ArcticSettings
            The settings that control the cti clocking algorithm (e.g. ccd well_depth express option).
        """
        ci_post_cti = self.add_cti_to_image(cti_params, cti_settings)
        return ci_frame.CIFrame(frame_geometry=self.frame_geometry, array=ci_post_cti, ci_pattern=self.ci_pattern)


class CIPreCTIFast(CIPreCTI):

    def __init__(self, frame_geometry, array, ci_pattern):
        """A fast pre-cti image of a charge injection dataset, used for CTI calibration modeling.

        A fast pre-cti image serves the same purpose as a *CIPreCTI* image, but it exploits the fact that for a \
        uniform charge injection image one can add cti to just one column / row of ci_data and copy it across the entire \
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
        self.fast_column_pre_cti = self.frame_geometry.rotate_before_parallel_cti(fast_column_pre_cti)
        fast_row_pre_cti = self.ci_pattern.compute_fast_row(self.shape[1])
        self.fast_row_pre_cti = self.frame_geometry.rotate_before_serial_cti(fast_row_pre_cti)

    def compute_fast_column_post_cti(self, cti_params, cti_settings):
        """Add cti to the fast-column, using cti_settings.

        Parameters
        ----------
        cti_params : ArcticParams.ArcticParams
            The CTI model parameters (trap density, trap lifetimes etc.).
        cti_settings : ArcticSettings.ArcticSettings
            The settings that control the cti clocking algorithm (e.g. ccd well_depth express option).
        """
        return pyarctic.add_parallel_cti_to_image(self.fast_column_pre_cti, cti_params, cti_settings)

    def map_fast_column_post_cti_to_image(self, fast_column_post_cti):
        """Map the post-cti fast column to the image, thus making the complete post-cti image after parallel clocking.
        """

        ci_post_cti = np.zeros(self.shape)

        fast_column_post_cti = fast_column_post_cti[:, 0][:, np.newaxis]

        ci_post_cti += fast_column_post_cti

        return self.frame_geometry.rotate_after_parallel_cti(ci_post_cti)

    def compute_fast_row_post_cti(self, cti_params, cti_settings):
        """Add cti to the fast-row, using cti_settings.

        Parameters
        ----------
        cti_params : ArcticParams.ArcticParams
            The CTI model parameters (trap density, trap lifetimes etc.).
        cti_settings : ArcticSettings.ArcticSettings
            The settings that control the cti clocking algorithm (e.g. ccd well_depth express option).
        """
        return pyarctic.add_serial_cti_to_image(self.fast_row_pre_cti, cti_params, cti_settings)

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

            fast_column_post_cti = self.compute_fast_column_post_cti(cti_params, cti_settings)
            post_cti_image = self.map_fast_column_post_cti_to_image(fast_column_post_cti)

        elif cti_settings.serial is not None and cti_settings.parallel is None:

            fast_row_post_cti = self.compute_fast_row_post_cti(cti_params, cti_settings)
            post_cti_image = self.map_fast_row_post_cti_to_image(fast_row_post_cti)

        else:

            raise exc.CIPreCTIException(' Cannot use CIPostCTIFast in both parallel and serial directions')

        return post_cti_image


def create_baseline_noise(shape, read_noise):
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
    return np.ones(shape) * read_noise

def create_read_noise_map(shape, read_noise, noise_seed=-1):
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
    read_noise_map = np.random.normal(loc=0.0, scale=read_noise, size=shape)
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
        seed = np.random.randint(0, 1e9)  # Use one seed, so all regions have identical column non-uniformity.
    np.random.seed(seed)

def compute_variances_from_noise(noise):
    """The variances are the noises (standard deviations) squared."""
    return np.square(noise)