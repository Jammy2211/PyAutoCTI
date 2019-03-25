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

from autocti.charge_injection import ci_frame


class Mask(np.ndarray):

    def __new__(cls, array, *args, **kwargs):
        mask = np.array(array, dtype='float64').view(cls)
        return mask

    @classmethod
    def empty_for_shape(cls, shape, **kwargs):
        """
        Create the mask used for CTI Calibration as all False's (e.g. no masking).

        Parameters
        ----------
        shape : (int, int)
            The dimensions of the 2D mask.
        """
        # noinspection PyArgumentList
        return cls(array=np.full(shape=shape, fill_value=False))

    @classmethod
    def from_masked_regions(cls, shape, masked_regions, **kwargs):

        mask = cls.empty_for_shape(shape)
        masked_regions = list(map(lambda r: ci_frame.Region(r), masked_regions))
        for region in masked_regions:
            mask[region.y0:region.y1, region.x0:region.x1] = True

        return mask

    @classmethod
    def from_cosmic_ray_image(cls, shape, frame_geometry, cosmic_ray_image, cosmic_ray_parallel_buffer=0,
                              cosmic_ray_serial_buffer=0, cosmic_ray_diagonal_buffer=0, **kwargs):
        """
        Create the mask used for CTI Calibration, which is all False unless specific regions are input for masking.

        Parameters
        ----------
        shape : (int, int)
            The dimensions of the 2D mask.
        frame_geometry : ci_frame.CIQuadGeometry
            The quadrant geometry of the simulated image, defining where the parallel / serial overscans are and \
            therefore the direction of clocking and rotations before input into the cti algorithm.
        cosmic_ray_image : ndarray
            2D array flagging where cosmic rays on the image.
        cosmic_ray_parallel_buffer : int
            If a cosmic-ray mask is supplied, the number of pixels from each ray pixels are masked in the parallel \
            direction.
        cosmic_ray_serial_buffer : int
            If a cosmic-ray mask is supplied, the number of pixels from each ray pixels are masked in the serial \
            direction.
        """
        mask = cls.empty_for_shape(shape)

        cosmic_ray_mask = (cosmic_ray_image > 0.0).astype('bool')

        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if cosmic_ray_mask[y, x]:
                    y0, y1 = frame_geometry.parallel_trail_from_y(y, cosmic_ray_parallel_buffer)
                    mask[y0:y1, x] = True
                    x0, x1 = frame_geometry.serial_trail_from_x(x, cosmic_ray_serial_buffer)
                    mask[y, x0:x1] = True
                    y0, y1 = frame_geometry.parallel_trail_from_y(y, cosmic_ray_diagonal_buffer)
                    x0, x1 = frame_geometry.serial_trail_from_x(x, cosmic_ray_diagonal_buffer)
                    mask[y0:y1, x0:x1] = True

        return mask
