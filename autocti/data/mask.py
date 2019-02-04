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
    def empty_for_shape(cls, shape):
        """
        Create the mask used for CTI Calibration as all False's (e.g. no masking).

        Parameters
        ----------
        ci_pattern
        shape : (int, int)
            The dimensions of the 2D mask.
        frame_geometry : ci_frame.CIQuadGeometry
            The quadrant geometry of the simulated image, defining where the parallel / serial overscans are and \
            therefore the direction of clocking and rotations before input into the cti algorithm.
        """
        # noinspection PyArgumentList
        return cls(array=np.full(shape, False))

    @classmethod
    def create(cls, shape, frame_geometry, regions=None, cosmic_rays=None, cr_parallel=0, cr_serial=0,
               cr_diagonal=0):
        """
        Create the mask used for CTI Calibration, which is all False unless specific regions are input for masking.

        Parameters
        ----------
        cr_diagonal
        shape : (int, int)
            The dimensions of the 2D mask.
        frame_geometry : ci_frame.CIQuadGeometry
            The quadrant geometry of the simulated image, defining where the parallel / serial overscans are and \
            therefore the direction of clocking and rotations before input into the cti algorithm.
        regions : [(int,)]
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
        mask = cls.empty_for_shape(shape)

        if regions is not None:
            mask.regions = list(map(lambda r: ci_frame.Region(r), regions))
            for region in mask.regions:
                mask[region.y0:region.y1, region.x0:region.x1] = True
        elif regions is None:
            mask.regions = None

        if cosmic_rays is not None:
            for y in range(mask.shape[0]):
                for x in range(mask.shape[1]):
                    if cosmic_rays[y, x]:
                        y0, y1 = frame_geometry.parallel_trail_from_y(y, cr_parallel)
                        mask[y0:y1, x] = True
                        x0, x1 = frame_geometry.serial_trail_from_x(x, cr_serial)
                        mask[y, x0:x1] = True
                        y0, y1 = frame_geometry.parallel_trail_from_y(y, cr_diagonal)
                        x0, x1 = frame_geometry.serial_trail_from_x(x, cr_diagonal)
                        mask[y0:y1, x0:x1] = True

        elif cosmic_rays is None:
            mask.cosmic_rays = None

        return mask
