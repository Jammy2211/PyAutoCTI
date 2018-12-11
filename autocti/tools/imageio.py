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
File: python/SHE_ArCTIC/ImageIO.py

Created on: 04/23/18
Author: James Nightingale
"""

import os

import numpy as np
from astropy.io import fits


def make_path_if_does_not_exist(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def numpy_array_to_fits(array, path, filename):
    make_path_if_does_not_exist(path)

    try:
        os.remove(path + filename + '.fits')
    except OSError:
        pass

    new_hdr = fits.Header()
    hdu = fits.PrimaryHDU(array, new_hdr)
    hdu.writeto(path + filename + '.fits')


def numpy_array_from_fits(path, filename, hdu):
    hdu_list = fits.open(path + filename + '.fits')
    return np.array(hdu_list[hdu].data)
