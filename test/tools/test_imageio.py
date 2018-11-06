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
File: tests/python/imageio_test.py

Created on: 04/23/18
Author: user
"""

from __future__ import division, print_function
import sys
if sys.version_info[0] < 3:
    from future_builtins import *

from autocti.tools import imageio

import pytest
import os
import numpy as np

path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))


class TestFits:

    def test__numpy_array_from_fits__3x3_all_ones(self):

        array = imageio.numpy_array_from_fits(path=path + 'files/fits/', filename='3x3_ones', hdu=0)

        assert (array == np.ones((3,3))).all()

    def test__numpy_array_from_fits__4x3_all_ones(self):

        array = imageio.numpy_array_from_fits(path=path + 'files/fits/', filename='4x3_ones', hdu=0)

        assert (array == np.ones((4,3))).all()

    def test__numpy_array_to_fits__output_and_load(self):

        if os.path.exists(path+'files/fits/test.fits'):
            os.remove(path+'files/fits/test.fits')

        array = np.array([[10., 30., 40.],
                          [92., 19., 20.]])

        imageio.numpy_array_to_fits(array, path=path + 'files/fits/', filename='test')

        array_load = imageio.numpy_array_from_fits(path=path + 'files/fits/', filename='test', hdu=0)

        assert (array == array_load).all()