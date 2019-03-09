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
File: tests/python/util_test.py

Created on: 04/23/18
Author: user
"""

import os
import shutil

import numpy as np
import pytest

from autocti.data import util

path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))

test_data_path = "{}/../test_files/array/".format(os.path.dirname(os.path.realpath(__file__)))


class TestFits:

    def test__numpy_array_from_fits__3x3_all_ones(self):
        arr = util.numpy_array_2d_from_fits(file_path=test_data_path + '3x3_ones.fits', hdu=0)

        assert (arr == np.ones((3, 3))).all()

    def test__numpy_array_from_fits__4x3_all_ones(self):
        arr = util.numpy_array_2d_from_fits(file_path=test_data_path + '4x3_ones.fits', hdu=0)

        assert (arr == np.ones((4, 3))).all()

    def test__numpy_array_to_fits__output_and_load(self):
        if os.path.exists(test_data_path + 'test.fits'):
            os.remove(test_data_path + 'test.fits')

        arr = np.array([[10., 30., 40.],
                        [92., 19., 20.]])

        util.numpy_array_2d_to_fits(arr, file_path=test_data_path + 'test.fits')

        array_load = util.numpy_array_2d_from_fits(file_path=test_data_path + 'test.fits', hdu=0)

        assert arr == pytest.approx(array_load, 1.0e-4)


class TestDirectories:

    def test__1_directory_input__makes_directory__returns_path(self):

        path = util.make_and_return_path(path=test_data_path, folder_names=['test1'])

        assert path == test_data_path + 'test1/'
        assert os.path.exists(path=test_data_path+'test1')

        shutil.rmtree(test_data_path+'test1')

    def test__multiple_directories_input__makes_directory_structure__returns_full_path(self):

        path = util.make_and_return_path(path=test_data_path, folder_names=['test1', 'test2', 'test3'])

        assert path == test_data_path + 'test1/test2/test3/'
        assert os.path.exists(path=test_data_path+'test1')
        assert os.path.exists(path=test_data_path+'test1/test2')
        assert os.path.exists(path=test_data_path+'test1/test2/test3')

        shutil.rmtree(test_data_path+'test1')