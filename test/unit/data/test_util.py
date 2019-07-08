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
