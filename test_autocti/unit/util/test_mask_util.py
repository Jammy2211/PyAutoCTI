import os

import numpy as np
from autocti import util

test_data_path = "{}/files/array/".format(os.path.dirname(os.path.realpath(__file__)))


def test__total_image_pixels_from_mask():
    mask = np.array([[True, False, True], [False, False, False], [True, False, True]])

    assert util.mask.total_pixels_from_mask_2d(mask_2d=mask) == 5
