import os

import numpy as np
import autocti as ac

test_data_path = "{}/files/array/".format(os.path.dirname(os.path.realpath(__file__)))


def test__total_image_pixels_from_mask():
    mask = np.array([[True, False, True], [False, False, False], [True, False, True]])

    assert ac.util.mask.total_pixels_from_mask(mask=mask) == 5
