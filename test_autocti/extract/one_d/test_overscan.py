import numpy as np

import autocti as ac


def test__array_1d_list_from(array, masked_array):

    extract = ac.Extract1DOverscan(overscan=(1, 4))

    overscan_list = extract.array_1d_list_from(array=array, pixels=(0, 1))

    assert (overscan_list[0] == np.array([1.0])).all()

    overscan = extract.array_1d_list_from(array=array, pixels=(2, 3))
    assert (overscan[0] == np.array([3.0])).all()

    overscan_list = extract.array_1d_list_from(array=array, pixels=(0, 3))
    assert (overscan_list[0] == np.array([1.0, 2.0, 3.0])).all()

    overscan_list = extract.array_1d_list_from(array=array, pixels=(-1, 1))
    assert (overscan_list[0] == np.array([0.0, 1.0])).all()

    overscan_list = extract.array_1d_list_from(array=masked_array, pixels=(0, 3))

    assert (overscan_list[0].mask == np.array([False, True, False])).all()
