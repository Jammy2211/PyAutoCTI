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

    overscan_list = extract.array_1d_list_from(array=masked_array, pixels=(0, 3))

    assert (overscan_list[0].mask == np.array([False, True, False])).all()


def test__stacked_array_1d_from(array, masked_array):

    extract = ac.Extract1DOverscan(overscan=(1, 4))

    stacked_overscans = extract.stacked_array_1d_from(array=array, pixels=(0, 3))

    assert (stacked_overscans == np.array([1.0, 2.0, 3.0])).all()

    extract = ac.Extract1DOverscan(overscan=(1, 3))

    stacked_overscans = extract.stacked_array_1d_from(array=array, pixels=(0, 2))

    assert (stacked_overscans == np.array([1.0, 2.0])).all()

    stacked_overscans = extract.stacked_array_1d_from(array=masked_array, pixels=(0, 2))

    assert (stacked_overscans == np.ma.array([1.0, 0.0])).all()
    assert (stacked_overscans.mask == np.ma.array([False, True])).all()
