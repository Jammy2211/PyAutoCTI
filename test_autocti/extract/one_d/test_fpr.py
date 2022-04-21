import numpy as np

import autocti as ac


def test__array_1d_list_from(array, masked_array):

    extract = ac.Extract1DFPR(region_list=[(1, 4)])

    fpr_list = extract.array_1d_list_from(array=array, pixels=(0, 1))
    assert (fpr_list[0] == np.array([1.0])).all()

    fpr = extract.array_1d_list_from(array=array, pixels=(2, 3))
    assert (fpr[0] == np.array([3.0])).all()

    extract = ac.Extract1DFPR(region_list=[(1, 4), (5, 8)])

    fpr_list = extract.array_1d_list_from(array=array, pixels=(0, 1))
    assert (fpr_list[0] == np.array([[1.0]])).all()
    assert (fpr_list[1] == np.array([[5.0]])).all()

    fpr_list = extract.array_1d_list_from(array=array, pixels=(2, 3))
    assert (fpr_list[0] == np.array([[3.0]])).all()
    assert (fpr_list[1] == np.array([[7.0]])).all()

    fpr_list = extract.array_1d_list_from(array=array, pixels=(0, 3))
    assert (fpr_list[0] == np.array([1.0, 2.0, 3.0])).all()
    assert (fpr_list[1] == np.array([5.0, 6.0, 7.0])).all()

    fpr_list = extract.array_1d_list_from(array=masked_array, pixels=(0, 3))

    assert (fpr_list[0].mask == np.array([False, True, False])).all()


def test__stacked_array_1d_from(array, masked_array):

    extract = ac.Extract1DFPR(region_list=[(1, 4), (5, 8)])

    stacked_fprs = extract.stacked_array_1d_from(array=array, pixels=(0, 3))

    assert (stacked_fprs == np.array([3.0, 4.0, 5.0])).all()

    extract = ac.Extract1DFPR(region_list=[(1, 3), (5, 8)])

    stacked_fprs = extract.stacked_array_1d_from(array=array, pixels=(0, 2))

    assert (stacked_fprs == np.array([3.0, 4.0])).all()

    stacked_fprs = extract.stacked_array_1d_from(array=masked_array, pixels=(0, 2))

    assert (stacked_fprs == np.ma.array([1.0, 6.0])).all()
    assert (stacked_fprs.mask == np.ma.array([False, False])).all()
