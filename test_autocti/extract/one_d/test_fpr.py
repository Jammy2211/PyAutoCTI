import numpy as np

import autocti as ac


def test__region_list_from__via_array_1d_list_from(array, masked_array):
    extract = ac.Extract1DFPR(region_list=[(1, 4)])

    fpr_list = extract.array_1d_list_from(
        array=array, settings=ac.SettingsExtract(pixels=(0, 1))
    )
    assert (fpr_list[0] == np.array([1.0])).all()

    fpr = extract.array_1d_list_from(
        array=array, settings=ac.SettingsExtract(pixels=(2, 3))
    )
    assert (fpr[0] == np.array([3.0])).all()

    fpr = extract.array_1d_list_from(
        array=array, settings=ac.SettingsExtract(pixels=(-1, 1))
    )
    assert (fpr[0] == np.array([0.0, 1.0])).all()

    extract = ac.Extract1DFPR(region_list=[(1, 4), (5, 8)])

    fpr_list = extract.array_1d_list_from(
        array=array, settings=ac.SettingsExtract(pixels=(0, 1))
    )
    assert (fpr_list[0] == np.array([[1.0]])).all()
    assert (fpr_list[1] == np.array([[5.0]])).all()

    fpr_list = extract.array_1d_list_from(
        array=array, settings=ac.SettingsExtract(pixels=(2, 3))
    )
    assert (fpr_list[0] == np.array([[3.0]])).all()
    assert (fpr_list[1] == np.array([[7.0]])).all()

    fpr_list = extract.array_1d_list_from(
        array=array, settings=ac.SettingsExtract(pixels=(0, 3))
    )
    assert (fpr_list[0] == np.array([1.0, 2.0, 3.0])).all()
    assert (fpr_list[1] == np.array([5.0, 6.0, 7.0])).all()

    fpr_list = extract.array_1d_list_from(
        array=masked_array, settings=ac.SettingsExtract(pixels=(0, 3))
    )

    assert (fpr_list[0].mask == np.array([False, True, False])).all()


def test__region_list_from__via_array_1d_list_from__pixels_from_end(
    array, masked_array
):
    extract = ac.Extract1DFPR(region_list=[(1, 4)])

    fpr = extract.array_1d_list_from(
        array=array, settings=ac.SettingsExtract(pixels_from_end=1)
    )
    assert (fpr[0] == np.array([3.0])).all()

    extract = ac.Extract1DFPR(region_list=[(1, 4), (5, 8)])

    fpr_list = extract.array_1d_list_from(
        array=array, settings=ac.SettingsExtract(pixels_from_end=1)
    )
    assert (fpr_list[0] == np.array([[3.0]])).all()
    assert (fpr_list[1] == np.array([[7.0]])).all()

    fpr_list = extract.array_1d_list_from(
        array=array, settings=ac.SettingsExtract(pixels_from_end=3)
    )
    assert (fpr_list[0] == np.array([1.0, 2.0, 3.0])).all()
    assert (fpr_list[1] == np.array([5.0, 6.0, 7.0])).all()

    fpr_list = extract.array_1d_list_from(
        array=masked_array, settings=ac.SettingsExtract(pixels_from_end=3)
    )

    assert (fpr_list[0].mask == np.array([False, True, False])).all()
