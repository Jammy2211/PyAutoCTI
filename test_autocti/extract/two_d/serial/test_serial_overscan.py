import numpy as np

import autocti as ac


def test__region_list_from(serial_array, serial_masked_array):
    extract = ac.Extract2DSerialOverscan(serial_overscan=(0, 3, 1, 4))

    array_2d_list = extract.array_2d_list_from(
        array=serial_array, settings=ac.SettingsExtract(pixels=(0, 1))
    )

    assert (array_2d_list[0] == np.array([[1.0], [1.0], [1.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=serial_array, settings=ac.SettingsExtract(pixels=(1, 2))
    )

    assert (array_2d_list[0] == np.array([[2.0], [2.0], [2.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=serial_array, settings=ac.SettingsExtract(pixels=(2, 3))
    )

    assert (array_2d_list[0] == np.array([[3.0], [3.0], [3.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=serial_array, settings=ac.SettingsExtract(pixels=(-1, 1))
    )

    assert (array_2d_list[0] == np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])).all()

    extract = ac.Extract2DSerialOverscan(serial_overscan=(0, 3, 1, 5))

    array_2d_list = extract.array_2d_list_from(
        array=serial_array, settings=ac.SettingsExtract(pixels=(0, 2))
    )

    assert (array_2d_list[0] == np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=serial_array, settings=ac.SettingsExtract(pixels=(1, 4))
    )

    assert (
        array_2d_list[0]
        == np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
    ).all()

    array_2d_list = extract.array_2d_list_from(
        array=serial_masked_array, settings=ac.SettingsExtract(pixels=(0, 3))
    )

    assert (
        (array_2d_list[0].mask)
        == np.array(
            [[False, False, False], [False, True, False], [False, False, False]]
        )
    ).all()


def test__region_list_from__pixels_from_end(serial_array, serial_masked_array):
    extract = ac.Extract2DSerialOverscan(serial_overscan=(0, 3, 1, 4))

    array_2d_list = extract.array_2d_list_from(
        array=serial_array, settings=ac.SettingsExtract(pixels_from_end=1)
    )

    assert (array_2d_list[0] == np.array([[3.0], [3.0], [3.0]])).all()

    extract = ac.Extract2DSerialOverscan(serial_overscan=(0, 3, 1, 5))

    array_2d_list = extract.array_2d_list_from(
        array=serial_array, settings=ac.SettingsExtract(pixels_from_end=3)
    )

    assert (
        array_2d_list[0]
        == np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
    ).all()

    array_2d_list = extract.array_2d_list_from(
        array=serial_masked_array, settings=ac.SettingsExtract(pixels_from_end=3)
    )

    assert (
        (array_2d_list[0].mask)
        == np.array(
            [[False, False, False], [True, False, False], [False, False, False]]
        )
    ).all()


def test__binned_region_1d_from():
    extract = ac.Extract2DSerialOverscan(region_list=[(1, 3, 0, 3)])

    binned_region_1d_list = extract.binned_region_1d_from(
        settings=ac.SettingsExtract(pixels=(0, 1))
    )

    assert binned_region_1d_list == None

    binned_region_1d_list = extract.binned_region_1d_from(
        settings=ac.SettingsExtract(pixels=(-1, 1))
    )

    assert binned_region_1d_list == (0, 1)

    binned_region_1d_list = extract.binned_region_1d_from(
        settings=ac.SettingsExtract(pixels=(-7, 18))
    )

    assert binned_region_1d_list == (0, 7)

    binned_region_1d_list = extract.binned_region_1d_from(
        settings=ac.SettingsExtract(pixels=(-3, -1))
    )

    assert binned_region_1d_list == (0, 2)
