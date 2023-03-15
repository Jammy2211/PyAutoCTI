import numpy as np

import autocti as ac


def test__region_list_from__array_2d_list_from(serial_array, serial_masked_array):

    extract = ac.Extract2DSerialEPER(region_list=[(0, 3, 1, 4)])

    array_2d_list = extract.array_2d_list_from(
        array=serial_array, settings=ac.SettingsExtract(pixels=(0, 1))
    )

    assert (array_2d_list[0] == np.array([[4.0], [4.0], [4.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=serial_array, settings=ac.SettingsExtract(pixels=(1, 2))
    )

    assert (array_2d_list[0] == np.array([[5.0], [5.0], [5.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=serial_array, settings=ac.SettingsExtract(pixels=(2, 3))
    )

    assert (array_2d_list[0] == np.array([[6.0], [6.0], [6.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=serial_array, settings=ac.SettingsExtract(pixels=(0, 2))
    )

    assert (array_2d_list[0] == np.array([[4.0, 5.0], [4.0, 5.0], [4.0, 5.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=serial_array, settings=ac.SettingsExtract(pixels=(-1, 1))
    )

    assert (array_2d_list[0] == np.array([[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=serial_array, settings=ac.SettingsExtract(pixels=(1, 4))
    )

    assert (
        array_2d_list[0]
        == np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
    ).all()

    extract = ac.Extract2DSerialEPER(region_list=[(0, 3, 1, 4), (0, 3, 5, 8)])

    array_2d_list = extract.array_2d_list_from(
        array=serial_array, settings=ac.SettingsExtract(pixels=(0, 1))
    )

    assert (array_2d_list[0] == np.array([[4.0], [4.0], [4.0]])).all()
    assert (array_2d_list[1] == np.array([[8.0], [8.0], [8.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=serial_array, settings=ac.SettingsExtract(pixels=(1, 2))
    )

    assert (array_2d_list[0] == np.array([[5.0], [5.0], [5.0]])).all()
    assert (array_2d_list[1] == np.array([[9.0], [9.0], [9.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=serial_array, settings=ac.SettingsExtract(pixels=(2, 3))
    )

    assert (array_2d_list[0] == np.array([[6.0], [6.0], [6.0]])).all()
    assert (array_2d_list[1] == np.array([[10.0], [10.0], [10.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=serial_array, settings=ac.SettingsExtract(pixels=(0, 3))
    )

    assert (
        array_2d_list[0]
        == np.array([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]])
    ).all()

    assert (array_2d_list[1] == np.array([[8.0, 9.0], [8.0, 9.0], [8.0, 9.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=serial_masked_array, settings=ac.SettingsExtract(pixels=(0, 3))
    )

    assert (
        array_2d_list[0].mask
        == np.array([[False, True, False], [False, False, True], [False, False, False]])
    ).all()

    assert (
        array_2d_list[1].mask
        == np.array([[False, False], [False, False], [False, False]])
    ).all()


def test__region_list_from__array_2d_list_from__pixels_from_end(
    serial_array, serial_masked_array
):

    extract = ac.Extract2DSerialEPER(
        shape_2d=serial_array.shape_native, region_list=[(0, 3, 1, 4)]
    )

    array_2d_list = extract.array_2d_list_from(
        array=serial_array, settings=ac.SettingsExtract(pixels_from_end=1)
    )

    assert (array_2d_list[0] == np.array([[9.0], [9.0], [9.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=serial_array, settings=ac.SettingsExtract(pixels_from_end=2)
    )

    assert (array_2d_list[0] == np.array([[8.0, 9.0], [8.0, 9.0], [8.0, 9.0]])).all()


def test__array_2d_from():

    extract = ac.Extract2DSerialEPER(
        region_list=[(0, 4, 0, 2)], serial_overscan=(0, 4, 2, 3)
    )

    array = ac.Array2D.no_mask(
        values=[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        pixel_scales=1.0,
    )

    array_2d = extract.array_2d_from(array=array)

    assert (
        array_2d
        == np.array(
            [[0.0, 0.0, 2.0], [0.0, 0.0, 5.0], [0.0, 0.0, 8.0], [0.0, 0.0, 11.0]]
        )
    ).all()

    extract = ac.Extract2DSerialEPER(
        region_list=[(0, 4, 0, 2)], serial_overscan=(0, 4, 2, 4)
    )

    array = ac.Array2D.no_mask(
        values=[
            [0.0, 1.0, 2.0, 0.5],
            [3.0, 4.0, 5.0, 0.5],
            [6.0, 7.0, 8.0, 0.5],
            [9.0, 10.0, 11.0, 0.5],
        ],
        pixel_scales=1.0,
    )

    array_2d = extract.array_2d_from(array=array)

    assert (
        array_2d
        == np.array(
            [
                [0.0, 0.0, 2.0, 0.5],
                [0.0, 0.0, 5.0, 0.5],
                [0.0, 0.0, 8.0, 0.5],
                [0.0, 0.0, 11.0, 0.5],
            ]
        )
    ).all()


def test__binned_region_1d_from():

    extract = ac.Extract2DSerialEPER(region_list=[(1, 3, 0, 3)])

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
