import numpy as np

import autocti as ac


def test__region_list_from__array_2d_list_from(serial_array, serial_masked_array):

    extract = ac.Extract2DSerialEPER(region_list=[(0, 3, 1, 4)])

    eper_list = extract.array_2d_list_from(array=serial_array, pixels=(0, 1))

    assert (eper_list[0] == np.array([[4.0], [4.0], [4.0]])).all()

    eper_list = extract.array_2d_list_from(array=serial_array, pixels=(1, 2))

    assert (eper_list[0] == np.array([[5.0], [5.0], [5.0]])).all()

    eper_list = extract.array_2d_list_from(array=serial_array, pixels=(2, 3))

    assert (eper_list[0] == np.array([[6.0], [6.0], [6.0]])).all()

    eper_list = extract.array_2d_list_from(array=serial_array, pixels=(0, 2))

    assert (eper_list[0] == np.array([[4.0, 5.0], [4.0, 5.0], [4.0, 5.0]])).all()

    eper_list = extract.array_2d_list_from(array=serial_array, pixels=(-1, 1))

    assert (eper_list[0] == np.array([[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]])).all()

    eper_list = extract.array_2d_list_from(array=serial_array, pixels=(1, 4))

    assert (
        eper_list[0] == np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
    ).all()

    extract = ac.Extract2DSerialEPER(region_list=[(0, 3, 1, 4), (0, 3, 5, 8)])

    eper_list = extract.array_2d_list_from(array=serial_array, pixels=(0, 1))

    assert (eper_list[0] == np.array([[4.0], [4.0], [4.0]])).all()
    assert (eper_list[1] == np.array([[8.0], [8.0], [8.0]])).all()

    eper_list = extract.array_2d_list_from(array=serial_array, pixels=(1, 2))

    assert (eper_list[0] == np.array([[5.0], [5.0], [5.0]])).all()
    assert (eper_list[1] == np.array([[9.0], [9.0], [9.0]])).all()

    eper_list = extract.array_2d_list_from(array=serial_array, pixels=(2, 3))

    assert (eper_list[0] == np.array([[6.0], [6.0], [6.0]])).all()
    assert (eper_list[1] == np.array([[10.0], [10.0], [10.0]])).all()

    eper_list = extract.array_2d_list_from(array=serial_array, pixels=(0, 3))

    assert (
        eper_list[0] == np.array([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]])
    ).all()

    assert (eper_list[1] == np.array([[8.0, 9.0], [8.0, 9.0], [8.0, 9.0]])).all()

    eper_list = extract.array_2d_list_from(array=serial_masked_array, pixels=(0, 3))

    assert (
        eper_list[0].mask
        == np.array([[False, True, False], [False, False, True], [False, False, False]])
    ).all()

    assert (
        eper_list[1].mask == np.array([[False, False], [False, False], [False, False]])
    ).all()


def test__binned_region_1d_from():

    extract = ac.Extract2DSerialEPER(region_list=[(1, 3, 0, 3)])

    region_1d_list = extract.binned_region_1d_from(pixels=(0, 1))

    assert region_1d_list == None

    region_1d_list = extract.binned_region_1d_from(pixels=(-1, 1))

    assert region_1d_list == (0, 1)

    region_1d_list = extract.binned_region_1d_from(pixels=(-7, 18))

    assert region_1d_list == (0, 7)

    region_1d_list = extract.binned_region_1d_from(pixels=(-3, -1))

    assert region_1d_list == (0, 2)


def test__array_2d_from():

    extract = ac.Extract2DSerialEPER(
        region_list=[(0, 4, 0, 2)], serial_overscan=(0, 4, 2, 3)
    )

    array = ac.Array2D.no_mask(
        values=[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        pixel_scales=1.0,
    )

    new_array = extract.array_2d_from(array=array)

    assert (
        new_array
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

    new_array = extract.array_2d_from(array=array)

    assert (
        new_array
        == np.array(
            [
                [0.0, 0.0, 2.0, 0.5],
                [0.0, 0.0, 5.0, 0.5],
                [0.0, 0.0, 8.0, 0.5],
                [0.0, 0.0, 11.0, 0.5],
            ]
        )
    ).all()
