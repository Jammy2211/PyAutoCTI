import numpy as np

import autocti as ac


def test__array_2d_list_from(serial_array, serial_masked_array):

    extract = ac.Extract2DSerialEPER(region_list=[(0, 3, 1, 4)])

    trails_list = extract.array_2d_list_from(array=serial_array, pixels=(0, 1))

    assert (trails_list == np.array([[4.0], [4.0], [4.0]])).all()

    trails_list = extract.array_2d_list_from(array=serial_array, pixels=(1, 2))

    assert (trails_list == np.array([[5.0], [5.0], [5.0]])).all()

    trails_list = extract.array_2d_list_from(array=serial_array, pixels=(2, 3))

    assert (trails_list == np.array([[6.0], [6.0], [6.0]])).all()

    trails_list = extract.array_2d_list_from(array=serial_array, pixels=(0, 2))

    assert (trails_list == np.array([[4.0, 5.0], [4.0, 5.0], [4.0, 5.0]])).all()

    trails_list = extract.array_2d_list_from(array=serial_array, pixels=(1, 4))

    assert (
        trails_list == np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
    ).all()

    extract = ac.Extract2DSerialEPER(region_list=[(0, 3, 1, 4), (0, 3, 5, 8)])

    trails_list = extract.array_2d_list_from(array=serial_array, pixels=(0, 1))

    assert (trails_list[0] == np.array([[4.0], [4.0], [4.0]])).all()
    assert (trails_list[1] == np.array([[8.0], [8.0], [8.0]])).all()

    trails_list = extract.array_2d_list_from(array=serial_array, pixels=(1, 2))

    assert (trails_list[0] == np.array([[5.0], [5.0], [5.0]])).all()
    assert (trails_list[1] == np.array([[9.0], [9.0], [9.0]])).all()

    trails_list = extract.array_2d_list_from(array=serial_array, pixels=(2, 3))

    assert (trails_list[0] == np.array([[6.0], [6.0], [6.0]])).all()
    assert (trails_list[1] == np.array([[10.0], [10.0], [10.0]])).all()

    trails_list = extract.array_2d_list_from(array=serial_array, pixels=(0, 3))

    assert (
        trails_list[0] == np.array([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]])
    ).all()

    assert (trails_list[1] == np.array([[8.0, 9.0], [8.0, 9.0], [8.0, 9.0]])).all()

    trails_list = extract.array_2d_list_from(array=serial_masked_array, pixels=(0, 3))

    assert (
        trails_list[0].mask
        == np.array([[False, True, False], [False, False, True], [False, False, False]])
    ).all()

    assert (
        trails_list[1].mask
        == np.array([[False, False], [False, False], [False, False]])
    ).all()


def test__stacked_array_2d_from(serial_array, serial_masked_array):
    extract = ac.Extract2DSerialEPER(region_list=[(0, 3, 1, 4), (0, 3, 5, 8)])

    stacked_trails = extract.stacked_array_2d_from(array=serial_array, pixels=(0, 2))

    assert (stacked_trails == np.array([[6.0, 7.0], [6.0, 7.0], [6.0, 7.0]])).all()

    stacked_trails = extract.stacked_array_2d_from(
        array=serial_masked_array, pixels=(0, 2)
    )

    assert (stacked_trails == np.array([[6.0, 9.0], [6.0, 7.0], [6.0, 7.0]])).all()
    assert (
        stacked_trails.mask
        == np.array([[False, False], [False, False], [False, False]])
    ).all()


def test__binned_array_1d_from(serial_array, serial_masked_array):
    extract = ac.Extract2DSerialEPER(region_list=[(0, 3, 1, 4), (0, 3, 5, 8)])

    trails_line = extract.binned_array_1d_from(array=serial_array, pixels=(0, 2))

    assert (trails_line == np.array([6.0, 7.0])).all()

    extract = ac.Extract2DSerialEPER(region_list=[(0, 3, 1, 4), (0, 3, 5, 8)])

    trails_line = extract.binned_array_1d_from(array=serial_masked_array, pixels=(0, 2))

    assert (trails_line == np.array([6.0, 23.0 / 3.0])).all()


def test__array_2d_from():

    extract = ac.Extract2DSerialEPER(
        region_list=[(0, 4, 0, 2)], serial_overscan=(0, 4, 2, 3)
    )

    array = ac.Array2D.manual(
        array=[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
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

    array = ac.Array2D.manual(
        array=[
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
