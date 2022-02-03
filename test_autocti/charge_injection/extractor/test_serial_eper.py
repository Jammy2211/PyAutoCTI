import numpy as np

import autocti as ac


def test__array_2d_list_from(serial_array, serial_masked_array):

    extractor = ac.Extractor2DSerialEPER(region_list=[(0, 3, 1, 4)])

    trails_list = extractor.array_2d_list_from(array=serial_array, columns=(0, 1))

    assert (trails_list == np.array([[4.0], [4.0], [4.0]])).all()

    trails_list = extractor.array_2d_list_from(array=serial_array, columns=(1, 2))

    assert (trails_list == np.array([[5.0], [5.0], [5.0]])).all()

    trails_list = extractor.array_2d_list_from(array=serial_array, columns=(2, 3))

    assert (trails_list == np.array([[6.0], [6.0], [6.0]])).all()

    trails_list = extractor.array_2d_list_from(array=serial_array, columns=(0, 2))

    assert (trails_list == np.array([[4.0, 5.0], [4.0, 5.0], [4.0, 5.0]])).all()

    trails_list = extractor.array_2d_list_from(array=serial_array, columns=(1, 4))

    assert (
        trails_list == np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
    ).all()

    extractor = ac.Extractor2DSerialEPER(region_list=[(0, 3, 1, 4), (0, 3, 5, 8)])

    trails_list = extractor.array_2d_list_from(array=serial_array, columns=(0, 1))

    assert (trails_list[0] == np.array([[4.0], [4.0], [4.0]])).all()
    assert (trails_list[1] == np.array([[8.0], [8.0], [8.0]])).all()

    trails_list = extractor.array_2d_list_from(array=serial_array, columns=(1, 2))

    assert (trails_list[0] == np.array([[5.0], [5.0], [5.0]])).all()
    assert (trails_list[1] == np.array([[9.0], [9.0], [9.0]])).all()

    trails_list = extractor.array_2d_list_from(array=serial_array, columns=(2, 3))

    assert (trails_list[0] == np.array([[6.0], [6.0], [6.0]])).all()
    assert (trails_list[1] == np.array([[10.0], [10.0], [10.0]])).all()

    trails_list = extractor.array_2d_list_from(array=serial_array, columns=(0, 3))

    assert (
        trails_list[0] == np.array([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]])
    ).all()

    assert (trails_list[1] == np.array([[8.0, 9.0], [8.0, 9.0], [8.0, 9.0]])).all()

    trails_list = extractor.array_2d_list_from(
        array=serial_masked_array, columns=(0, 3)
    )

    assert (
        trails_list[0].mask
        == np.array([[False, True, False], [False, False, True], [False, False, False]])
    ).all()

    assert (
        trails_list[1].mask
        == np.array([[False, False], [False, False], [False, False]])
    ).all()


def test__stacked_array_2d_from(serial_array, serial_masked_array):
    extractor = ac.Extractor2DSerialEPER(region_list=[(0, 3, 1, 4), (0, 3, 5, 8)])

    stacked_trails = extractor.stacked_array_2d_from(array=serial_array, columns=(0, 2))

    assert (stacked_trails == np.array([[6.0, 7.0], [6.0, 7.0], [6.0, 7.0]])).all()

    stacked_trails = extractor.stacked_array_2d_from(
        array=serial_masked_array, columns=(0, 2)
    )

    assert (stacked_trails == np.array([[6.0, 9.0], [6.0, 7.0], [6.0, 7.0]])).all()
    assert (
        stacked_trails.mask
        == np.array([[False, False], [False, False], [False, False]])
    ).all()


def test__binned_array_1d_from(serial_array, serial_masked_array):
    extractor = ac.Extractor2DSerialEPER(region_list=[(0, 3, 1, 4), (0, 3, 5, 8)])

    trails_line = extractor.binned_array_1d_from(array=serial_array, columns=(0, 2))

    assert (trails_line == np.array([6.0, 7.0])).all()

    extractor = ac.Extractor2DSerialEPER(region_list=[(0, 3, 1, 4), (0, 3, 5, 8)])

    trails_line = extractor.binned_array_1d_from(
        array=serial_masked_array, columns=(0, 2)
    )

    assert (trails_line == np.array([6.0, 23.0 / 3.0])).all()
