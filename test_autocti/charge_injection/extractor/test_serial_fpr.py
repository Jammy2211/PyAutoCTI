import numpy as np

import autocti as ac


def test__array_2d_list_from(serial_array, serial_masked_array):
    extractor = ac.Extractor2DSerialFPR(region_list=[(0, 3, 1, 4)])

    front_edge = extractor.array_2d_list_from(array=serial_array, columns=(0, 1))

    assert (front_edge == np.array([[1.0], [1.0], [1.0]])).all()

    front_edge = extractor.array_2d_list_from(array=serial_array, columns=(1, 2))

    assert (front_edge == np.array([[2.0], [2.0], [2.0]])).all()

    front_edge = extractor.array_2d_list_from(array=serial_array, columns=(2, 3))

    assert (front_edge == np.array([[3.0], [3.0], [3.0]])).all()

    extractor = ac.Extractor2DSerialFPR(region_list=[(0, 3, 1, 5)])

    front_edge = extractor.array_2d_list_from(array=serial_array, columns=(0, 2))

    assert (front_edge == np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])).all()

    front_edge = extractor.array_2d_list_from(array=serial_array, columns=(1, 4))

    assert (
        front_edge == np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
    ).all()

    extractor = ac.Extractor2DSerialFPR(region_list=[(0, 3, 1, 4), (0, 3, 5, 8)])

    front_edge_list = extractor.array_2d_list_from(array=serial_array, columns=(0, 1))

    assert (front_edge_list[0] == np.array([[1.0], [1.0], [1.0]])).all()
    assert (front_edge_list[1] == np.array([[5.0], [5.0], [5.0]])).all()

    front_edge_list = extractor.array_2d_list_from(array=serial_array, columns=(1, 2))

    assert (front_edge_list[0] == np.array([[2.0], [2.0], [2.0]])).all()
    assert (front_edge_list[1] == np.array([[6.0], [6.0], [6.0]])).all()

    front_edge_list = extractor.array_2d_list_from(array=serial_array, columns=(2, 3))

    assert (front_edge_list[0] == np.array([[3.0], [3.0], [3.0]])).all()
    assert (front_edge_list[1] == np.array([[7.0], [7.0], [7.0]])).all()

    front_edge_list = extractor.array_2d_list_from(array=serial_array, columns=(0, 3))

    assert (
        front_edge_list[0]
        == np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    ).all()

    assert (
        front_edge_list[1]
        == np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
    ).all()

    front_edge_list = extractor.array_2d_list_from(
        array=serial_masked_array, columns=(0, 3)
    )

    assert (
        (front_edge_list[0].mask)
        == np.array(
            [[False, False, False], [False, True, False], [False, False, False]]
        )
    ).all()

    assert (
        front_edge_list[1].mask
        == np.array([[True, False, False], [False, True, False], [False, False, True]])
    ).all()


def test__stacked_array_2d_from(serial_array, serial_masked_array):
    extractor = ac.Extractor2DSerialFPR(region_list=[(0, 3, 1, 4), (0, 3, 5, 8)])

    stacked_front_edge_list = extractor.stacked_array_2d_from(
        array=serial_array, columns=(0, 3)
    )

    # [[1.0, 2.0, 3.0],
    #  [1.0, 2.0, 3.0],
    #  [1.0, 2.0, 3.0]]

    # [[5.0, 6.0, 7.0],
    #  [5.0, 6.0, 7.0],
    #  [5.0, 6.0, 7.0]]

    assert (
        stacked_front_edge_list
        == np.array([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0], [3.0, 4.0, 5.0]])
    ).all()

    stacked_front_edge_list = extractor.stacked_array_2d_from(
        array=serial_masked_array, columns=(0, 3)
    )

    assert (
        stacked_front_edge_list
        == np.array([[1.0, 4.0, 5.0], [3.0, 2.0, 5.0], [3.0, 4.0, 3.0]])
    ).all()
    assert (
        stacked_front_edge_list.mask
        == np.array(
            [[False, False, False], [False, True, False], [False, False, False]]
        )
    ).all()


def test__binned_array_1d_from(serial_array, serial_masked_array):
    extractor = ac.Extractor2DSerialFPR(region_list=[(0, 3, 1, 4), (0, 3, 5, 8)])

    front_edge_line = extractor.binned_array_1d_from(array=serial_array, columns=(0, 3))

    assert (front_edge_line == np.array([3.0, 4.0, 5.0])).all()

    front_edge_line = extractor.binned_array_1d_from(
        array=serial_masked_array, columns=(0, 3)
    )

    assert (front_edge_line == np.array([7.0 / 3.0, 4.0, 13.0 / 3.0])).all()
