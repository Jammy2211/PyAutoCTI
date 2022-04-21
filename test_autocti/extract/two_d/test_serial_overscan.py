import numpy as np

import autocti as ac


def test__array_2d_list_from(serial_array, serial_masked_array):
    extract = ac.Extract2DSerialOverscan(serial_overscan=(0, 3, 1, 4))

    overscan_list = extract.array_2d_list_from(array=serial_array, pixels=(0, 1))

    assert (overscan_list[0] == np.array([[1.0], [1.0], [1.0]])).all()

    overscan_list = extract.array_2d_list_from(array=serial_array, pixels=(1, 2))

    assert (overscan_list[0] == np.array([[2.0], [2.0], [2.0]])).all()

    overscan_list = extract.array_2d_list_from(array=serial_array, pixels=(2, 3))

    assert (overscan_list[0] == np.array([[3.0], [3.0], [3.0]])).all()

    extract = ac.Extract2DSerialOverscan(serial_overscan=(0, 3, 1, 5))

    overscan_list = extract.array_2d_list_from(array=serial_array, pixels=(0, 2))

    assert (overscan_list[0] == np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])).all()

    overscan_list = extract.array_2d_list_from(array=serial_array, pixels=(1, 4))

    assert (
        overscan_list[0]
        == np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
    ).all()

    overscan_list = extract.array_2d_list_from(array=serial_masked_array, pixels=(0, 3))

    assert (
        (overscan_list[0].mask)
        == np.array(
            [[False, False, False], [False, True, False], [False, False, False]]
        )
    ).all()


def test__stacked_array_2d_from(serial_array, serial_masked_array):
    extract = ac.Extract2DSerialOverscan(serial_overscan=(0, 3, 1, 4))

    stacked_overscan = extract.stacked_array_2d_from(array=serial_array, pixels=(0, 3))

    # [[1.0, 2.0, 3.0],
    #  [1.0, 2.0, 3.0],
    #  [1.0, 2.0, 3.0]]

    # [[5.0, 6.0, 7.0],
    #  [5.0, 6.0, 7.0],
    #  [5.0, 6.0, 7.0]]

    assert (
        stacked_overscan
        == np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    ).all()

    stacked_overscan = extract.stacked_array_2d_from(
        array=serial_masked_array, pixels=(0, 3)
    )

    assert (
        stacked_overscan
        == np.array([[1.0, 2.0, 3.0], [1.0, 0.0, 3.0], [1.0, 2.0, 3.0]])
    ).all()
    assert (
        stacked_overscan.mask
        == np.array(
            [[False, False, False], [False, True, False], [False, False, False]]
        )
    ).all()


def test__binned_array_1d_from(serial_array, serial_masked_array):
    extract = ac.Extract2DSerialOverscan(serial_overscan=(0, 3, 1, 4))

    binned_overscan = extract.binned_array_1d_from(array=serial_array, pixels=(0, 3))

    assert (binned_overscan == np.array([1.0, 2.0, 3.0])).all()

    binned_overscan = extract.binned_array_1d_from(
        array=serial_masked_array, pixels=(0, 3)
    )

    assert (binned_overscan == np.array([1.0, 2.0, 3.0])).all()
