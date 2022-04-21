import numpy as np

import autocti as ac


def test__region_list_from__array_2d_list_from(serial_array, serial_masked_array):
    extract = ac.Extract2DSerialFPR(region_list=[(0, 3, 1, 4)])

    fpr = extract.array_2d_list_from(array=serial_array, pixels=(0, 1))

    assert (fpr == np.array([[1.0], [1.0], [1.0]])).all()

    fpr = extract.array_2d_list_from(array=serial_array, pixels=(1, 2))

    assert (fpr == np.array([[2.0], [2.0], [2.0]])).all()

    fpr = extract.array_2d_list_from(array=serial_array, pixels=(2, 3))

    assert (fpr == np.array([[3.0], [3.0], [3.0]])).all()

    extract = ac.Extract2DSerialFPR(region_list=[(0, 3, 1, 5)])

    fpr = extract.array_2d_list_from(array=serial_array, pixels=(0, 2))

    assert (fpr == np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])).all()

    fpr = extract.array_2d_list_from(array=serial_array, pixels=(1, 4))

    assert (fpr == np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])).all()

    fpr = extract.array_2d_list_from(array=serial_array, pixels=(-1, 1))

    assert (fpr == np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])).all()

    extract = ac.Extract2DSerialFPR(region_list=[(0, 3, 1, 4), (0, 3, 5, 8)])

    fpr_list = extract.array_2d_list_from(array=serial_array, pixels=(0, 1))

    assert (fpr_list[0] == np.array([[1.0], [1.0], [1.0]])).all()
    assert (fpr_list[1] == np.array([[5.0], [5.0], [5.0]])).all()

    fpr_list = extract.array_2d_list_from(array=serial_array, pixels=(1, 2))

    assert (fpr_list[0] == np.array([[2.0], [2.0], [2.0]])).all()
    assert (fpr_list[1] == np.array([[6.0], [6.0], [6.0]])).all()

    fpr_list = extract.array_2d_list_from(array=serial_array, pixels=(2, 3))

    assert (fpr_list[0] == np.array([[3.0], [3.0], [3.0]])).all()
    assert (fpr_list[1] == np.array([[7.0], [7.0], [7.0]])).all()

    fpr_list = extract.array_2d_list_from(array=serial_array, pixels=(0, 3))

    assert (
        fpr_list[0] == np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    ).all()

    assert (
        fpr_list[1] == np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
    ).all()

    fpr_list = extract.array_2d_list_from(array=serial_masked_array, pixels=(0, 3))

    assert (
        (fpr_list[0].mask)
        == np.array(
            [[False, False, False], [False, True, False], [False, False, False]]
        )
    ).all()

    assert (
        fpr_list[1].mask
        == np.array([[True, False, False], [False, True, False], [False, False, True]])
    ).all()
