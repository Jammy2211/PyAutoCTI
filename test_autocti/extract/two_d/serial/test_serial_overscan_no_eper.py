import numpy as np

import autocti as ac


def test__region_list_from(serial_array, serial_masked_array):

    extract = ac.Extract2DSerialOverscanNoEPER(
        region_list=[(0, 8, 0, 1)], serial_overscan=(0, 3, 8, 10)
    )

    array_2d_list = extract.array_2d_list_from(array=serial_array, pixels=(0, 1))

    assert (array_2d_list[0] == np.array([[8.0], [8.0]])).all()

    array_2d_list = extract.array_2d_list_from(array=serial_array, pixels=(0, 2))

    assert (array_2d_list[0] == np.array([[8.0, 9.0], [8.0, 9.0]])).all()
