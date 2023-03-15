import numpy as np

import autocti as ac


def test__region_list_from(serial_array, serial_masked_array):

    extract = ac.Extract2DSerialOverscanNoEPER(
        shape_2d=serial_array.shape_native,
        region_list=[(0, 1, 0, 8)],
        serial_overscan=(0, 3, 8, 10),
    )

    array_2d_list = extract.array_2d_list_from(array=serial_array, pixels=(0, 1))

    assert (array_2d_list[0] == np.array([[8.0], [8.0]])).all()

    array_2d_list = extract.array_2d_list_from(array=serial_array, pixels=(0, 2))

    assert (array_2d_list[0] == np.array([[8.0, 9.0], [8.0, 9.0]])).all()

    serial_array = ac.Array2D.no_mask(
        values=[
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 11.0, 21.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 12.0, 22.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 13.0, 23.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 14.0, 24.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 15.0, 25.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 16.0, 26.0],
        ],
        pixel_scales=1.0,
    )

    extract = ac.Extract2DSerialOverscanNoEPER(
        shape_2d=serial_array.shape_native,
        region_list=[(0, 1, 0, 7), (2, 4, 0, 7)],
        serial_overscan=(0, 5, 7, 10),
    )

    array_2d_list = extract.array_2d_list_from(array=serial_array, pixels=(0, 2))

    assert (array_2d_list[0] == np.array([[2.0, 12.0]])).all()
    assert (array_2d_list[1] == np.array([[5.0, 15.0], [6.0, 16.0]])).all()



def test__region_list_from__pixels_from_end(serial_array, serial_masked_array):

    extract = ac.Extract2DSerialOverscanNoEPER(
        shape_2d=serial_array.shape_native,
        region_list=[(0, 1, 0, 8)],
        serial_overscan=(0, 3, 8, 10),
    )

    array_2d_list = extract.array_2d_list_from(array=serial_array, pixels_from_end=2)

    assert (array_2d_list[0] == np.array([[8.0, 9.0], [8.0, 9.0]])).all()

    serial_array = ac.Array2D.no_mask(
        values=[
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 11.0, 21.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 12.0, 22.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 13.0, 23.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 14.0, 24.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 15.0, 25.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 16.0, 26.0],
        ],
        pixel_scales=1.0,
    )

    extract = ac.Extract2DSerialOverscanNoEPER(
        shape_2d=serial_array.shape_native,
        region_list=[(0, 1, 0, 7), (2, 4, 0, 7)],
        serial_overscan=(0, 5, 7, 10),
    )

    array_2d_list = extract.array_2d_list_from(array=serial_array, pixels_from_end=2)

    assert (array_2d_list[0] == np.array([[12.0, 22.0]])).all()
    assert (array_2d_list[1] == np.array([[15.0, 25.0], [16.0, 26.0]])).all()