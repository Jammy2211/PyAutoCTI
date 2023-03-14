import numpy as np

import autocti as ac


def test_region_list_from__via_array_2d_list_from(
    parallel_array, parallel_masked_array
):

    extract = ac.Extract2DParallelPedestal(
        shape_2d=parallel_array.shape_native,
        parallel_overscan=(8, 10, 0, 2),
        serial_overscan=(0, 8, 2, 3),
    )

    array_2d_list = extract.array_2d_list_from(array=parallel_array, pixels=(0, 1))
    assert (array_2d_list[0] == np.array([[8.0]])).all()

    array_2d_list = extract.array_2d_list_from(array=parallel_array, pixels=(0, 2))
    print(array_2d_list)
    assert (array_2d_list[0] == np.array([[8.0], [9.0]])).all()

    array_2d_list = extract.array_2d_list_from(array=parallel_array, pixels=(-1, 1))
    assert (array_2d_list[0] == np.array([[7.0], [8.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_masked_array, pixels=(-5, -3)
    )

    print(array_2d_list[0].mask)

    assert (array_2d_list[0].mask == np.array([[True], [False]])).all()


def test_region_list_from__via_array_2d_list_from__pixels_from_end(
    parallel_array, parallel_masked_array
):

    extract = ac.Extract2DParallelPedestal(
        shape_2d=parallel_array.shape_native,
        parallel_overscan=(8, 10, 0, 2),
        serial_overscan=(0, 8, 2, 3),
    )

    array_2d_list = extract.array_2d_list_from(array=parallel_array, pixels_from_end=1)
    assert (array_2d_list[0] == np.array([[9.0]])).all()

    array_2d_list = extract.array_2d_list_from(array=parallel_array, pixels_from_end=2)
    assert (array_2d_list[0] == np.array([[8.0], [9.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_masked_array, pixels_from_end=2
    )

    assert (
        array_2d_list[0].mask
        == np.array([[False], [False]])
    ).all()