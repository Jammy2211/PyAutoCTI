import numpy as np

import autocti as ac


def test_region_list_from__via_array_2d_list_from(
    parallel_array, parallel_masked_array
):

    extract = ac.Extract2DParallelOverscan(parallel_overscan=(1, 4, 0, 3))

    array_2d_list = extract.array_2d_list_from(array=parallel_array, pixels=(0, 1))
    assert (array_2d_list[0] == np.array([[1.0, 1.0, 1.0]])).all()

    array_2d_list = extract.array_2d_list_from(array=parallel_array, pixels=(2, 3))
    assert (array_2d_list[0] == np.array([[3.0, 3.0, 3.0]])).all()

    array_2d_list = extract.array_2d_list_from(array=parallel_array, pixels=(0, 3))
    assert (
        array_2d_list[0]
        == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    ).all()

    array_2d_list = extract.array_2d_list_from(array=parallel_array, pixels=(-1, 1))
    assert (array_2d_list[0] == np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_masked_array, pixels=(0, 3)
    )

    assert (
        array_2d_list[0].mask
        == np.array([[False, False, False], [False, True, False], [False, False, True]])
    ).all()


def test_region_list_from__via_array_2d_list_from__pixels_from_end(
    parallel_array, parallel_masked_array
):

    extract = ac.Extract2DParallelOverscan(parallel_overscan=(1, 4, 0, 3))

    array_2d_list = extract.array_2d_list_from(array=parallel_array, pixels_from_end=1)
    assert (array_2d_list[0] == np.array([[3.0, 3.0, 3.0]])).all()

    array_2d_list = extract.array_2d_list_from(array=parallel_array, pixels_from_end=3)
    assert (
        array_2d_list[0]
        == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    ).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_masked_array, pixels_from_end=3
    )

    assert (
        array_2d_list[0].mask
        == np.array([[False, False, False], [False, True, False], [False, False, True]])
    ).all()


def test__binned_region_1d_from():

    extract = ac.Extract2DParallelOverscan(region_list=[(1, 3, 0, 3)])

    binned_region_1d = extract.binned_region_1d_from(pixels=(0, 1))

    assert binned_region_1d == None

    binned_region_1d = extract.binned_region_1d_from(pixels=(-1, 1))

    assert binned_region_1d == (0, 1)

    binned_region_1d = extract.binned_region_1d_from(pixels=(-7, 18))

    assert binned_region_1d == (0, 7)

    binned_region_1d = extract.binned_region_1d_from(pixels=(-3, -1))

    assert binned_region_1d == (0, 2)
