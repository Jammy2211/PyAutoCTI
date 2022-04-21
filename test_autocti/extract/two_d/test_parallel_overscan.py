import numpy as np

import autocti as ac


def test_region_list_from__via_array_2d_list_from(
    parallel_array, parallel_masked_array
):

    extract = ac.Extract2DParallelOverscan(parallel_overscan=(1, 4, 0, 3))

    overscan_list = extract.array_2d_list_from(array=parallel_array, pixels=(0, 1))
    assert (overscan_list[0] == np.array([[1.0, 1.0, 1.0]])).all()

    overscan_list = extract.array_2d_list_from(array=parallel_array, pixels=(2, 3))
    assert (overscan_list[0] == np.array([[3.0, 3.0, 3.0]])).all()

    overscan_list = extract.array_2d_list_from(array=parallel_array, pixels=(0, 3))
    assert (
        overscan_list[0]
        == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    ).all()

    overscan_list = extract.array_2d_list_from(array=parallel_array, pixels=(-1, 1))
    assert (overscan_list[0] == np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])).all()

    overscan_list = extract.array_2d_list_from(
        array=parallel_masked_array, pixels=(0, 3)
    )

    assert (
        overscan_list[0].mask
        == np.array([[False, False, False], [False, True, False], [False, False, True]])
    ).all()


def test__binned_region_1d_from():

    extract = ac.Extract2DParallelOverscan(region_list=[(1, 3, 0, 3)])

    region_1d_list = extract.binned_region_1d_from(pixels=(0, 1))

    assert region_1d_list == None

    region_1d_list = extract.binned_region_1d_from(pixels=(-1, 1))

    assert region_1d_list == (0, 1)

    region_1d_list = extract.binned_region_1d_from(pixels=(-7, 18))

    assert region_1d_list == (0, 7)

    region_1d_list = extract.binned_region_1d_from(pixels=(-3, -1))

    assert region_1d_list == (0, 2)
