import numpy as np

import autocti as ac


def test__array_2d_list_from(parallel_array, parallel_masked_array):

    extract = ac.Extract2DParallelOverscan(parallel_overscan=(1, 4, 0, 3))

    overscan_list = extract.array_2d_list_from(array=parallel_array, pixels=(0, 1))
    print(overscan_list)
    assert (overscan_list[0] == np.array([[1.0, 1.0, 1.0]])).all()

    overscan_list = extract.array_2d_list_from(array=parallel_array, pixels=(2, 3))
    assert (overscan_list[0] == np.array([[3.0, 3.0, 3.0]])).all()

    overscan_list = extract.array_2d_list_from(array=parallel_array, pixels=(0, 3))
    assert (
        overscan_list[0]
        == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    ).all()

    overscan_list = extract.array_2d_list_from(
        array=parallel_masked_array, pixels=(0, 3)
    )

    assert (
        overscan_list[0].mask
        == np.array([[False, False, False], [False, True, False], [False, False, True]])
    ).all()


def test__stacked_array_2d_from(parallel_array, parallel_masked_array):
    extract = ac.Extract2DParallelOverscan(parallel_overscan=(1, 4, 0, 3))

    stacked_overscan_list = extract.stacked_array_2d_from(
        array=parallel_array, pixels=(0, 3)
    )

    assert (
        stacked_overscan_list
        == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    ).all()

    extract = ac.Extract2DParallelOverscan(parallel_overscan=(1, 4, 0, 3))

    stacked_overscan_list = extract.stacked_array_2d_from(
        array=parallel_array, pixels=(0, 2)
    )

    assert (stacked_overscan_list == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])).all()

    stacked_overscan = extract.stacked_array_2d_from(
        array=parallel_masked_array, pixels=(0, 3)
    )

    assert (
        stacked_overscan
        == np.ma.array([[1.0, 1.0, 1.0], [2.0, 0.0, 2.0], [3.0, 3.0, 0.0]])
    ).all()
    assert (
        stacked_overscan.mask
        == np.ma.array(
            [[False, False, False], [False, True, False], [False, False, True]]
        )
    ).all()


def test__binned_array_1d_from(parallel_array, parallel_masked_array):
    extract = ac.Extract2DParallelOverscan(parallel_overscan=(1, 4, 0, 3))

    binned_overscan_1d = extract.binned_array_1d_from(
        array=parallel_array, pixels=(0, 3)
    )

    assert (binned_overscan_1d == np.array([1.0, 2.0, 3.0])).all()

    binned_overscan_1d = extract.binned_array_1d_from(
        array=parallel_array, pixels=(0, 2)
    )

    assert (binned_overscan_1d == np.array([1.0, 2.0])).all()

    binned_overscan_1d = extract.binned_array_1d_from(
        array=parallel_masked_array, pixels=(0, 3)
    )

    assert (binned_overscan_1d == np.array([1.0, 2.0, 3.0])).all()
