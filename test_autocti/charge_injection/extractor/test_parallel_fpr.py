import numpy as np

import autocti as ac


def test__array_2d_list_from(parallel_array, parallel_masked_array):
    extractor = ac.Extractor2DParallelFPR(region_list=[(1, 4, 0, 3)])

    front_edge_list = extractor.array_2d_list_from(array=parallel_array, pixels=(0, 1))
    assert (front_edge_list[0] == np.array([[1.0, 1.0, 1.0]])).all()

    front_edge_list = extractor.array_2d_list_from(array=parallel_array, pixels=(2, 3))
    assert (front_edge_list[0] == np.array([[3.0, 3.0, 3.0]])).all()

    extractor = ac.Extractor2DParallelFPR(region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

    front_edge_list = extractor.array_2d_list_from(array=parallel_array, pixels=(0, 1))
    assert (front_edge_list[0] == np.array([[1.0, 1.0, 1.0]])).all()
    assert (front_edge_list[1] == np.array([[5.0, 5.0, 5.0]])).all()

    front_edge_list = extractor.array_2d_list_from(array=parallel_array, pixels=(2, 3))
    assert (front_edge_list[0] == np.array([[3.0, 3.0, 3.0]])).all()
    assert (front_edge_list[1] == np.array([[7.0, 7.0, 7.0]])).all()

    front_edge_list = extractor.array_2d_list_from(array=parallel_array, pixels=(0, 3))
    assert (
        front_edge_list[0]
        == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    ).all()
    assert (
        front_edge_list[1]
        == np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])
    ).all()

    front_edge_list = extractor.array_2d_list_from(
        array=parallel_masked_array, pixels=(0, 3)
    )

    assert (
        front_edge_list[0].mask
        == np.array([[False, False, False], [False, True, False], [False, False, True]])
    ).all()

    assert (
        front_edge_list[1].mask
        == np.array(
            [[False, False, False], [False, False, False], [True, False, False]]
        )
    ).all()


def test__stacked_array_2d_from(parallel_array, parallel_masked_array):
    extractor = ac.Extractor2DParallelFPR(region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

    stacked_front_edge_list = extractor.stacked_array_2d_from(
        array=parallel_array, pixels=(0, 3)
    )

    assert (
        stacked_front_edge_list
        == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])
    ).all()

    extractor = ac.Extractor2DParallelFPR(region_list=[(1, 3, 0, 3), (5, 8, 0, 3)])

    stacked_front_edge_list = extractor.stacked_array_2d_from(
        array=parallel_array, pixels=(0, 2)
    )

    assert (
        stacked_front_edge_list == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
    ).all()

    stacked_front_edge_list = extractor.stacked_array_2d_from(
        array=parallel_masked_array, pixels=(0, 3)
    )

    assert (
        stacked_front_edge_list
        == np.ma.array([[3.0, 3.0, 3.0], [4.0, 6.0, 4.0], [3.0, 5.0, 7.0]])
    ).all()
    assert (
        stacked_front_edge_list.mask
        == np.ma.array(
            [[False, False, False], [False, False, False], [False, False, False]]
        )
    ).all()


def test__binned_array_1d_from(parallel_array, parallel_masked_array):
    extractor = ac.Extractor2DParallelFPR(region_list=[(1, 3, 0, 3), (5, 8, 0, 3)])

    front_edge_line = extractor.binned_array_1d_from(
        array=parallel_array, pixels=(0, 3)
    )

    assert (front_edge_line == np.array([3.0, 4.0, 5.0])).all()

    extractor = ac.Extractor2DParallelFPR(region_list=[(1, 3, 0, 3), (5, 8, 0, 3)])

    front_edge_line = extractor.binned_array_1d_from(
        array=parallel_array, pixels=(0, 2)
    )

    assert (front_edge_line == np.array([3.0, 4.0])).all()

    front_edge_line = extractor.binned_array_1d_from(
        array=parallel_masked_array, pixels=(0, 3)
    )

    assert (front_edge_line == np.array([9.0 / 3.0, 14.0 / 3.0, 5.0])).all()
