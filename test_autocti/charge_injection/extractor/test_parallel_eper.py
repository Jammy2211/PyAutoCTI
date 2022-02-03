import numpy as np

import autocti as ac


def test__array_2d_list_from(parallel_array, parallel_masked_array):
    extractor = ac.Extractor2DParallelEPER(region_list=[(1, 3, 0, 3)])

    trails_list = extractor.array_2d_list_from(array=parallel_array, rows=(0, 1))
    assert (trails_list == np.array([[3.0, 3.0, 3.0]])).all()

    trails_list = extractor.array_2d_list_from(array=parallel_array, rows=(2, 3))
    assert (trails_list == np.array([[5.0, 5.0, 5.0]])).all()

    trails_list = extractor.array_2d_list_from(array=parallel_array, rows=(0, 2))
    assert (trails_list == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])).all()

    trails_list = extractor.array_2d_list_from(array=parallel_array, rows=(1, 3))
    assert (trails_list == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])).all()

    trails_list = extractor.array_2d_list_from(array=parallel_array, rows=(1, 4))
    assert (
        trails_list == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
    ).all()

    extractor = ac.Extractor2DParallelEPER(region_list=[(1, 3, 0, 3), (4, 6, 0, 3)])

    trails_list = extractor.array_2d_list_from(array=parallel_array, rows=(0, 1))
    assert (trails_list[0] == np.array([[3.0, 3.0, 3.0]])).all()
    assert (trails_list[1] == np.array([[6.0, 6.0, 6.0]])).all()

    trails_list = extractor.array_2d_list_from(array=parallel_array, rows=(0, 2))
    assert (trails_list[0] == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])).all()
    assert (trails_list[1] == np.array([[6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])).all()

    trails_list = extractor.array_2d_list_from(array=parallel_array, rows=(1, 4))
    assert (
        trails_list[0] == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
    ).all()
    assert (
        trails_list[1] == np.array([[7.0, 7.0, 7.0], [8.0, 8.0, 8.0], [9.0, 9.0, 9.0]])
    ).all()

    trails_list = extractor.array_2d_list_from(array=parallel_masked_array, rows=(0, 2))

    assert (
        trails_list[0].mask == np.array([[False, False, True], [False, False, False]])
    ).all()

    assert (
        trails_list[1].mask == np.array([[False, False, False], [True, False, False]])
    ).all()


def test__stacked_array_2d_from(parallel_array, parallel_masked_array):
    extractor = ac.Extractor2DParallelEPER(region_list=[(1, 3, 0, 3), (4, 6, 0, 3)])

    stacked_trails = extractor.stacked_array_2d_from(array=parallel_array, rows=(0, 2))

    assert (stacked_trails == np.array([[4.5, 4.5, 4.5], [5.5, 5.5, 5.5]])).all()

    stacked_trails = extractor.stacked_array_2d_from(
        array=parallel_masked_array, rows=(0, 2)
    )

    assert (stacked_trails == np.array([[4.5, 4.5, 6.0], [4.0, 5.5, 5.5]])).all()
    assert (
        stacked_trails.mask == np.array([[False, False, False], [False, False, False]])
    ).all()


def test__binned_array_1d_from(parallel_array, parallel_masked_array):
    extractor = ac.Extractor2DParallelEPER(region_list=[(1, 3, 0, 3), (4, 6, 0, 3)])

    trails_line = extractor.binned_array_1d_from(array=parallel_array, rows=(0, 2))

    assert (trails_line == np.array([4.5, 5.5])).all()

    trails_line = extractor.binned_array_1d_from(
        array=parallel_masked_array, rows=(0, 2)
    )

    assert (trails_line == np.array([5.0, 5.0])).all()
