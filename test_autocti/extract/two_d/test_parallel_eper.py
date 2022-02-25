import numpy as np

import autocti as ac


def test__array_2d_list_from(parallel_array, parallel_masked_array):
    extract = ac.Extract2DParallelEPER(region_list=[(1, 3, 0, 3)])

    trails_list = extract.array_2d_list_from(array=parallel_array, pixels=(0, 1))
    assert (trails_list == np.array([[3.0, 3.0, 3.0]])).all()

    trails_list = extract.array_2d_list_from(array=parallel_array, pixels=(2, 3))
    assert (trails_list == np.array([[5.0, 5.0, 5.0]])).all()

    trails_list = extract.array_2d_list_from(array=parallel_array, pixels=(0, 2))
    assert (trails_list == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])).all()

    trails_list = extract.array_2d_list_from(array=parallel_array, pixels=(1, 3))
    assert (trails_list == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])).all()

    trails_list = extract.array_2d_list_from(array=parallel_array, pixels=(1, 4))
    assert (
        trails_list == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
    ).all()

    extract = ac.Extract2DParallelEPER(region_list=[(1, 3, 0, 3), (4, 6, 0, 3)])

    trails_list = extract.array_2d_list_from(array=parallel_array, pixels=(0, 1))
    assert (trails_list[0] == np.array([[3.0, 3.0, 3.0]])).all()
    assert (trails_list[1] == np.array([[6.0, 6.0, 6.0]])).all()

    trails_list = extract.array_2d_list_from(array=parallel_array, pixels=(0, 2))
    assert (trails_list[0] == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])).all()
    assert (trails_list[1] == np.array([[6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])).all()

    trails_list = extract.array_2d_list_from(array=parallel_array, pixels=(1, 4))
    assert (
        trails_list[0] == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
    ).all()
    assert (
        trails_list[1] == np.array([[7.0, 7.0, 7.0], [8.0, 8.0, 8.0], [9.0, 9.0, 9.0]])
    ).all()

    trails_list = extract.array_2d_list_from(array=parallel_masked_array, pixels=(0, 2))

    assert (
        trails_list[0].mask == np.array([[False, False, True], [False, False, False]])
    ).all()

    assert (
        trails_list[1].mask == np.array([[False, False, False], [True, False, False]])
    ).all()


def test__stacked_array_2d_from(parallel_array, parallel_masked_array):
    extract = ac.Extract2DParallelEPER(region_list=[(1, 3, 0, 3), (4, 6, 0, 3)])

    stacked_trails = extract.stacked_array_2d_from(array=parallel_array, pixels=(0, 2))

    assert (stacked_trails == np.array([[4.5, 4.5, 4.5], [5.5, 5.5, 5.5]])).all()

    stacked_trails = extract.stacked_array_2d_from(
        array=parallel_masked_array, pixels=(0, 2)
    )

    assert (stacked_trails == np.array([[4.5, 4.5, 6.0], [4.0, 5.5, 5.5]])).all()
    assert (
        stacked_trails.mask == np.array([[False, False, False], [False, False, False]])
    ).all()


def test__binned_array_1d_from(parallel_array, parallel_masked_array):
    extract = ac.Extract2DParallelEPER(region_list=[(1, 3, 0, 3), (4, 6, 0, 3)])

    trails_line = extract.binned_array_1d_from(array=parallel_array, pixels=(0, 2))

    assert (trails_line == np.array([4.5, 5.5])).all()

    trails_line = extract.binned_array_1d_from(
        array=parallel_masked_array, pixels=(0, 2)
    )

    assert (trails_line == np.array([5.0, 5.0])).all()


def test__array_2d_from():

    extract = ac.Extract2DParallelEPER(
        region_list=[(0, 3, 0, 3)],
        serial_prescan=(3, 5, 2, 3),
        serial_overscan=(3, 5, 0, 1),
    )

    array = ac.Array2D.manual(
        array=[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0],
            [12.0, 13.0, 14.0],
        ],
        pixel_scales=1.0,
    )

    array_extracted = extract.array_2d_from(array=array)

    assert (
        array_extracted
        == np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 13.0, 0.0],
            ]
        )
    ).all()

    extract = ac.Extract2DParallelEPER(
        region_list=[(0, 1, 0, 3), (3, 4, 0, 3)],
        serial_prescan=(1, 2, 0, 3),
        serial_overscan=(0, 1, 0, 1),
    )

    array_extracted = extract.array_2d_from(array=array)

    assert (
        array_extracted.native
        == np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [6.0, 7.0, 8.0],
                [0.0, 0.0, 0.0],
                [12.0, 13.0, 14.0],
            ]
        )
    ).all()
