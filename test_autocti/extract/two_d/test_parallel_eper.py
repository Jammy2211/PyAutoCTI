import numpy as np

import autocti as ac


def test__region_list_from__via_array_2d_list_from(
    parallel_array, parallel_masked_array
):
    extract = ac.Extract2DParallelEPER(region_list=[(1, 3, 0, 3)])

    eper_list = extract.array_2d_list_from(array=parallel_array, pixels=(0, 1))
    assert (eper_list == np.array([[3.0, 3.0, 3.0]])).all()

    eper_list = extract.array_2d_list_from(array=parallel_array, pixels=(2, 3))
    assert (eper_list == np.array([[5.0, 5.0, 5.0]])).all()

    eper_list = extract.array_2d_list_from(array=parallel_array, pixels=(0, 2))
    assert (eper_list == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])).all()

    eper_list = extract.array_2d_list_from(array=parallel_array, pixels=(-1, 1))
    assert (eper_list == np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])).all()

    eper_list = extract.array_2d_list_from(array=parallel_array, pixels=(1, 3))
    assert (eper_list == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])).all()

    eper_list = extract.array_2d_list_from(array=parallel_array, pixels=(1, 4))
    assert (
        eper_list == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
    ).all()

    extract = ac.Extract2DParallelEPER(region_list=[(1, 3, 0, 3), (4, 6, 0, 3)])

    eper_list = extract.array_2d_list_from(array=parallel_array, pixels=(0, 1))
    assert (eper_list[0] == np.array([[3.0, 3.0, 3.0]])).all()
    assert (eper_list[1] == np.array([[6.0, 6.0, 6.0]])).all()

    eper_list = extract.array_2d_list_from(array=parallel_array, pixels=(0, 2))
    assert (eper_list[0] == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])).all()
    assert (eper_list[1] == np.array([[6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])).all()

    eper_list = extract.array_2d_list_from(array=parallel_array, pixels=(1, 4))
    assert (
        eper_list[0] == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
    ).all()
    assert (
        eper_list[1] == np.array([[7.0, 7.0, 7.0], [8.0, 8.0, 8.0], [9.0, 9.0, 9.0]])
    ).all()

    eper_list = extract.array_2d_list_from(array=parallel_masked_array, pixels=(0, 2))

    assert (
        eper_list[0].mask == np.array([[False, False, True], [False, False, False]])
    ).all()

    assert (
        eper_list[1].mask == np.array([[False, False, False], [True, False, False]])
    ).all()


def test__binned_region_1d_from():

    extract = ac.Extract2DParallelEPER(region_list=[(1, 3, 0, 3)])

    region_1d_list = extract.binned_region_1d_from(pixels=(0, 1))

    assert region_1d_list == None

    region_1d_list = extract.binned_region_1d_from(pixels=(-1, 1))

    assert region_1d_list == (0, 1)

    region_1d_list = extract.binned_region_1d_from(pixels=(-7, 18))

    assert region_1d_list == (0, 7)

    region_1d_list = extract.binned_region_1d_from(pixels=(-3, -1))

    assert region_1d_list == (0, 2)


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
