import numpy as np

import autocti as ac


def test__region_list_from__via_array_2d_list_from(
    parallel_array, parallel_masked_array
):
    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3)])

    fpr_list = extract.array_2d_list_from(array=parallel_array, pixels=(0, 1))
    assert (fpr_list[0] == np.array([[1.0, 1.0, 1.0]])).all()

    fpr_list = extract.array_2d_list_from(array=parallel_array, pixels=(2, 3))
    assert (fpr_list[0] == np.array([[3.0, 3.0, 3.0]])).all()

    fpr_list = extract.array_2d_list_from(array=parallel_array, pixels=(-1, 1))
    assert (fpr_list[0] == np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])).all()

    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

    fpr_list = extract.array_2d_list_from(array=parallel_array, pixels=(0, 1))
    assert (fpr_list[0] == np.array([[1.0, 1.0, 1.0]])).all()
    assert (fpr_list[1] == np.array([[5.0, 5.0, 5.0]])).all()

    fpr_list = extract.array_2d_list_from(array=parallel_array, pixels=(2, 3))
    assert (fpr_list[0] == np.array([[3.0, 3.0, 3.0]])).all()
    assert (fpr_list[1] == np.array([[7.0, 7.0, 7.0]])).all()

    fpr_list = extract.array_2d_list_from(array=parallel_array, pixels=(0, 3))
    assert (
        fpr_list[0] == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    ).all()
    assert (
        fpr_list[1] == np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])
    ).all()

    fpr_list = extract.array_2d_list_from(array=parallel_masked_array, pixels=(0, 3))

    assert (
        fpr_list[0].mask
        == np.array([[False, False, False], [False, True, False], [False, False, True]])
    ).all()

    assert (
        fpr_list[1].mask
        == np.array(
            [[False, False, False], [False, False, False], [True, False, False]]
        )
    ).all()


def test__region_list_from__via_array_2d_list_from__pixels_from_end(
    parallel_array, parallel_masked_array
):
    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3)])

    fpr_list = extract.array_2d_list_from(array=parallel_array, pixels_from_end=1)
    assert (fpr_list[0] == np.array([[3.0, 3.0, 3.0]])).all()

    fpr_list = extract.array_2d_list_from(array=parallel_array, pixels_from_end=2)
    assert (fpr_list[0] == np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])).all()

    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

    fpr_list = extract.array_2d_list_from(array=parallel_array, pixels_from_end=3)
    assert (
        fpr_list[0] == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    ).all()
    assert (
        fpr_list[1] == np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])
    ).all()

    fpr_list = extract.array_2d_list_from(
        array=parallel_masked_array, pixels_from_end=3
    )

    assert (
        fpr_list[0].mask
        == np.array([[False, False, False], [False, True, False], [False, False, True]])
    ).all()

    assert (
        fpr_list[1].mask
        == np.array(
            [[False, False, False], [False, False, False], [True, False, False]]
        )
    ).all()


def test__binned_region_1d_from():

    extract = ac.Extract2DParallelFPR(region_list=[(1, 3, 0, 3)])

    region_1d_list = extract.binned_region_1d_from(pixels=(0, 1))

    assert region_1d_list == (0, 1)

    region_1d_list = extract.binned_region_1d_from(pixels=(-1, 1))

    assert region_1d_list == (1, 2)

    region_1d_list = extract.binned_region_1d_from(pixels=(-7, 18))

    assert region_1d_list == (7, 25)

    region_1d_list = extract.binned_region_1d_from(pixels=(-3, -1))

    assert region_1d_list == None
