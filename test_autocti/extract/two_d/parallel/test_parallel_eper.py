import numpy as np
import pytest

import autocti as ac


def test__region_list_from__via_array_2d_list_from(
    parallel_array, parallel_masked_array
):
    extract = ac.Extract2DParallelEPER(region_list=[(1, 3, 0, 3)])

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(0, 1))
    )
    assert (array_2d_list == np.array([[3.0, 3.0, 3.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(2, 3))
    )
    assert (array_2d_list == np.array([[5.0, 5.0, 5.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(0, 2))
    )
    assert (array_2d_list == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(-1, 1))
    )
    assert (array_2d_list == np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(1, 3))
    )
    assert (array_2d_list == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(1, 4))
    )
    assert (
        array_2d_list == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
    ).all()

    extract = ac.Extract2DParallelEPER(region_list=[(1, 3, 0, 3), (4, 6, 0, 3)])

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(0, 1))
    )
    assert (array_2d_list[0] == np.array([[3.0, 3.0, 3.0]])).all()
    assert (array_2d_list[1] == np.array([[6.0, 6.0, 6.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(0, 2))
    )
    assert (array_2d_list[0] == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])).all()
    assert (array_2d_list[1] == np.array([[6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(1, 4))
    )
    assert (
        array_2d_list[0]
        == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
    ).all()
    assert (
        array_2d_list[1]
        == np.array([[7.0, 7.0, 7.0], [8.0, 8.0, 8.0], [9.0, 9.0, 9.0]])
    ).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_masked_array, settings=ac.SettingsExtract(pixels=(0, 2))
    )

    assert (
        array_2d_list[0].mask == np.array([[False, False, True], [False, False, False]])
    ).all()

    assert (
        array_2d_list[1].mask == np.array([[False, False, False], [True, False, False]])
    ).all()


def test__region_list_from__via_array_2d_list_from__pixels_from_end(
    parallel_array, parallel_masked_array
):
    extract = ac.Extract2DParallelEPER(
        region_list=[(1, 3, 0, 3)], shape_2d=parallel_array.shape_native
    )

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels_from_end=1)
    )
    assert (array_2d_list == np.array([[9.0, 9.0, 9.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels_from_end=2)
    )
    assert (array_2d_list == np.array([[8.0, 8.0, 8.0], [9.0, 9.0, 9.0]])).all()

    extract = ac.Extract2DParallelEPER(
        region_list=[(1, 3, 0, 3), (4, 6, 0, 3)], shape_2d=parallel_array.shape_native
    )

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels_from_end=2)
    )

    assert (array_2d_list[0] == np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])).all()
    assert (array_2d_list[1] == np.array([[8.0, 8.0, 8.0], [9.0, 9.0, 9.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_masked_array, settings=ac.SettingsExtract(pixels_from_end=2)
    )

    assert (
        array_2d_list[0].mask == np.array([[False, True, False], [False, False, True]])
    ).all()


def test__region_list_from__via_array_2d_list_from__pixels_from_end_minus_1_special_case__extracts_complete_arrays_of_full_eper_region(
    parallel_array, parallel_masked_array
):
    extract = ac.Extract2DParallelEPER(
        region_list=[(1, 3, 0, 3)], shape_2d=parallel_array.shape_native
    )

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels_from_end=-1)
    )
    assert (
        array_2d_list
        == np.array(
            [
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
                [5.0, 5.0, 5.0],
                [6.0, 6.0, 6.0],
                [7.0, 7.0, 7.0],
                [8.0, 8.0, 8.0],
                [9.0, 9.0, 9.0],
            ]
        )
    ).all()

    extract = ac.Extract2DParallelEPER(
        region_list=[(1, 3, 0, 3), (4, 6, 0, 3)], shape_2d=parallel_array.shape_native
    )

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels_from_end=-1)
    )

    assert (
        array_2d_list[0]
        == np.array(
            [
                [3.0, 3.0, 3.0],
            ]
        )
    ).all()

    assert (
        array_2d_list[1]
        == np.array(
            [
                [6.0, 6.0, 6.0],
                [7.0, 7.0, 7.0],
                [8.0, 8.0, 8.0],
                [9.0, 9.0, 9.0],
            ]
        )
    ).all()


def test__array_2d_from():
    extract = ac.Extract2DParallelEPER(
        region_list=[(0, 3, 0, 3)],
        serial_prescan=(3, 5, 2, 3),
        serial_overscan=(3, 5, 0, 1),
    )

    array = ac.Array2D.no_mask(
        values=[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0],
            [12.0, 13.0, 14.0],
        ],
        pixel_scales=1.0,
    )

    array_2d = extract.array_2d_from(array=array)

    assert (
        array_2d
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

    array_2d = extract.array_2d_from(array=array)

    assert (
        array_2d.native
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


def test__binned_region_1d_from():
    extract = ac.Extract2DParallelEPER(region_list=[(1, 3, 0, 3)])

    binned_region_1d = extract.binned_region_1d_from(
        settings=ac.SettingsExtract(pixels=(0, 1))
    )

    assert binned_region_1d == None

    binned_region_1d = extract.binned_region_1d_from(
        settings=ac.SettingsExtract(pixels=(-1, 1))
    )

    assert binned_region_1d == (0, 1)

    binned_region_1d = extract.binned_region_1d_from(
        settings=ac.SettingsExtract(pixels=(-7, 18))
    )

    assert binned_region_1d == (0, 7)

    binned_region_1d = extract.binned_region_1d_from(
        settings=ac.SettingsExtract(pixels=(-3, -1))
    )

    assert binned_region_1d == (0, 2)


def test__masking_integration_test():
    parallel_array = ac.Array2D.no_mask(
        values=[
            [0.0, 0.0, 0.0, 4.0, 5.0],
            [1.0, 1.0, 1.0, 9.0, 3.0],
            [2.0, 2.0, 2.0, 8.0, 2.0],
            [3.0, 3.0, 3.0, 7.0, 1.0],
            [4.0, 4.0, 4.0, 6.0, 0.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [6.0, 6.0, 6.0, 4.0, 4.0],
            [7.0, 7.0, 7.0, 3.0, 3.0],
            [8.0, 8.0, 8.0, 9.0, 5.0],
            [9.0, 9.0, 9.0, 2.0, 2.0],
        ],
        pixel_scales=1.0,
    )

    mask = ac.Mask2D(
        mask=[
            [False, False, False, True, False],
            [False, False, False, True, True],
            [False, True, False, False, True],
            [False, False, True, False, False],
            [False, False, False, True, True],
            [False, False, False, True, False],
            [False, False, False, False, False],
            [True, False, False, True, True],
            [False, False, False, False, True],
            [False, False, False, True, False],
        ],
        pixel_scales=1.0,
    )

    arr = ac.Array2D(values=parallel_array.native, mask=mask)

    extract = ac.Extract2DParallelEPER(
        region_list=[
            (1, 3, 0, 5),
            (4, 6, 0, 5),
            (7, 8, 0, 5)
        ])

    binned_array_1d = extract.binned_array_1d_from(
        array=arr, settings=ac.SettingsExtract(pixels=(0, 2))
    )

    assert binned_array_1d == pytest.approx([5.5, 5.45833333], 1.0e-4)