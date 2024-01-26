import numpy as np
import pytest

import autocti as ac


def test__region_list_from__via_array_2d_list_from(
    parallel_array, parallel_masked_array
):
    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3)])

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(0, 1))
    )
    assert (array_2d_list[0] == np.array([[1.0, 1.0, 1.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(2, 3))
    )
    assert (array_2d_list[0] == np.array([[3.0, 3.0, 3.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(-1, 1))
    )
    assert (array_2d_list[0] == np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])).all()

    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(0, 1))
    )
    assert (array_2d_list[0] == np.array([[1.0, 1.0, 1.0]])).all()
    assert (array_2d_list[1] == np.array([[5.0, 5.0, 5.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(2, 3))
    )
    assert (array_2d_list[0] == np.array([[3.0, 3.0, 3.0]])).all()
    assert (array_2d_list[1] == np.array([[7.0, 7.0, 7.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(0, 3))
    )
    assert (
        array_2d_list[0]
        == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    ).all()
    assert (
        array_2d_list[1]
        == np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])
    ).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_masked_array, settings=ac.SettingsExtract(pixels=(0, 3))
    )

    assert (
        array_2d_list[0].mask
        == np.array([[False, False, False], [False, True, False], [False, False, True]])
    ).all()

    assert (
        array_2d_list[1].mask
        == np.array(
            [[False, False, False], [False, False, False], [True, False, False]]
        )
    ).all()


def test__region_list_from__via_array_2d_list_from__pixels_from_end(
    parallel_array, parallel_masked_array
):
    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3)], shape_2d=(5, 5))

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels_from_end=1)
    )
    assert (array_2d_list[0] == np.array([[3.0, 3.0, 3.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels_from_end=2)
    )
    assert (array_2d_list[0] == np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])).all()

    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels_from_end=3)
    )
    assert (
        array_2d_list[0]
        == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    ).all()
    assert (
        array_2d_list[1]
        == np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])
    ).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_masked_array, settings=ac.SettingsExtract(pixels_from_end=3)
    )

    assert (
        array_2d_list[0].mask
        == np.array([[False, False, False], [False, True, False], [False, False, True]])
    ).all()

    assert (
        array_2d_list[1].mask
        == np.array(
            [[False, False, False], [False, False, False], [True, False, False]]
        )
    ).all()


def test__binned_region_1d_from():
    extract = ac.Extract2DParallelFPR(region_list=[(1, 3, 0, 3)])

    binned_region_1d = extract.binned_region_1d_from(
        settings=ac.SettingsExtract(pixels=(0, 1))
    )

    assert binned_region_1d == (0, 1)

    binned_region_1d = extract.binned_region_1d_from(
        settings=ac.SettingsExtract(pixels=(-1, 1))
    )

    assert binned_region_1d == (1, 2)

    binned_region_1d = extract.binned_region_1d_from(
        settings=ac.SettingsExtract(pixels=(-7, 18))
    )

    assert binned_region_1d == (7, 25)

    binned_region_1d = extract.binned_region_1d_from(
        settings=ac.SettingsExtract(pixels=(-3, -1))
    )

    assert binned_region_1d == None


def test__estimate_capture():
    extract = ac.Extract2DParallelFPR(region_list=[(1, 5, 0, 3)], shape_2d=(6, 3))

    array = ac.Array2D.no_mask(
        values=[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],  # <- Front edge .
            [1.0, 1.0, 1.0],  # <- Next front edge row.
            [3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0],
            [0.0, 0.0, 0.0],
        ],
        pixel_scales=1.0,
    )

    capture = extract.capture_estimate_from(
        array=array,
        pixels_from_start=1,
        pixels_from_end=1,
    )

    assert capture == pytest.approx(2.0, 1.0e-4)

    capture = extract.capture_estimate_from(
        array=array,
        pixels_from_start=2,
        pixels_from_end=2,
    )

    assert capture == pytest.approx(4.0, 1.0e-4)

    mask = ac.Mask2D(
        mask=[
            [False, False, False],
            [False, False, False],
            [True, True, True],
            [False, False, False],
            [False, False, False],
            [False, False, False],
        ],
        pixel_scales=1.0,
    )

    masked_array = ac.Array2D(values=array.native, mask=mask)

    capture = extract.capture_estimate_from(
        array=masked_array,
        pixels_from_start=2,
        pixels_from_end=2,
    )

    assert capture == pytest.approx(2.0, 1.0e-4)
