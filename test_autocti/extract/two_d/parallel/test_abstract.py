import numpy as np
import pytest

import autocti as ac


def test__median_list_from(parallel_array, parallel_masked_array):

    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3)])

    # Extracts [1.0, 2.0, 3.0] of every injection in `parallel_array`

    median_list = extract.median_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(0, 3))
    )
    assert median_list == [2.0, 2.0, 2.0]

    # Reduce pixels to only extract 1.0.

    median_list = extract.median_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(0, 1))
    )
    assert median_list == [1.0, 1.0, 1.0]

    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

    median_list = extract.median_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(0, 3))
    )
    assert median_list == [4.0, 4.0, 4.0]

    median_list = extract.median_list_from(
        array=parallel_masked_array, settings=ac.SettingsExtract(pixels=(0, 3))
    )

    assert median_list == [3.0, 5.0, 5.0]


def test__median_list_from__pixels_from_end(parallel_array, parallel_masked_array):

    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3)])

    # Extracts [1.0, 2.0, 3.0] of every injection in `parallel_array`

    median_list = extract.median_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels_from_end=1)
    )
    assert median_list == [3.0, 3.0, 3.0]

    # Reduce pixels to only extract 1.0.

    median_list = extract.median_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels_from_end=2)
    )
    assert median_list == [2.5, 2.5, 2.5]

    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3), (5, 10, 0, 3)])
    median_list = extract.median_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels_from_end=3)
    )

    assert median_list == [5.0, 5.0, 5.0]

    median_list = extract.median_list_from(
        array=parallel_masked_array, settings=ac.SettingsExtract(pixels_from_end=3)
    )

    assert median_list == [3.0, 7.0, 7.0]


def test__median_list_of_lists_from(parallel_array, parallel_masked_array):

    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3)])

    # Extracts [1.0, 2.0, 3.0] of every injection in `parallel_array`

    median_list_of_lists = extract.median_list_of_lists_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(0, 3))
    )
    assert median_list_of_lists[0] == [2.0, 2.0, 2.0]

    # Reduce pixels to only extract 1.0.

    median_list_of_lists = extract.median_list_of_lists_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(0, 1))
    )
    assert median_list_of_lists[0] == [1.0, 1.0, 1.0]

    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

    median_list_of_lists = extract.median_list_of_lists_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(0, 3))
    )

    assert median_list_of_lists[0] == [2.0, 2.0, 2.0]
    assert median_list_of_lists[1] == [6.0, 6.0, 6.0]

    median_list_of_lists = extract.median_list_of_lists_from(
        array=parallel_masked_array, settings=ac.SettingsExtract(pixels=(0, 3))
    )
    assert median_list_of_lists[0] == [2.0, 2.0, 1.5]
    assert median_list_of_lists[1] == [5.5, 6.0, 6.0]


def test__median_list_of_lists_from__pixels_from_end(
    parallel_array, parallel_masked_array
):

    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3)])

    # Extracts [1.0, 2.0, 3.0] of every injection in `parallel_array`

    median_list_of_lists = extract.median_list_of_lists_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels_from_end=3)
    )
    assert median_list_of_lists[0] == [2.0, 2.0, 2.0]

    # Reduce pixels to only extract 1.0.

    median_list_of_lists = extract.median_list_of_lists_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels_from_end=1)
    )
    assert median_list_of_lists[0] == [3.0, 3.0, 3.0]

    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

    median_list_of_lists = extract.median_list_of_lists_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels_from_end=2)
    )

    assert median_list_of_lists[0] == [2.5, 2.5, 2.5]
    assert median_list_of_lists[1] == [6.5, 6.5, 6.5]

    median_list_of_lists = extract.median_list_of_lists_from(
        array=parallel_masked_array, settings=ac.SettingsExtract(pixels_from_end=2)
    )
    assert median_list_of_lists[0] == [2.5, 3.0, 2.0]
    assert median_list_of_lists[1] == [6.0, 6.5, 6.5]


def test__std_list_from(parallel_array, parallel_masked_array):

    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3)])

    # Extracts [1.0, 2.0, 3.0] of every injection in `parallel_array`

    std_list = extract.std_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(0, 3))
    )
    assert std_list == pytest.approx([0.81649, 0.81649, 0.81649], 1.0e-4)

    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

    std_list = extract.std_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(0, 3))
    )

    assert std_list == pytest.approx([2.1602, 2.1602, 2.1602], 1.0e-4)

    std_list = extract.std_list_from(
        array=parallel_masked_array, settings=ac.SettingsExtract(pixels=(0, 3))
    )
    assert std_list == pytest.approx([1.85472, 2.15406, 2.31516], 1.0e-4)


def test__std_list_from__pixels_from_end(parallel_array, parallel_masked_array):

    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3)])

    # Extracts [1.0, 2.0, 3.0] of every injection in `parallel_array`

    std_list = extract.std_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels_from_end=3)
    )
    assert std_list == pytest.approx([0.81649, 0.81649, 0.81649], 1.0e-4)

    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

    std_list = extract.std_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels_from_end=2)
    )

    assert std_list == pytest.approx([2.0615, 2.0615, 2.0615], 1.0e-4)

    std_list = extract.std_list_from(
        array=parallel_masked_array, settings=ac.SettingsExtract(pixels_from_end=2)
    )
    assert std_list == pytest.approx([1.6996, 1.6996, 2.16024], 1.0e-4)


def test__std_list_of_lists_from(parallel_array, parallel_masked_array):

    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3)])

    # Extracts [1.0, 2.0, 3.0] of every injection in `parallel_array`

    std_list_of_lists = extract.std_list_of_lists_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(0, 3))
    )
    assert std_list_of_lists[0] == pytest.approx([0.81649, 0.81649, 0.81649], 1.0e-4)

    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

    std_list_of_lists = extract.std_list_of_lists_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(0, 3))
    )
    assert std_list_of_lists[0] == pytest.approx([0.81649, 0.81649, 0.81649], 1.0e-4)
    assert std_list_of_lists[1] == pytest.approx([0.81649, 0.81649, 0.81649], 1.0e-4)

    std_list_of_lists = extract.std_list_of_lists_from(
        array=parallel_masked_array, settings=ac.SettingsExtract(pixels=(0, 3))
    )
    assert std_list_of_lists[0] == pytest.approx([0.81649, 1.0, 0.5], 1.0e-4)
    assert std_list_of_lists[1] == pytest.approx([0.5, 0.81649, 0.81649], 1.0e-4)


def test__std_list_of_lists_from__pixels_from_end(
    parallel_array, parallel_masked_array
):

    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3)])

    # Extracts [1.0, 2.0, 3.0] of every injection in `parallel_array`

    std_list_of_lists = extract.std_list_of_lists_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels_from_end=3)
    )
    assert std_list_of_lists[0] == pytest.approx([0.81649, 0.81649, 0.81649], 1.0e-4)

    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

    std_list_of_lists = extract.std_list_of_lists_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels_from_end=2)
    )
    assert std_list_of_lists[0] == pytest.approx([0.5, 0.5, 0.5], 1.0e-4)
    assert std_list_of_lists[1] == pytest.approx([0.5, 0.5, 0.5], 1.0e-4)

    std_list_of_lists = extract.std_list_of_lists_from(
        array=parallel_masked_array, settings=ac.SettingsExtract(pixels_from_end=3)
    )
    assert std_list_of_lists[0] == pytest.approx([0.81649, 1.0, 0.5], 1.0e-4)
    assert std_list_of_lists[1] == pytest.approx([0.5, 0.81649, 0.81649], 1.0e-4)
