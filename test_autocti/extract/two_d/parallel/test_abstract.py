import numpy as np
import pytest

import autocti as ac


def test__median_list_from(parallel_array, parallel_masked_array):

    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3)])

    # Extracts [1.0, 2.0, 3.0] of every injection in `parallel_array`

    median_list = extract.median_list_from(array=parallel_array, pixels=(0, 3))
    assert median_list == [2.0, 2.0, 2.0]

    # Reduce pixels to only extract 1.0.

    median_list = extract.median_list_from(array=parallel_array, pixels=(0, 1))
    assert median_list == [1.0, 1.0, 1.0]

    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

    median_list = extract.median_list_from(array=parallel_array, pixels=(0, 3))
    assert median_list == [4.0, 4.0, 4.0]

    median_list = extract.median_list_from(array=parallel_masked_array, pixels=(0, 3))

    assert median_list == [2.5, 4.0, 3.5]



def test__median_list_of_lists_from(parallel_array, parallel_masked_array):

    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3)])

    # Extracts [1.0, 2.0, 3.0] of every injection in `parallel_array`

    median_list_of_lists = extract.median_list_of_lists_from(
        array=parallel_array, pixels=(0, 3)
    )
    assert median_list_of_lists[0] == [2.0, 2.0, 2.0]

    # Reduce pixels to only extract 1.0.

    median_list_of_lists = extract.median_list_of_lists_from(
        array=parallel_array, pixels=(0, 1)
    )
    assert median_list_of_lists[0] == [1.0, 1.0, 1.0]

    extract = ac.Extract2DParallelFPR(region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

    median_list_of_lists = extract.median_list_of_lists_from(
        array=parallel_array, pixels=(0, 3)
    )

    assert median_list_of_lists[0] == [2.0, 2.0, 2.0]
    assert median_list_of_lists[1] == [6.0, 6.0, 6.0]

    median_list_of_lists = extract.median_list_of_lists_from(
        array=parallel_masked_array, pixels=(0, 3)
    )
    assert median_list_of_lists[0] == [2.0, 2.0, 1.5]
    assert median_list_of_lists[1] == [5.5, 6.0, 6.0]
