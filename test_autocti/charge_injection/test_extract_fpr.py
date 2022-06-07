import numpy as np
import pytest

import autocti as ac


def test__injection_normalization_lists_from(parallel_array, parallel_masked_array):

    extract = ac.Extract2DParallelFPRCI(region_list=[(1, 4, 0, 3)])

    # Extracts [1.0, 2.0, 3.0] of every injection in `parallel_array`

    injection_normalization_lists = extract.injection_normalization_lists_from(
        array=parallel_array, pixels=(0, 3)
    )
    assert injection_normalization_lists[0] == [2.0, 2.0, 2.0]

    # Reduce pixels to only extract 1.0.

    injection_normalization_lists = extract.injection_normalization_lists_from(
        array=parallel_array, pixels=(0, 1)
    )
    assert injection_normalization_lists[0] == [1.0, 1.0, 1.0]

    extract = ac.Extract2DParallelFPRCI(region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

    injection_normalization_lists = extract.injection_normalization_lists_from(
        array=parallel_array, pixels=(0, 3)
    )

    assert injection_normalization_lists[0] == [2.0, 2.0, 2.0]
    assert injection_normalization_lists[1] == [6.0, 6.0, 6.0]

    injection_normalization_lists = extract.injection_normalization_lists_from(
        array=parallel_masked_array, pixels=(0, 3)
    )
    assert injection_normalization_lists[0] == [2.0, 2.0, 1.5]
    assert injection_normalization_lists[1] == [5.5, 6.0, 6.0]


def test__injection_normalization_list_from(parallel_array, parallel_masked_array):

    extract = ac.Extract2DParallelFPRCI(region_list=[(1, 4, 0, 3)])

    # Extracts [1.0, 2.0, 3.0] of every injection in `parallel_array`

    injection_normalization_list = extract.injection_normalization_list_from(
        array=parallel_array, pixels=(0, 3)
    )
    assert injection_normalization_list == [2.0, 2.0, 2.0]

    # Reduce pixels to only extract 1.0.

    injection_normalization_list = extract.injection_normalization_list_from(
        array=parallel_array, pixels=(0, 1)
    )
    assert injection_normalization_list == [1.0, 1.0, 1.0]

    extract = ac.Extract2DParallelFPRCI(region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

    injection_normalization_list = extract.injection_normalization_list_from(
        array=parallel_array, pixels=(0, 3)
    )
    assert injection_normalization_list == [4.0, 4.0, 4.0]

    injection_normalization_list = extract.injection_normalization_list_from(
        array=parallel_masked_array, pixels=(0, 3)
    )

    assert injection_normalization_list == [2.5, 4.0, 3.5]
