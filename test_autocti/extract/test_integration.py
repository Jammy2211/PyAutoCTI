
import pytest

import autocti as ac


def test__parallel_binned_array_integration_test():
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