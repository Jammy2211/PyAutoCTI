
import pytest

import autocti as ac

@pytest.fixture(name="parallel_masked_array")
def make_parallel_masked_array(parallel_array):

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

    return ac.Array2D(values=parallel_array.native, mask=mask)


# @pytest.fixture(name="serial_array")
# def make_serial_array():
#     return ac.Array2D.no_mask(
#         values=[
#             [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
#             [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
#             [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
#         ],
#         pixel_scales=1.0,
#     )
#
#
# @pytest.fixture(name="serial_masked_array")
# def make_serial_masked_array(serial_array):
#     mask = ac.Mask2D(
#         mask=[
#             [False, False, False, False, False, True, False, False, False, False],
#             [False, False, True, False, False, False, True, False, False, False],
#             [False, False, False, False, False, False, False, True, False, False],
#         ],
#         pixel_scales=1.0,
#     )
#
#     return ac.Array2D(values=serial_array.native, mask=mask)


def test__stacked_array_2d_from(parallel_masked_array):

    extract = ac.Extract2DParallelEPER(
        region_list=[
            (1, 3, 0, 5),
            (4, 6, 0, 5),
            (7, 8, 0, 5)
        ])

    stacked_array = extract.stacked_array_2d_from(
        array=parallel_masked_array,
        settings=ac.SettingsExtract(pixels=(0, 2))
    )

    assert stacked_array[0] == pytest.approx([5.66666667, 5.66666667, 7.        , 6.66666667, 2.5], 1.0e-4)
    assert stacked_array[1] == pytest.approx([6.5       , 6.66666667, 6.66666667, 0.        , 2.], 1.0e-4)

    stacked_array_2d_total_pixels = extract.stacked_array_2d_total_pixels_from(
        array=parallel_masked_array,
        settings=ac.SettingsExtract(pixels=(0, 2))
    )

    assert stacked_array_2d_total_pixels[0] == pytest.approx([3, 3, 2, 3, 2], 1.0e-4)
    assert stacked_array_2d_total_pixels[1] == pytest.approx([2, 3, 3, 0, 1], 1.0e-4)

def test__binned_array_1d_from(parallel_masked_array):

    extract = ac.Extract2DParallelEPER(
        region_list=[
            (1, 3, 0, 5),
            (4, 6, 0, 5),
            (7, 8, 0, 5)
        ])

    binned_array_1d = extract.binned_array_1d_from(
        array=parallel_masked_array, settings=ac.SettingsExtract(pixels=(0, 2))
    )

    assert binned_array_1d == pytest.approx([5.5, 5.45833333], 1.0e-4)

    binned_array_1d_total_pixels = extract.binned_array_1d_total_pixels_from(
        array=parallel_masked_array, settings=ac.SettingsExtract(pixels=(0, 2))
    )

    assert binned_array_1d_total_pixels == pytest.approx([13, 9], 1.0e-4)

def test__abstract___value_list_from(parallel_masked_array):

    extract = ac.Extract2DParallelEPER(
        region_list=[
            (1, 3, 0, 5),
            (4, 6, 0, 5),
            (7, 8, 0, 5)
        ])

    median_list = extract._value_list_from(
        array=parallel_masked_array,
        value_str="median",
        settings=ac.SettingsExtract(pixels=(0, 2))
    )

    assert median_list == pytest.approx([6.0, 6.5, 7.0, 7.0, 2.0], 1.0e-4)

    median_list_of_lists = extract._value_list_of_lists_from(
        array=parallel_masked_array,
        value_str="median",
        settings=ac.SettingsExtract(pixels=(0, 2))
    )

    assert median_list_of_lists[0] == pytest.approx([3.5, 3.5, 4.0, 7.0, 1.0], 1.0e-4)
    assert median_list_of_lists[1] == pytest.approx([6.0, 6.5, 6.5, 4.0, 4.0], 1.0e-4)
    assert median_list_of_lists[2] == pytest.approx([8.5, 8.5, 8.5, 9.0, 2.0], 1.0e-4)