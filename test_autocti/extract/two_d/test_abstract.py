import numpy as np
import pytest
from typing import List, Optional, Tuple

import autoarray as aa
import autocti as ac

from autocti.extract.two_d.abstract import Extract2D


class MockExtract2D(Extract2D):
    @property
    def binning_axis(self) -> int:
        """
        The axis over which binning is performed to turn a 2D parallel FPR into a 1D FPR.

        For a parallel extract `axis=1` such that binning is performed over the rows containing the FPR.
        """
        return 1

    def region_list_from(
        self,
        pixels: Optional[Tuple[int, int]] = None,
        pixels_from_end: Optional[int] = None,
    ) -> List[aa.Region2D]:
        """
        To test the `Extract2D` object we use the example of the parallel front edge (which is the method
        used by the `Extract2DParallelFPR` class.
        """
        return [
            region.parallel_front_region_from(pixels=pixels)
            for region in self.region_list
        ]

    def binned_region_1d_from(self, pixels: Tuple[int, int]) -> aa.Region1D:
        return ac.util.extract_2d.binned_region_1d_fpr_from(pixels=pixels)


def test__array_2d_list_from(parallel_array, parallel_masked_array):
    extract = MockExtract2D(region_list=[(1, 4, 0, 3)])

    array_2d_list = extract.array_2d_list_from(array=parallel_array, pixels=(0, 1))
    assert (array_2d_list[0] == np.array([[1.0, 1.0, 1.0]])).all()

    array_2d_list = extract.array_2d_list_from(array=parallel_array, pixels=(2, 3))
    assert (array_2d_list[0] == np.array([[3.0, 3.0, 3.0]])).all()

    array_2d_list = extract.array_2d_list_from(array=parallel_array, pixels=(-1, 1))
    assert (array_2d_list[0] == np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])).all()

    extract = MockExtract2D(region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

    array_2d_list = extract.array_2d_list_from(array=parallel_array, pixels=(0, 1))
    assert (array_2d_list[0] == np.array([[1.0, 1.0, 1.0]])).all()
    assert (array_2d_list[1] == np.array([[5.0, 5.0, 5.0]])).all()

    array_2d_list = extract.array_2d_list_from(array=parallel_array, pixels=(2, 3))
    assert (array_2d_list[0] == np.array([[3.0, 3.0, 3.0]])).all()
    assert (array_2d_list[1] == np.array([[7.0, 7.0, 7.0]])).all()

    array_2d_list = extract.array_2d_list_from(array=parallel_array, pixels=(0, 3))
    assert (
        array_2d_list[0]
        == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    ).all()
    assert (
        array_2d_list[1]
        == np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])
    ).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_masked_array, pixels=(0, 3)
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


def test__stacked_array_2d_from(parallel_array, parallel_masked_array):
    extract = MockExtract2D(region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

    stacked_array_2d = extract.stacked_array_2d_from(
        array=parallel_array, pixels=(0, 3)
    )

    assert (
        stacked_array_2d
        == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])
    ).all()

    stacked_array_2d = extract.stacked_array_2d_from(
        array=parallel_array, pixels=(-1, 1)
    )

    assert (stacked_array_2d == np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])).all()

    extract = MockExtract2D(region_list=[(1, 3, 0, 3), (5, 8, 0, 3)])

    stacked_array_2d = extract.stacked_array_2d_from(
        array=parallel_array, pixels=(0, 2)
    )

    assert (stacked_array_2d == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])).all()

    stacked_array_2d = extract.stacked_array_2d_from(
        array=parallel_masked_array, pixels=(0, 3)
    )

    assert (
        stacked_array_2d
        == np.ma.array([[3.0, 3.0, 3.0], [4.0, 6.0, 4.0], [3.0, 5.0, 7.0]])
    ).all()
    assert (
        stacked_array_2d.mask
        == np.ma.array(
            [[False, False, False], [False, False, False], [False, False, False]]
        )
    ).all()


def test__stacked_array_2d_total_pixels_from(parallel_array, parallel_masked_array):
    extract = MockExtract2D(region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

    stacked_total_pixels = extract.stacked_array_2d_total_pixels_from(
        array=parallel_array, pixels=(0, 3)
    )

    assert (stacked_total_pixels == np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]])).all()

    stacked_total_pixels = extract.stacked_array_2d_total_pixels_from(
        array=parallel_array, pixels=(-1, 1)
    )

    assert (stacked_total_pixels == np.array([[2, 2, 2], [2, 2, 2]])).all()

    extract = MockExtract2D(region_list=[(1, 3, 0, 3), (5, 8, 0, 3)])

    stacked_total_pixels = extract.stacked_array_2d_total_pixels_from(
        array=parallel_masked_array, pixels=(0, 3)
    )

    assert (
        stacked_total_pixels == np.ma.array([[2, 2, 2], [2, 1, 2], [1, 2, 1]])
    ).all()
    assert (
        stacked_total_pixels.mask
        == np.ma.array(
            [[False, False, False], [False, False, False], [False, False, False]]
        )
    ).all()


def test__binned_array_1d_from(parallel_array, parallel_masked_array):
    extract = MockExtract2D(region_list=[(1, 3, 0, 3), (5, 8, 0, 3)])

    binned_array_1d = extract.binned_array_1d_from(array=parallel_array, pixels=(0, 3))

    assert (binned_array_1d == np.array([3.0, 4.0, 5.0])).all()

    binned_array_1d = extract.binned_array_1d_from(array=parallel_array, pixels=(-1, 1))

    assert (binned_array_1d == np.array([2.0, 3.0])).all()

    extract = MockExtract2D(region_list=[(1, 3, 0, 3), (5, 8, 0, 3)])

    binned_array_1d = extract.binned_array_1d_from(array=parallel_array, pixels=(0, 2))

    assert (binned_array_1d == np.array([3.0, 4.0])).all()

    binned_array_1d = extract.binned_array_1d_from(
        array=parallel_masked_array, pixels=(0, 3)
    )

    assert (binned_array_1d == np.array([9.0 / 3.0, 14.0 / 3.0, 5.0])).all()


def test__binned_array_1d_total_pixels_from(parallel_array, parallel_masked_array):

    extract = MockExtract2D(region_list=[(1, 3, 0, 3), (5, 8, 0, 3)])

    binned_array_1d_total_pixels = extract.binned_array_1d_total_pixels_from(
        array=parallel_array, pixels=(0, 3)
    )

    assert (binned_array_1d_total_pixels == np.array([6, 6, 6])).all()

    binned_array_1d_total_pixels = extract.binned_array_1d_total_pixels_from(
        array=parallel_array, pixels=(-1, 1)
    )

    assert (binned_array_1d_total_pixels == np.array([6, 6])).all()

    extract = MockExtract2D(region_list=[(1, 3, 0, 3), (5, 8, 0, 3)])

    binned_array_1d_total_pixels = extract.binned_array_1d_total_pixels_from(
        array=parallel_masked_array, pixels=(0, 3)
    )

    assert (binned_array_1d_total_pixels == np.array([6, 5, 4])).all()


def test__total_rows_minimum():

    extract = MockExtract2D(region_list=[(1, 2, 0, 1)])

    assert extract.total_rows_min == 1

    extract = MockExtract2D(region_list=[(1, 3, 0, 1)])

    assert extract.total_rows_min == 2

    extract = MockExtract2D(region_list=[(1, 2, 0, 1), (3, 4, 0, 1)])

    assert extract.total_rows_min == 1

    extract = MockExtract2D(region_list=[(1, 2, 0, 1), (3, 5, 0, 1)])

    assert extract.total_rows_min == 1


def test__total_columns_minimum():

    extract = MockExtract2D(region_list=[(0, 1, 1, 2)])

    assert extract.total_columns_min == 1

    extract = MockExtract2D(region_list=[(0, 1, 1, 3)])

    assert extract.total_columns_min == 2

    extract = MockExtract2D(region_list=[(0, 1, 1, 2), (0, 1, 3, 4)])

    assert extract.total_columns_min == 1

    extract = MockExtract2D(region_list=[(0, 1, 1, 2), (0, 1, 3, 5)])

    assert extract.total_columns_min == 1


def test__dataset_1d_from(imaging_ci_7x7):

    extract = MockExtract2D(region_list=[(0, 3, 1, 3)])

    dataset_1d = extract.dataset_1d_from(dataset_2d=imaging_ci_7x7, pixels=(0, 2))

    assert (dataset_1d.data == np.array([1.0, 1.0])).all()
    assert (
        dataset_1d.noise_map == np.array([2.0 / np.sqrt(2), 2.0 / np.sqrt(2)])
    ).all()
    assert (dataset_1d.pre_cti_data == np.array([10.0, 10.0])).all()
    assert dataset_1d.layout.region_list == [(0, 2)]


def test__add_gaussian_noise_to(parallel_array):

    extract = MockExtract2D(region_list=[(1, 4, 0, 3)])

    array_with_noise = extract.add_gaussian_noise_to(
        array=parallel_array, pixels=(0, 1), noise_sigma=1.0, noise_seed=1
    )

    assert array_with_noise == pytest.approx(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [2.62434, 0.38824, 0.47182],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
                [5.0, 5.0, 5.0],
                [6.0, 6.0, 6.0],
                [7.0, 7.0, 7.0],
                [8.0, 8.0, 8.0],
                [9.0, 9.0, 9.0],
            ]
        ),
        1.0e-4,
    )

    extract = MockExtract2D(region_list=[(1, 3, 0, 3), (5, 8, 0, 3)])

    array_with_noise = extract.add_gaussian_noise_to(
        array=parallel_array, pixels=(0, 1), noise_sigma=1.0, noise_seed=1
    )

    assert array_with_noise == pytest.approx(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [2.62434, 0.38824, 0.47182],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
                [6.62434, 4.38824, 4.47182],
                [6.0, 6.0, 6.0],
                [7.0, 7.0, 7.0],
                [8.0, 8.0, 8.0],
                [9.0, 9.0, 9.0],
            ]
        ),
        1.0e-4,
    )

    extract = MockExtract2D(region_list=[(1, 3, 0, 3), (5, 8, 0, 3)])

    array_with_noise = extract.add_gaussian_noise_to(
        array=parallel_array, pixels=(1, 3), noise_sigma=1.0, noise_seed=1
    )

    assert array_with_noise == pytest.approx(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [3.6243, 1.3882, 1.4718],
                [1.9270, 3.8654, 0.6984],
                [
                    4.0,
                    4.0,
                    4.0,
                ],
                [5.0, 5.0, 5.0],
                [7.6243, 5.3882, 5.4718],
                [5.9270, 7.8654, 4.6984],
                [8.0, 8.0, 8.0],
                [9.0, 9.0, 9.0],
            ]
        ),
        1.0e-4,
    )
