import numpy as np
from typing import List, Tuple

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

    def region_list_from(self, pixels: Tuple[int, int]) -> List[aa.Region2D]:
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

    fpr_list = extract.array_2d_list_from(array=parallel_array, pixels=(0, 1))
    assert (fpr_list[0] == np.array([[1.0, 1.0, 1.0]])).all()

    fpr_list = extract.array_2d_list_from(array=parallel_array, pixels=(2, 3))
    assert (fpr_list[0] == np.array([[3.0, 3.0, 3.0]])).all()

    fpr_list = extract.array_2d_list_from(array=parallel_array, pixels=(-1, 1))
    assert (fpr_list[0] == np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])).all()

    extract = MockExtract2D(region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

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


def test__stacked_array_2d_from(parallel_array, parallel_masked_array):
    extract = MockExtract2D(region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

    stacked_fpr_list = extract.stacked_array_2d_from(
        array=parallel_array, pixels=(0, 3)
    )

    assert (
        stacked_fpr_list
        == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])
    ).all()

    stacked_fpr_list = extract.stacked_array_2d_from(
        array=parallel_array, pixels=(-1, 1)
    )

    assert (stacked_fpr_list == np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])).all()

    extract = MockExtract2D(region_list=[(1, 3, 0, 3), (5, 8, 0, 3)])

    stacked_fpr_list = extract.stacked_array_2d_from(
        array=parallel_array, pixels=(0, 2)
    )

    assert (stacked_fpr_list == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])).all()

    stacked_fpr_list = extract.stacked_array_2d_from(
        array=parallel_masked_array, pixels=(0, 3)
    )

    assert (
        stacked_fpr_list
        == np.ma.array([[3.0, 3.0, 3.0], [4.0, 6.0, 4.0], [3.0, 5.0, 7.0]])
    ).all()
    assert (
        stacked_fpr_list.mask
        == np.ma.array(
            [[False, False, False], [False, False, False], [False, False, False]]
        )
    ).all()


def test__binned_array_1d_from(parallel_array, parallel_masked_array):
    extract = MockExtract2D(region_list=[(1, 3, 0, 3), (5, 8, 0, 3)])

    fpr_line = extract.binned_array_1d_from(array=parallel_array, pixels=(0, 3))

    assert (fpr_line == np.array([3.0, 4.0, 5.0])).all()

    fpr_line = extract.binned_array_1d_from(array=parallel_array, pixels=(-1, 1))

    assert (fpr_line == np.array([2.0, 3.0])).all()

    extract = MockExtract2D(region_list=[(1, 3, 0, 3), (5, 8, 0, 3)])

    fpr_line = extract.binned_array_1d_from(array=parallel_array, pixels=(0, 2))

    assert (fpr_line == np.array([3.0, 4.0])).all()

    fpr_line = extract.binned_array_1d_from(array=parallel_masked_array, pixels=(0, 3))

    assert (fpr_line == np.array([9.0 / 3.0, 14.0 / 3.0, 5.0])).all()


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

    extract = MockExtract2D(region_list=[(0, 1, 1, 2)])

    dataset_1d = extract.dataset_1d_from(dataset_2d=imaging_ci_7x7, pixels=(0, 2))

    assert (dataset_1d.data == np.array([1.0, 1.0])).all()
    assert (dataset_1d.noise_map == np.array([2.0, 2.0])).all()
    assert (dataset_1d.pre_cti_data == np.array([10.0, 10.0])).all()
    assert dataset_1d.layout.region_list == [(0, 2)]
