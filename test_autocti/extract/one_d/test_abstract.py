import numpy as np
from typing import List, Tuple

import autoarray as aa

from autocti.extract.one_d.abstract import Extract1D


class MockExtract1D(Extract1D):
    def region_list_from(self, pixels: Tuple[int, int]) -> List[aa.Region2D]:
        """
        To test the `Extract1D` object we use the example of the fpr (which is the method
        used by the `Extract1DFPR` class.
        """
        return list(
            map(
                lambda region: region.front_region_from(pixels=pixels), self.region_list
            )
        )


def test__array_1d_list_from(array, masked_array):

    extract = MockExtract1D(region_list=[(1, 4)])

    fpr_list = extract.array_1d_list_from(array=array, pixels=(0, 1))
    assert (fpr_list[0] == np.array([1.0])).all()

    fpr = extract.array_1d_list_from(array=array, pixels=(2, 3))
    assert (fpr[0] == np.array([3.0])).all()

    extract = MockExtract1D(region_list=[(1, 4), (5, 8)])

    fpr_list = extract.array_1d_list_from(array=array, pixels=(0, 1))
    assert (fpr_list[0] == np.array([[1.0]])).all()
    assert (fpr_list[1] == np.array([[5.0]])).all()

    fpr_list = extract.array_1d_list_from(array=array, pixels=(2, 3))
    assert (fpr_list[0] == np.array([[3.0]])).all()
    assert (fpr_list[1] == np.array([[7.0]])).all()

    fpr_list = extract.array_1d_list_from(array=array, pixels=(0, 3))
    assert (fpr_list[0] == np.array([1.0, 2.0, 3.0])).all()
    assert (fpr_list[1] == np.array([5.0, 6.0, 7.0])).all()

    fpr_list = extract.array_1d_list_from(array=masked_array, pixels=(0, 3))

    assert (fpr_list[0].mask == np.array([False, True, False])).all()


def test__stacked_array_1d_from(array, masked_array):

    extract = MockExtract1D(region_list=[(1, 4), (5, 8)])

    stacked_fpr = extract.stacked_array_1d_from(array=array, pixels=(0, 3))

    assert (stacked_fpr == np.array([3.0, 4.0, 5.0])).all()

    extract = MockExtract1D(region_list=[(1, 3), (5, 8)])

    stacked_fpr = extract.stacked_array_1d_from(array=array, pixels=(0, 2))

    assert (stacked_fpr == np.array([3.0, 4.0])).all()

    stacked_fpr = extract.stacked_array_1d_from(array=masked_array, pixels=(0, 2))

    assert (stacked_fpr == np.ma.array([1.0, 6.0])).all()
    assert (stacked_fpr.mask == np.ma.array([False, False])).all()


def test__total_pixels_minimum():
    layout = MockExtract1D(region_list=[(1, 2)])

    assert layout.total_pixels_min == 1

    layout = MockExtract1D(region_list=[(1, 3)])

    assert layout.total_pixels_min == 2

    layout = MockExtract1D(region_list=[(1, 3), (0, 5)])

    assert layout.total_pixels_min == 2

    layout = MockExtract1D(region_list=[(1, 3), (4, 5)])

    assert layout.total_pixels_min == 1


def test__total_pixel_spacing_min():
    layout = MockExtract1D(region_list=[(1, 2)])

    assert layout.total_pixels_min == 1
