import numpy as np
import pytest
from typing import List, Optional, Tuple

import autoarray as aa
import autocti as ac

from autocti.extract.one_d.abstract import Extract1D


class MockExtract1D(Extract1D):
    def region_list_from(
        self,
        settings: ac.SettingsExtract,
    ) -> List[aa.Region2D]:
        """
        To test the `Extract1D` object we use the example of the fpr (which is the method
        used by the `Extract1DFPR` class.
        """
        return list(
            map(
                lambda region: region.front_region_from(
                    pixels=settings.pixels, pixels_from_end=settings.pixels_from_end
                ),
                self.region_list,
            )
        )


def test__array_1d_list_from(array, masked_array):
    extract = MockExtract1D(region_list=[(1, 4)])

    fpr_list = extract.array_1d_list_from(
        array=array, settings=ac.SettingsExtract(pixels=(0, 1))
    )
    assert (fpr_list[0] == np.array([1.0])).all()

    fpr = extract.array_1d_list_from(
        array=array, settings=ac.SettingsExtract(pixels=(2, 3))
    )
    assert (fpr[0] == np.array([3.0])).all()

    extract = MockExtract1D(region_list=[(1, 4), (5, 8)])

    fpr_list = extract.array_1d_list_from(
        array=array, settings=ac.SettingsExtract(pixels=(0, 1))
    )
    assert (fpr_list[0] == np.array([[1.0]])).all()
    assert (fpr_list[1] == np.array([[5.0]])).all()

    fpr_list = extract.array_1d_list_from(
        array=array, settings=ac.SettingsExtract(pixels=(2, 3))
    )
    assert (fpr_list[0] == np.array([[3.0]])).all()
    assert (fpr_list[1] == np.array([[7.0]])).all()

    fpr_list = extract.array_1d_list_from(
        array=array, settings=ac.SettingsExtract(pixels=(0, 3))
    )
    assert (fpr_list[0] == np.array([1.0, 2.0, 3.0])).all()
    assert (fpr_list[1] == np.array([5.0, 6.0, 7.0])).all()

    fpr_list = extract.array_1d_list_from(
        array=masked_array, settings=ac.SettingsExtract(pixels=(0, 3))
    )

    assert (fpr_list[0].mask == np.array([False, True, False])).all()


def test__stacked_array_1d_from(array, masked_array):
    extract = MockExtract1D(region_list=[(1, 4), (5, 8)])

    stacked_fpr = extract.stacked_array_1d_from(
        array=array, settings=ac.SettingsExtract(pixels=(0, 3))
    )

    assert (stacked_fpr == np.array([3.0, 4.0, 5.0])).all()

    extract = MockExtract1D(region_list=[(1, 3), (5, 8)])

    stacked_fpr = extract.stacked_array_1d_from(
        array=array, settings=ac.SettingsExtract(pixels=(0, 2))
    )

    assert (stacked_fpr == np.array([3.0, 4.0])).all()

    stacked_fpr = extract.stacked_array_1d_from(
        array=masked_array, settings=ac.SettingsExtract(pixels=(0, 2))
    )

    assert (stacked_fpr == np.ma.array([1.0, 6.0])).all()
    assert (stacked_fpr.mask == np.ma.array([False, False])).all()

    extract = MockExtract1D(region_list=[(1, 3), (4, 6), (7, 9)])

    stacked_fpr = extract.stacked_array_1d_from(
        array=masked_array, settings=ac.SettingsExtract(pixels=(0, 2))
    )

    assert (stacked_fpr == np.ma.array([4.0, 0.0])).all()
    assert (stacked_fpr.mask == np.ma.array([False, True])).all()

def test__stacked_array_1d_from__pixels_from_end(array, masked_array):
    extract = MockExtract1D(region_list=[(1, 4), (5, 8)])

    stacked_fpr = extract.stacked_array_1d_from(
        array=array, settings=ac.SettingsExtract(pixels_from_end=3)
    )

    assert (stacked_fpr == np.array([3.0, 4.0, 5.0])).all()


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


def test__add_gaussian_noise_to(array):
    extract = MockExtract1D(region_list=[(1, 4)])

    array_with_noise = extract.add_gaussian_noise_to(
        array=array,
        settings=ac.SettingsExtract(pixels=(0, 1)),
        noise_sigma=1.0,
        noise_seed=1,
    )
    assert array_with_noise == pytest.approx(
        np.array(
            [
                0.0,
                2.62434,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
            ]
        ),
        1.0e-4,
    )

    extract = MockExtract1D(region_list=[(1, 4), (5, 8)])

    array_with_noise = extract.add_gaussian_noise_to(
        array=array,
        settings=ac.SettingsExtract(pixels=(1, 3)),
        noise_sigma=1.0,
        noise_seed=1,
    )
    assert array_with_noise == pytest.approx(
        np.array(
            [
                0.0,
                1.0,
                3.6243,
                2.3882,
                4.0,
                5.0,
                7.6243,
                6.3882,
                8.0,
                9.0,
            ]
        ),
        1.0e-4,
    )
