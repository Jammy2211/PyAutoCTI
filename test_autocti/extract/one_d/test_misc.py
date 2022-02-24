import numpy as np
import pytest
import autocti as ac
from autocti import exc


def test__array_1d_of_regions_from():

    extract = ac.Extract1DMisc(region_list=[(0, 3)])

    array = ac.Array1D.manual_native(array=[0.0, 1.0, 2.0, 3.0], pixel_scales=1.0)

    array_extracted = extract.array_1d_of_regions_from(array=array)

    assert (array_extracted == np.array([0.0, 1.0, 2.0, 0.0])).all()

    extract = ac.Extract1DMisc(region_list=[(0, 1), (2, 3)])

    array_extracted = extract.array_1d_of_regions_from(array=array)

    assert (array_extracted == np.array([0.0, 0.0, 2.0, 0.0])).all()


def test__array_1d_of_non_regions_from():

    extract = ac.Extract1DMisc(region_list=[(0, 3)])

    array = ac.Array1D.manual_native(array=[0.0, 1.0, 2.0, 3.0], pixel_scales=1.0)

    array_extracted = extract.array_1d_of_non_regions_from(array=array)

    assert (array_extracted == np.array([0.0, 0.0, 0.0, 3.0])).all()

    extract = ac.Extract1DMisc(region_list=[(0, 1), (3, 4)])

    array_extracted = extract.array_1d_of_non_regions_from(array=array)

    assert (array_extracted == np.array([0.0, 1.0, 2.0, 0.0])).all()


def test__array_1d_of_trails_from():

    extract = ac.Extract1DMisc(region_list=[(0, 2)], prescan=(0, 1), overscan=(0, 1))

    array = ac.Array1D.manual_native(array=[0.0, 1.0, 2.0, 3.0], pixel_scales=1.0)

    array_extracted = extract.array_1d_of_trails_from(array=array)

    assert (array_extracted == np.array([0.0, 0.0, 2.0, 3.0])).all()

    extract = ac.Extract1DMisc(region_list=[(0, 2)], prescan=(2, 3), overscan=(3, 4))

    array_extracted = extract.array_1d_of_trails_from(array=array)

    assert (array_extracted == np.array([0.0, 0.0, 0.0, 0.0])).all()

    extract = ac.Extract1DMisc(
        region_list=[(0, 1), (2, 3)], prescan=(2, 3), overscan=(2, 3)
    )

    array_extracted = extract.array_1d_of_trails_from(array=array)

    assert (array_extracted.native == np.array([0.0, 1.0, 0.0, 3.0])).all()


def test__array_1d_of_edges_and_epers_from(array):

    extract = ac.Extract1DMisc(region_list=[(0, 4)])

    extracted_array = extract.array_1d_of_edges_and_epers_from(
        array=array, fpr_pixels=(0, 2), trails_pixels=(0, 2)
    )

    assert (
        extracted_array == np.array([0.0, 1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0])
    ).all()

    extract = ac.Extract1DMisc(region_list=[(0, 1), (3, 4)])

    extracted_array = extract.array_1d_of_edges_and_epers_from(
        array=array, fpr_pixels=(0, 1), trails_pixels=(0, 1)
    )

    assert (
        extracted_array == np.array([0.0, 1.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ).all()
