import numpy as np
import autocti as ac


def test__regions_array_1d_from():

    extract = ac.Extract1DMaster(region_list=[(0, 3)])

    array = ac.Array1D.no_mask(values=[0.0, 1.0, 2.0, 3.0], pixel_scales=1.0)

    array_extracted = extract.regions_array_1d_from(array=array)

    assert (array_extracted == np.array([0.0, 1.0, 2.0, 0.0])).all()

    extract = ac.Extract1DMaster(region_list=[(0, 1), (2, 3)])

    array_extracted = extract.regions_array_1d_from(array=array)

    assert (array_extracted == np.array([0.0, 0.0, 2.0, 0.0])).all()


def test__non_regions_array_1d_from():

    extract = ac.Extract1DMaster(region_list=[(0, 3)])

    array = ac.Array1D.no_mask(values=[0.0, 1.0, 2.0, 3.0], pixel_scales=1.0)

    array_extracted = extract.non_regions_array_1d_from(array=array)

    assert (array_extracted == np.array([0.0, 0.0, 0.0, 3.0])).all()

    extract = ac.Extract1DMaster(region_list=[(0, 1), (3, 4)])

    array_extracted = extract.non_regions_array_1d_from(array=array)

    assert (array_extracted == np.array([0.0, 1.0, 2.0, 0.0])).all()


def test__array_1d_of_edges_and_eper_from(array):

    extract = ac.Extract1DMaster(region_list=[(0, 4)])

    extracted_array = extract.array_1d_of_edges_and_eper_from(
        array=array, fpr_pixels=(0, 2), eper_pixels=(0, 2)
    )

    assert (
        extracted_array == np.array([0.0, 1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0])
    ).all()

    extract = ac.Extract1DMaster(region_list=[(0, 1), (3, 4)])

    extracted_array = extract.array_1d_of_edges_and_eper_from(
        array=array, fpr_pixels=(0, 1), eper_pixels=(0, 1)
    )

    assert (
        extracted_array == np.array([0.0, 1.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ).all()
