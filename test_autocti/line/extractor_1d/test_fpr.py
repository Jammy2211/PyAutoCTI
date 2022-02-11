import numpy as np
import pytest
import autocti as ac
from autocti import exc


def test__array_1d_list_from(array, masked_array):

    extractor = ac.Extractor1DFPR(region_list=[(1, 4)])

    front_edge_list = extractor.array_1d_list_from(array=array, pixels=(0, 1))
    assert (front_edge_list[0] == np.array([1.0])).all()

    front_edge = extractor.array_1d_list_from(array=array, pixels=(2, 3))
    assert (front_edge[0] == np.array([3.0])).all()

    extractor = ac.Extractor1DFPR(region_list=[(1, 4), (5, 8)])

    front_edge_list = extractor.array_1d_list_from(array=array, pixels=(0, 1))
    assert (front_edge_list[0] == np.array([[1.0]])).all()
    assert (front_edge_list[1] == np.array([[5.0]])).all()

    front_edge_list = extractor.array_1d_list_from(array=array, pixels=(2, 3))
    assert (front_edge_list[0] == np.array([[3.0]])).all()
    assert (front_edge_list[1] == np.array([[7.0]])).all()

    front_edge_list = extractor.array_1d_list_from(array=array, pixels=(0, 3))
    assert (front_edge_list[0] == np.array([1.0, 2.0, 3.0])).all()
    assert (front_edge_list[1] == np.array([5.0, 6.0, 7.0])).all()

    front_edge_list = extractor.array_1d_list_from(array=masked_array, pixels=(0, 3))

    assert (front_edge_list[0].mask == np.array([False, True, False])).all()


def test__stacked_array_1d_from(array, masked_array):

    extractor = ac.Extractor1DFPR(region_list=[(1, 4), (5, 8)])

    stacked_front_edges = extractor.stacked_array_1d_from(array=array, pixels=(0, 3))

    assert (stacked_front_edges == np.array([3.0, 4.0, 5.0])).all()

    extractor = ac.Extractor1DFPR(region_list=[(1, 3), (5, 8)])

    stacked_front_edges = extractor.stacked_array_1d_from(array=array, pixels=(0, 2))

    assert (stacked_front_edges == np.array([3.0, 4.0])).all()

    stacked_front_edges = extractor.stacked_array_1d_from(
        array=masked_array, pixels=(0, 2)
    )

    assert (stacked_front_edges == np.ma.array([1.0, 6.0])).all()
    assert (stacked_front_edges.mask == np.ma.array([False, False])).all()
