import numpy as np

import autocti as ac


def test__for_four_quadrants__loads_data_and_dimensions(euclid_data):

    euclid_array = ac.euclid.Array2DEuclid.top_left(array_electrons=euclid_data)

    assert euclid_array.shape_native == (2086, 2128)
    assert (euclid_array.native == np.zeros((2086, 2128))).all()

    euclid_array = ac.euclid.Array2DEuclid.top_right(array_electrons=euclid_data)

    assert euclid_array.shape_native == (2086, 2128)
    assert (euclid_array.native == np.zeros((2086, 2128))).all()

    euclid_array = ac.euclid.Array2DEuclid.bottom_left(array_electrons=euclid_data)

    assert euclid_array.shape_native == (2086, 2128)
    assert (euclid_array.native == np.zeros((2086, 2128))).all()

    euclid_array = ac.euclid.Array2DEuclid.bottom_right(array_electrons=euclid_data)

    assert euclid_array.shape_native == (2086, 2128)
    assert (euclid_array.native == np.zeros((2086, 2128))).all()
