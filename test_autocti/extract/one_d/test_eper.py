import numpy as np

import autocti as ac


def test__array_1d_list_from(array, masked_array):

    extract = ac.Extract1DEPER(region_list=[(1, 3)])

    trails_list = extract.array_1d_list_from(array=array, pixels=(0, 1))
    assert (trails_list == np.array([3.0])).all()

    trails_list = extract.array_1d_list_from(array=array, pixels=(2, 3))
    assert (trails_list == np.array([5.0])).all()

    trails_list = extract.array_1d_list_from(array=array, pixels=(1, 4))
    assert (trails_list == np.array([4.0, 5.0, 6.0])).all()

    extract = ac.Extract1DEPER(region_list=[(1, 3), (4, 6)])

    trails_list = extract.array_1d_list_from(array=array, pixels=(0, 1))
    assert (trails_list[0] == np.array([3.0])).all()
    assert (trails_list[1] == np.array([6.0])).all()

    trails_list = extract.array_1d_list_from(array=array, pixels=(0, 2))
    assert (trails_list[0] == np.array([3.0, 4.0])).all()
    assert (trails_list[1] == np.array([6.0, 7.0])).all()

    trails_list = extract.array_1d_list_from(array=masked_array, pixels=(0, 2))

    assert (trails_list[0].mask == np.array([False, False])).all()

    assert (trails_list[1].mask == np.array([False, False])).all()


def test__stacked_array_1d_from(array, masked_array):

    extract = ac.Extract1DEPER(region_list=[(1, 3), (5, 7)])

    stacked_trails = extract.stacked_array_1d_from(array=array, pixels=(0, 2))

    assert (stacked_trails == np.array([5.0, 6.0])).all()

    stacked_trails = extract.stacked_array_1d_from(array=masked_array, pixels=(0, 2))

    assert (stacked_trails == np.array([5.0, 4.0])).all()
    assert (stacked_trails.mask == np.array([False, False])).all()
