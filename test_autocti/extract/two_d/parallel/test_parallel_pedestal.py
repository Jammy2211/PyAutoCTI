import numpy as np

import autocti as ac


def test_region_list_from__via_array_2d_list_from(
    parallel_array, parallel_masked_array
):
    extract = ac.Extract2DParallelOverscan(parallel_overscan=(1, 4, 0, 1))

    array_2d_list = extract.array_2d_list_from(array=parallel_array, pixels=(0, 1))
    assert (array_2d_list[0] == np.array([[5.0, 5.0]])).all()
