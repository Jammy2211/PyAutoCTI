import numpy as np
import pytest

import autocti as ac


@pytest.fixture(name="extract")
def make_extract():
    return ac.Extract2DSerialPrescan(serial_prescan=(0, 3, 1, 4))


@pytest.mark.parametrize(
    "pixels, array",
    [
        ((0, 1), [[1.0], [1.0], [1.0]]),
        ((1, 2), [[2.0], [2.0], [2.0]]),
        ((2, 3), [[3.0], [3.0], [3.0]]),
        ((-1, 1), [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
        ((0, 2), [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]),
        ((1, 4), [[2.0, 3.0, 4.0], [2.0, 3.0, 4.0], [2.0, 3.0, 4.0]]),
    ]
)
def test_region_list_serial_array(extract, serial_array, pixels, array):
    prescan_list = extract.array_2d_list_from(array=serial_array, pixels=pixels)

    assert (prescan_list[0] == np.array(array)).all()


def test_region_list_serial_masked_array(extract, serial_masked_array):
    prescan_list = extract.array_2d_list_from(array=serial_masked_array, pixels=(0, 3))

    assert (
            (prescan_list[0].mask)
            == np.array(
        [[False, False, False], [False, True, False], [False, False, False]]
    )
    ).all()
