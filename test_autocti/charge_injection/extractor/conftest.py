import pytest
import autocti as ac


@pytest.fixture(name="parallel_array")
def make_parallel_array():
    return ac.Array2D.manual(
        array=[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],  # <- Front edge .
            [2.0, 2.0, 2.0],  # <- Next front edge row.
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
            [6.0, 6.0, 6.0],
            [7.0, 7.0, 7.0],
            [8.0, 8.0, 8.0],
            [9.0, 9.0, 9.0],
        ],
        pixel_scales=1.0,
    )


@pytest.fixture(name="parallel_masked_array")
def make_parallel_masked_array(parallel_array):

    mask = ac.Mask2D.manual(
        mask=[
            [False, False, False],
            [False, False, False],
            [False, True, False],
            [False, False, True],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [True, False, False],
            [False, False, False],
            [False, False, False],
        ],
        pixel_scales=1.0,
    )

    return ac.Array2D.manual_mask(array=parallel_array.native, mask=mask)


@pytest.fixture(name="serial_array")
def make_serial_array():
    return ac.Array2D.manual(
        array=[
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ],
        pixel_scales=1.0,
    )


@pytest.fixture(name="serial_masked_array")
def make_serial_masked_array(serial_array):

    mask = ac.Mask2D.manual(
        mask=[
            [False, False, False, False, False, True, False, False, False, False],
            [False, False, True, False, False, False, True, False, False, False],
            [False, False, False, False, False, False, False, True, False, False],
        ],
        pixel_scales=1.0,
    )

    return ac.Array2D.manual_mask(array=serial_array.native, mask=mask)
