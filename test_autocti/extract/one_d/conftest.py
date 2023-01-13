import pytest
import autocti as ac


@pytest.fixture(name="masked_array")
def make_masked_array(array):

    mask = ac.Mask1D(
        mask=[False, False, True, False, False, True, False, False, True, True],
        pixel_scales=1.0,
    )

    return ac.Array1D(values=array.native, mask=mask)
