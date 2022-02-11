import pytest
import autocti as ac


@pytest.fixture(name="array")
def make_array():
    return ac.Array1D.manual_native(
        array=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], pixel_scales=1.0
    )


@pytest.fixture(name="masked_array")
def make_masked_array(array):

    mask = ac.Mask1D.manual(
        mask=[False, False, True, False, False, True, False, False, True],
        pixel_scales=1.0,
    )

    return ac.Array1D.manual_mask(array=array.native, mask=mask)
