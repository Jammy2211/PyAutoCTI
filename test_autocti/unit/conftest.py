import autofit as af
import os
import pytest
from test_autoarray.mock import mock_mask

directory = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path():
    af.conf.instance = af.conf.Config(
        os.path.join(directory, "test_files/config"), os.path.join(directory, "output")
    )


@pytest.fixture(name="ci_pre_cti_7x7")
def make_ci_pre_cti_7x7(
    imaging_7x7, mask_7x7, sub_grid_7x7, blurring_grid_7x7, convolver_7x7
):
    return mock_masked_dataset.MockMaskedImaging(
        imaging=imaging_7x7,
        mask=mask_7x7,
        grid=sub_grid_7x7,
        blurring_grid=blurring_grid_7x7,
        convolver=convolver_7x7,
    )

@pytest.fixture(name="ci_imaging_7x7")
def make_ci_imaging_7x7(
    imaging_7x7,
):
    return mock_masked_dataset.MockMaskedImaging(
        imaging=imaging_7x7,
        mask=mask_7x7,
        grid=sub_grid_7x7,
        blurring_grid=blurring_grid_7x7,
        convolver=convolver_7x7,
    )