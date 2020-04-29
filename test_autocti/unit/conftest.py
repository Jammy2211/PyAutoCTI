from arcticpy.test_arctic.unit.conftest import *

import autofit as af
from autocti import structures as struct
from autocti import dataset as ds
from autocti import charge_injection as ci
from autocti.pipeline.phase.dataset import PhaseDataset
from autocti.pipeline.phase.ci_imaging import PhaseCIImaging

import os
import pytest
import numpy as np

from test_autocti.mock import mock_pipeline

directory = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path():
    af.conf.instance = af.conf.Config(
        os.path.join(directory, "config"),
        os.path.join(directory, "pipeline/files/output"),
    )


### MASK ###


@pytest.fixture(name="mask_7x7")
def make_mask_7x7():
    return struct.Mask.unmasked(shape_2d=(7, 7), pixel_scales=(1.0, 1.0))


### FRAMES ###


@pytest.fixture(name="image_7x7")
def make_image_7x7():
    return struct.Frame.full(
        fill_value=1.0,
        shape_2d=(7, 7),
        pixel_scales=(1.0, 1.0),
        serial_overscan=(0, 6, 6, 7),
        serial_prescan=(0, 7, 0, 1),
        parallel_overscan=(6, 7, 1, 6),
    )


@pytest.fixture(name="noise_map_7x7")
def make_noise_map_7x7():
    return struct.Frame.full(
        fill_value=2.0,
        shape_2d=(7, 7),
        pixel_scales=(1.0, 1.0),
        serial_overscan=(0, 6, 6, 7),
        serial_prescan=(0, 7, 0, 1),
        parallel_overscan=(6, 7, 1, 6),
    )


### IMAGING ###


@pytest.fixture(name="imaging_7x7")
def make_imaging_7x7(image_7x7, noise_map_7x7):
    return ds.Imaging(image=image_7x7, noise_map=noise_map_7x7, name="mock_imaging_7x7")


### CHARGE INJECTION FRAMES ###


@pytest.fixture(name="ci_pattern_7x7")
def make_ci_pattern_7x7():
    return ci.CIPatternUniform(normalization=10.0, regions=[(1, 5, 1, 5)])


@pytest.fixture(name="ci_image_7x7")
def make_ci_image_7x7(ci_pattern_7x7):
    return ci.CIFrame.full(
        fill_value=1.0,
        shape_2d=(7, 7),
        pixel_scales=(1.0, 1.0),
        ci_pattern=ci_pattern_7x7,
        roe_corner=(1, 0),
        serial_overscan=(0, 6, 6, 7),
        serial_prescan=(0, 7, 0, 1),
        parallel_overscan=(6, 7, 1, 6),
    )


@pytest.fixture(name="ci_noise_map_7x7")
def make_ci_noise_map_7x7(ci_pattern_7x7):
    return ci.CIFrame.full(
        fill_value=2.0,
        shape_2d=(7, 7),
        pixel_scales=(1.0, 1.0),
        roe_corner=(1, 0),
        ci_pattern=ci_pattern_7x7,
        serial_overscan=(0, 6, 6, 7),
        serial_prescan=(0, 7, 0, 1),
        parallel_overscan=(6, 7, 1, 6),
    )


@pytest.fixture(name="ci_pre_cti_7x7")
def make_ci_pre_cti_7x7(ci_pattern_7x7):
    return ci.CIFrame.full(
        shape_2d=(7, 7),
        fill_value=10.0,
        pixel_scales=(1.0, 1.0),
        roe_corner=(1, 0),
        ci_pattern=ci_pattern_7x7,
        serial_overscan=(0, 6, 6, 7),
        serial_prescan=(0, 7, 0, 1),
        parallel_overscan=(6, 7, 1, 6),
    )


@pytest.fixture(name="ci_cosmic_ray_map_7x7")
def make_ci_cosmic_ray_map_7x7(ci_pattern_7x7):
    cosmic_ray_map = np.zeros(shape=(7, 7))

    return ci.CIFrame.manual(
        array=cosmic_ray_map,
        pixel_scales=(1.0, 1.0),
        roe_corner=(1, 0),
        ci_pattern=ci_pattern_7x7,
        serial_overscan=(0, 6, 6, 7),
        serial_prescan=(0, 7, 0, 1),
        parallel_overscan=(6, 7, 1, 6),
    )


@pytest.fixture(name="ci_noise_scaling_maps_7x7")
def make_ci_noise_scaling_maps_7x7(ci_pattern_7x7):

    return [
        ci.CIFrame.ones(
            shape_2d=(7, 7),
            pixel_scales=(1.0, 1.0),
            roe_corner=(1, 0),
            ci_pattern=ci_pattern_7x7,
        ),
        ci.CIFrame.full(
            shape_2d=(7, 7),
            roe_corner=(1, 0),
            fill_value=2.0,
            pixel_scales=(1.0, 1.0),
            ci_pattern=ci_pattern_7x7,
        ),
    ]


### CHARGE INJECTION IMAGING ###


@pytest.fixture(name="ci_imaging_7x7")
def make_ci_imaging_7x7(
    ci_image_7x7, ci_noise_map_7x7, ci_pre_cti_7x7, ci_cosmic_ray_map_7x7
):

    return ci.CIImaging(
        image=ci_image_7x7,
        noise_map=ci_noise_map_7x7,
        ci_pre_cti=ci_pre_cti_7x7,
        cosmic_ray_map=ci_cosmic_ray_map_7x7,
    )


@pytest.fixture(name="masked_ci_imaging_7x7")
def make_masked_ci_imaging_7x7(ci_imaging_7x7, mask_7x7, ci_noise_scaling_maps_7x7):
    return ci.MaskedCIImaging(
        ci_imaging=ci_imaging_7x7,
        mask=mask_7x7,
        noise_scaling_maps=ci_noise_scaling_maps_7x7,
    )


### CHARGE INJECTION FITS ###


@pytest.fixture(name="hyper_noise_scalars")
def make_hyper_noise_scalars():
    return [
        ci.CIHyperNoiseScalar(scale_factor=1.0),
        ci.CIHyperNoiseScalar(scale_factor=2.0),
    ]


@pytest.fixture(name="ci_fit_7x7")
def make_ci_fit_7x7(masked_ci_imaging_7x7, hyper_noise_scalars):
    return ci.CIFitImaging(
        masked_ci_imaging=masked_ci_imaging_7x7,
        ci_post_cti=masked_ci_imaging_7x7.ci_pre_cti,
        hyper_noise_scalars=hyper_noise_scalars,
    )


### PHASES ###


@pytest.fixture(name="phase_dataset_7x7")
def make_phase_data(mask_7x7):
    return PhaseDataset(
        non_linear_class=mock_pipeline.MockNLO, phase_tag="", phase_name="test_phase"
    )


@pytest.fixture(name="phase_ci_imaging_7x7")
def make_phase_ci_imaging_7x7():
    return PhaseCIImaging(
        non_linear_class=mock_pipeline.MockNLO, phase_name="test_phase"
    )
