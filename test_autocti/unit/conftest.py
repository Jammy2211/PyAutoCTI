from arcticpy.test_arctic.unit.conftest import *

import autofit as af
import autocti as ac
import os
import pytest
import numpy as np

directory = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path():
    af.conf.instance = af.conf.Config(
        os.path.join(directory, "test_files/config"), os.path.join(directory, "output")
    )


@pytest.fixture(name="mask_7x7")
def make_mask_7x7():
    return ac.Mask.unmasked(shape_2d=(7, 7))


@pytest.fixture(name="ci_pattern_7x7")
def make_ci_pattern_7x7():
    return ac.CIPatternUniform(normalization=10.0, regions=[(1, 3, 1, 3)])


@pytest.fixture(name="image_7x7")
def make_image_7x7(ci_pattern_7x7):
    return ac.ci_frame.full(
        fill_value=1.0,
        shape_2d=(7, 7),
        pixel_scales=(1.0, 1.0),
        ci_pattern=ci_pattern_7x7,
    )


@pytest.fixture(name="noise_map_7x7")
def make_noise_map_7x7(ci_pattern_7x7):
    return ac.ci_frame.full(
        fill_value=2.0,
        shape_2d=(7, 7),
        pixel_scales=(1.0, 1.0),
        ci_pattern=ci_pattern_7x7,
    )


@pytest.fixture(name="ci_pre_cti_7x7")
def make_ci_pre_cti_7x7(ci_pattern_7x7):
    return ac.ci_frame.full(
        shape_2d=(7, 7),
        fill_value=10.0,
        pixel_scales=(1.0, 1.0),
        ci_pattern=ci_pattern_7x7,
    )


@pytest.fixture(name="cosmic_ray_map_7x7")
def make_cosmic_ray_map_7x7(ci_pattern_7x7):
    cosmic_ray_map = np.zeros(shape=(7, 7))
    cosmic_ray_map[1, 1] = 4.0
    cosmic_ray_map[1, 2] = 7.0
    return ac.ci_frame.manual(
        array=cosmic_ray_map, pixel_scales=(1.0, 1.0), ci_pattern=ci_pattern_7x7
    )


@pytest.fixture(name="ci_imaging_7x7")
def make_ci_imaging_7x7(image_7x7, noise_map_7x7, ci_pre_cti_7x7, cosmic_ray_map_7x7):

    return ac.ci_imaging(
        image=image_7x7,
        noise_map=noise_map_7x7,
        ci_pre_cti=ci_pre_cti_7x7,
        cosmic_ray_map=cosmic_ray_map_7x7,
    )


@pytest.fixture(name="noise_scaling_maps_7x7")
def make_noise_scaling_maps_7x7(ci_pattern_7x7):

    return [
        ac.ci_frame.ones(
            shape_2d=(7, 7), pixel_scales=(1.0, 1.0), ci_pattern=ci_pattern_7x7
        ),
        ac.ci_frame.full(
            shape_2d=(7, 7),
            fill_value=2.0,
            pixel_scales=(1.0, 1.0),
            ci_pattern=ci_pattern_7x7,
        ),
    ]


@pytest.fixture(name="masked_ci_imaging_7x7")
def make_masked_ci_imaging_7x7(ci_imaging_7x7, mask_7x7, noise_scaling_maps_7x7):
    return ac.masked.ci_imaging(
        ci_imaging=ci_imaging_7x7,
        mask=mask_7x7,
        noise_scaling_maps=noise_scaling_maps_7x7,
    )


@pytest.fixture(name="hyper_noise_scalars")
def make_hyper_noise_scalars():
    return [
        ac.CIHyperNoiseScalar(scale_factor=1.0),
        ac.CIHyperNoiseScalar(scale_factor=2.0),
    ]


@pytest.fixture(name="fit_ci_imaging_7x7")
def make_fit_ci_imaging_7x7(
    masked_ci_imaging_7x7, cti_params, cti_settings, hyper_noise_scalars
):
    return ac.CIImagingFit(
        masked_ci_imaging=masked_ci_imaging_7x7,
        cti_params=cti_params,
        cti_settings=cti_settings,
        hyper_noise_scalars=hyper_noise_scalars,
    )
