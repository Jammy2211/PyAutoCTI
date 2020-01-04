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


@pytest.fixture(name="image_7x7")
def make_image_7x7():
    return ac.ci_frame.full(fill_value=1.0, shape_2d=(7, 7), pixel_scales=(1.0, 1.0))


@pytest.fixture(name="noise_map_7x7")
def make_noise_map_7x7():
    return ac.ci_frame.full(fill_value=2.0, shape_2d=(7, 7), pixel_scales=(1.0, 1.0))


@pytest.fixture(name="ci_pre_cti_7x7")
def make_ci_pre_cti_7x7():
    return ac.ci_frame.full(shape_2d=(7, 7), fill_value=10.0, pixel_scales=(1.0, 1.0))


@pytest.fixture(name="cosmic_ray_image_7x7")
def make_cosmic_ray_image_7x7():
    cosmic_ray_image = np.zeros(shape_2d=(7, 7))
    cosmic_ray_image[1, 1] = 4.0
    cosmic_ray_image[1, 2] = 7.0
    return ac.ci_frame.manual(array=cosmic_ray_image, pixel_scales=(1.0, 1.0))


@pytest.fixture(name="ci_imaging_7x7")
def make_ci_imaging_7x7(image_7x7, noise_map_7x7, ci_pre_cti_7x7, cosmic_ray_image_7x7):
    return ac.ci_imaging(
        image=image_7x7,
        noise_map=noise_map_7x7,
        ci_pre_cti=ci_pre_cti_7x7,
        cosmic_ray_image=cosmic_ray_image_7x7,
    )
