import numpy as np
import pytest

from autocti.data import cti_image
from autocti.data import mask as msk
from autocti.data.charge_injection import ci_frame
from autocti.data.fitting import fitting_data as fit_data


@pytest.fixture(name='image')
def make_image():
    return cti_image.CTIImage(array=np.ones((3, 3)), frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left())


@pytest.fixture(name="mask")
def make_mask():
    return msk.Mask(array=np.array([[True, True, True, True],
                                    [True, False, False, True],
                                    [True, False, False, True],
                                    [True, True, True, True]]), frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left())


@pytest.fixture(name="fitting_image")
def make_fitting_image(image, mask):
    return fit_data.FittingImage(image=image, noise_map=2.0 * np.ones((3, 3)), mask=mask)


@pytest.fixture(name="fitting_hyper_image")
def make_fitting_hyper_image(image, mask):
    return fit_data.FittingHyperImage(image=image, noise_map=2.0 * np.ones((3, 3)), mask=mask,
                                      noise_scalings=[3.0 * np.ones((3, 3)), 4.0 * np.ones((3, 3))])


class TestFittingImage(object):

    def test_attributes(self, image, mask, fitting_image):
        assert (fitting_image.image == image).all()
        assert (fitting_image.noise_map == 2.0 * np.ones((3, 3))).all()
        assert (fitting_image.mask == mask).all()


class TestFittingHyperImage(object):

    def test_attributes(self, image, mask, fitting_hyper_image):
        assert (fitting_hyper_image.image == image).all()
        assert (fitting_hyper_image.noise_map == 2.0 * np.ones((3, 3))).all()
        assert (fitting_hyper_image.mask == mask).all()
        assert (fitting_hyper_image.noise_scalings[0] == 3.0 * np.ones((3, 3))).all()
        assert (fitting_hyper_image.noise_scalings[1] == 4.0 * np.ones((3, 3))).all()
