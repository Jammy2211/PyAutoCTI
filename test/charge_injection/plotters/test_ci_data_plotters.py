import os
import shutil

import pytest
import numpy as np

from autofit import conf
from autocti.data import mask as msk
from autocti.charge_injection import ci_frame
from autocti.charge_injection import ci_data
from autocti.charge_injection.plotters import ci_data_plotters

from test.mock.mock import MockGeometry, MockPattern


@pytest.fixture(name='general_config')
def make_general_config():
    general_config_path = "{}/../../test_files/configs/plotting/".format(os.path.dirname(os.path.realpath(__file__)))
    conf.instance.general = conf.NamedConfig(general_config_path+"general.ini")

@pytest.fixture(name='data_plotter_path')
def make_data_plotter_setup():
    data_plotter_path = "{}/../../test_files/plotting/ci_data/".format(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(data_plotter_path):
        shutil.rmtree(data_plotter_path)

    os.mkdir(data_plotter_path)

    return data_plotter_path

@pytest.fixture(name='mask')
def make_mask():
    return msk.Mask.empty_for_shape(shape=(6,6), frame_geometry=MockGeometry(), ci_pattern=MockPattern())

@pytest.fixture(name='image')
def make_image():
    return ci_frame.CIFrame(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=np.ones((6,6)))

@pytest.fixture(name='noise_map')
def make_noise_map():
    return ci_frame.CIFrame(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=2.0*np.ones((6,6)))

@pytest.fixture(name='ci_pre_cti')
def make_ci_pre_cti():
    return ci_frame.CIFrame(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=3.0*np.ones((6,6)))

@pytest.fixture(name='ci_data')
def make_ci_data(image, noise_map, ci_pre_cti):
    return ci_data.CIData(image=image, noise_map=noise_map, ci_pre_cti=ci_pre_cti, noise_scalings=None)

def test__image_is_output(ci_data, mask, data_plotter_path):

    ci_data_plotters.plot_image(ci_data=ci_data, mask=mask, output_path=data_plotter_path, output_format='png')
    assert os.path.isfile(path=data_plotter_path+'image.png')
    os.remove(path=data_plotter_path+'image.png')

def test__noise_map_is_output(ci_data, mask, data_plotter_path):

    ci_data_plotters.plot_noise_map(ci_data=ci_data, mask=mask, output_path=data_plotter_path, output_format='png')
    assert os.path.isfile(path=data_plotter_path+'noise_map.png')
    os.remove(path=data_plotter_path+'noise_map.png')

def test__ci_pre_cti_is_output(ci_data, mask, data_plotter_path):

    ci_data_plotters.plot_ci_pre_cti(ci_data=ci_data, mask=mask, output_path=data_plotter_path, output_format='png')
    assert os.path.isfile(path=data_plotter_path+'ci_pre_cti.png')
    os.remove(path=data_plotter_path+'ci_pre_cti.png')

def test__signal_to_noise_map_is_output(ci_data, mask, data_plotter_path):

    ci_data_plotters.plot_signal_to_noise_map(ci_data=ci_data, mask=mask, output_path=data_plotter_path,
                                          output_format='png')
    assert os.path.isfile(path=data_plotter_path+'signal_to_noise_map.png')
    os.remove(path=data_plotter_path+'signal_to_noise_map.png')