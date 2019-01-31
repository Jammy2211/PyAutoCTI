import os
import shutil

import pytest
import numpy as np

from autofit import conf
from autocti.data import mask as msk
from autocti.charge_injection import ci_frame
from autocti.charge_injection import ci_data
from autocti.charge_injection.plotters import data_plotters

from test.mock.mock import MockGeometry, MockPattern


@pytest.fixture(name='general_config')
def make_general_config():
    general_config_path = "{}/../../test_files/configs/plotting/".format(os.path.dirname(os.path.realpath(__file__)))
    conf.instance.general = conf.NamedConfig(general_config_path+"general.ini")

@pytest.fixture(name='data_plotter_path')
def make_data_plotter_setup():
    data_plotter_path = "{}/../../test_files/plotting/data/".format(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(data_plotter_path):
        shutil.rmtree(data_plotter_path)

    os.mkdir(data_plotter_path)

    return data_plotter_path

@pytest.fixture(name='image')
def make_image():
    return ci_frame.CIFrame(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=np.ones((6,6)))

@pytest.fixture(name='noise_map')
def make_noise_map():
    return ci_frame.CIFrame(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=2.0*np.ones((6,6)))

@pytest.fixture(name='mask')
def make_mask():
    return msk.Mask.empty_for_shape(shape=(6,6), frame_geometry=MockGeometry(), ci_pattern=MockPattern())


def test__image_is_output(image, mask, data_plotter_path):

    data_plotters.plot_image(image=image, mask=mask, output_path=data_plotter_path, output_format='png')
    assert os.path.isfile(path=data_plotter_path+'image.png')
    os.remove(path=data_plotter_path+'image.png')

def test__noise_map_is_output(noise_map, mask, data_plotter_path):

    data_plotters.plot_noise_map(noise_map=noise_map, mask=mask, output_path=data_plotter_path, output_format='png')
    assert os.path.isfile(path=data_plotter_path+'noise_map.png')
    os.remove(path=data_plotter_path+'noise_map.png')

def test__signal_to_noise_map_is_output(image, noise_map, mask, data_plotter_path):

    data_plotters.plot_signal_to_noise_map(signal_to_noise_map=image/noise_map, mask=mask, output_path=data_plotter_path,
                                          output_format='png')
    assert os.path.isfile(path=data_plotter_path+'signal_to_noise_map.png')
    os.remove(path=data_plotter_path+'signal_to_noise_map.png')