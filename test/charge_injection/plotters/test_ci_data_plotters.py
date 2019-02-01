import os
import shutil

import numpy as np
import pytest
from autofit import conf
from matplotlib import pyplot

from autocti.charge_injection import ci_data
from autocti.charge_injection import ci_frame
from autocti.charge_injection.plotters import ci_data_plotters
from autocti.data import mask as msk
from test.mock.mock import MockGeometry, MockPattern


@pytest.fixture(name='general_config', autouse=True)
def make_general_config():
    general_config_path = "{}/../../test_files/configs/plotting/".format(os.path.dirname(os.path.realpath(__file__)))
    conf.instance.general = conf.NamedConfig(general_config_path + "general.ini")


@pytest.fixture(name='data_plotter_path')
def make_ci_data_plotter_setup():
    ci_data_plotter_path = "{}/../../test_files/plotting/ci_data/".format(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(ci_data_plotter_path):
        shutil.rmtree(ci_data_plotter_path)

    os.mkdir(ci_data_plotter_path)

    return ci_data_plotter_path


@pytest.fixture(name='mask')
def make_mask():
    return msk.Mask.empty_for_shape(shape=(6, 6), frame_geometry=MockGeometry(), ci_pattern=MockPattern())


@pytest.fixture(name='image')
def make_image():
    return ci_frame.CIFrame(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=np.ones((6, 6)))


@pytest.fixture(name='noise_map')
def make_noise_map():
    return ci_frame.CIFrame(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=2.0 * np.ones((6, 6)))


@pytest.fixture(name='ci_pre_cti')
def make_ci_pre_cti():
    return ci_frame.CIFrame(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=3.0 * np.ones((6, 6)))


@pytest.fixture(name='data')
def make_ci_data(image, noise_map, ci_pre_cti):
    return ci_data.CIData(image=image, noise_map=noise_map, ci_pre_cti=ci_pre_cti, noise_scalings=None)


class PlotPatch(object):
    def __init__(self):
        self.paths = []

    def __call__(self, path, *args, **kwargs):
        self.paths.append(path)


@pytest.fixture(name="plot_patch")
def make_plot_patch(monkeypatch):
    plot_patch = PlotPatch()
    monkeypatch.setattr(pyplot, 'savefig', plot_patch)
    return plot_patch


def test__ci_sub_plot_output_dependent_on_config(data, data_plotter_path, plot_patch):
    ci_data_plotters.plot_ci_subplot(ci_data=data, output_path=data_plotter_path, output_format='png')

    assert data_plotter_path + 'ci_data.png' in plot_patch.paths


def test__ci_individuals__output_dependent_on_config(data, data_plotter_path):
    ci_data_plotters.plot_ci_data_individual(ci_data=data, output_path=data_plotter_path, output_format='png')

    assert os.path.isfile(path=data_plotter_path + 'ci_image.png')
    os.remove(path=data_plotter_path + 'ci_image.png')

    assert not os.path.isfile(path=data_plotter_path + 'ci_noise_map.png')

    assert os.path.isfile(path=data_plotter_path + 'ci_pre_cti.png')
    os.remove(path=data_plotter_path + 'ci_pre_cti.png')

    assert not os.path.isfile(path=data_plotter_path + 'ci_signal_to_noise_map.png')


def test__image_is_output(data, mask, data_plotter_path):
    ci_data_plotters.plot_image(ci_data=data, mask=mask, output_path=data_plotter_path, output_format='png')
    assert os.path.isfile(path=data_plotter_path + 'ci_image.png')
    os.remove(path=data_plotter_path + 'ci_image.png')


def test__noise_map_is_output(data, mask, data_plotter_path):
    ci_data_plotters.plot_noise_map(ci_data=data, mask=mask, output_path=data_plotter_path, output_format='png')
    assert os.path.isfile(path=data_plotter_path + 'ci_noise_map.png')
    os.remove(path=data_plotter_path + 'ci_noise_map.png')


def test__ci_pre_cti_is_output(data, mask, data_plotter_path):
    ci_data_plotters.plot_ci_pre_cti(ci_data=data, mask=mask, output_path=data_plotter_path, output_format='png')
    assert os.path.isfile(path=data_plotter_path + 'ci_pre_cti.png')
    os.remove(path=data_plotter_path + 'ci_pre_cti.png')


def test__signal_to_noise_map_is_output(data, mask, data_plotter_path):
    ci_data_plotters.plot_signal_to_noise_map(ci_data=data, mask=mask, output_path=data_plotter_path,
                                              output_format='png')
    assert os.path.isfile(path=data_plotter_path + 'ci_signal_to_noise_map.png')
    os.remove(path=data_plotter_path + 'ci_signal_to_noise_map.png')
