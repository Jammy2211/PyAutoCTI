import numpy as np

from autocti.charge_injection import ci_data
from autocti.charge_injection import ci_frame
from autocti.charge_injection.plotters import ci_data_plotters
from autocti.data import mask as msk
from test.charge_injection.plotters.fixtures import *
from test.mock.mock import MockGeometry, MockPattern


@pytest.fixture(name='data_plotter_path')
def make_ci_data_plotter_setup():
    return "{}/../../test_files/plotting/ci_data/".format(os.path.dirname(os.path.realpath(__file__)))


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
    return ci_data.CIData(image=image, noise_map=noise_map, ci_pre_cti=ci_pre_cti, noise_scaling=None)


def test__ci_sub_plot_output_dependent_on_config(data, data_plotter_path, plot_patch):
    ci_data_plotters.plot_ci_subplot(ci_data=data, output_path=data_plotter_path, output_format='png')

    assert data_plotter_path + 'ci_data.png' in plot_patch.paths


def test__ci_individuals__output_dependent_on_config(data, data_plotter_path, plot_patch):
    ci_data_plotters.plot_ci_data_individual(ci_data=data, output_path=data_plotter_path, output_format='png')

    assert data_plotter_path + 'ci_image.png' in plot_patch.paths

    assert data_plotter_path + 'ci_noise_map.png' not in plot_patch.paths

    assert data_plotter_path + 'ci_pre_cti.png' in plot_patch.paths

    assert data_plotter_path + 'ci_signal_to_noise_map.png' not in plot_patch.paths


def test__image_is_output(data, mask, data_plotter_path, plot_patch):
    ci_data_plotters.plot_image(ci_data=data, mask=mask, output_path=data_plotter_path, output_format='png')
    assert data_plotter_path + 'ci_image.png' in plot_patch.paths


def test__noise_map_is_output(data, mask, data_plotter_path, plot_patch):
    ci_data_plotters.plot_noise_map(ci_data=data, mask=mask, output_path=data_plotter_path, output_format='png')
    assert data_plotter_path + 'ci_noise_map.png' in plot_patch.paths


def test__ci_pre_cti_is_output(data, mask, data_plotter_path, plot_patch):
    ci_data_plotters.plot_ci_pre_cti(ci_data=data, mask=mask, output_path=data_plotter_path, output_format='png')
    assert data_plotter_path + 'ci_pre_cti.png' in plot_patch.paths


def test__signal_to_noise_map_is_output(data, mask, data_plotter_path, plot_patch):
    ci_data_plotters.plot_signal_to_noise_map(ci_data=data, mask=mask, output_path=data_plotter_path,
                                              output_format='png')
    assert data_plotter_path + 'ci_signal_to_noise_map.png' in plot_patch.paths
