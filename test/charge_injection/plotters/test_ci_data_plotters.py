import numpy as np

from autocti.charge_injection import ci_data
from autocti.charge_injection.plotters import ci_data_plotters
from autocti.data import mask as msk
from test.charge_injection.plotters.fixtures import *
from test.mock.mock import MockGeometry, MockPattern


@pytest.fixture(name='data_plotter_path')
def make_ci_data_plotter_setup():
    return "{}/../../test_files/plotting/ci_data/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(name='mask')
def make_mask():
    return msk.Mask.empty_for_shape(shape=(6, 6))


@pytest.fixture(name='image')
def make_image():
    return np.ones((6, 6))


@pytest.fixture(name='noise_map')
def make_noise_map():
    return 2.0 * np.ones((6, 6))


@pytest.fixture(name='ci_pre_cti')
def make_ci_pre_cti():
    return 3.0 * np.ones((6, 6))


@pytest.fixture(name='data')
def make_ci_data(image, noise_map, ci_pre_cti):
    return ci_data.CIData(image=image, noise_map=noise_map, ci_pre_cti=ci_pre_cti, ci_pattern=MockPattern(),
                          ci_frame=MockGeometry())


def test__ci_sub_plot_output(data, data_plotter_path, plot_patch):
    ci_data_plotters.plot_ci_subplot(ci_data=data, extract_array_from_mask=True,
                                     cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                     output_path=data_plotter_path, output_format='png')

    assert data_plotter_path + 'ci_data.png' in plot_patch.paths


def test__ci_individuals__output_dependent_on_inputs(data, data_plotter_path, plot_patch):
    ci_data_plotters.plot_ci_data_individual(ci_data=data, extract_array_from_mask=True,
                                             should_plot_image=True,
                                             should_plot_ci_pre_cti=True,
                                             output_path=data_plotter_path, output_format='png')

    assert data_plotter_path + 'ci_image.png' in plot_patch.paths

    assert data_plotter_path + 'ci_noise_map.png' not in plot_patch.paths

    assert data_plotter_path + 'ci_pre_cti.png' in plot_patch.paths

    assert data_plotter_path + 'ci_signal_to_noise_map.png' not in plot_patch.paths


def test__image_is_output(data, mask, data_plotter_path, plot_patch):
    ci_data_plotters.plot_image(ci_data=data, mask=mask, extract_array_from_mask=True,
                                cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                output_path=data_plotter_path, output_format='png')
    assert data_plotter_path + 'ci_image.png' in plot_patch.paths


def test__noise_map_is_output(data, mask, data_plotter_path, plot_patch):
    ci_data_plotters.plot_noise_map(ci_data=data, mask=mask, extract_array_from_mask=True,
                                    cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                    output_path=data_plotter_path, output_format='png')
    assert data_plotter_path + 'ci_noise_map.png' in plot_patch.paths


def test__ci_pre_cti_is_output(data, mask, data_plotter_path, plot_patch):
    ci_data_plotters.plot_ci_pre_cti(ci_data=data, mask=mask, extract_array_from_mask=True,
                                     cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                     output_path=data_plotter_path, output_format='png')
    assert data_plotter_path + 'ci_pre_cti.png' in plot_patch.paths


def test__signal_to_noise_map_is_output(data, mask, data_plotter_path, plot_patch):
    ci_data_plotters.plot_signal_to_noise_map(ci_data=data, mask=mask, extract_array_from_mask=True,
                                              cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                              output_path=data_plotter_path, output_format='png')
    assert data_plotter_path + 'ci_signal_to_noise_map.png' in plot_patch.paths
