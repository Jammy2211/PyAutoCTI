import numpy as np

from autocti.charge_injection import ci_frame
from autocti.charge_injection.plotters import data_plotters
from autocti.data import mask as msk
from test.charge_injection.plotters.fixtures import *
from test.mock.mock import MockGeometry, MockPattern


@pytest.fixture(name='data_plotter_path')
def make_data_plotter_setup():
    return "{}/../../test_files/plotting/data/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(name='mask')
def make_mask():
    return msk.Mask.empty_for_shape(shape=(6, 6))


@pytest.fixture(name='image')
def make_image():
    return ci_frame.CIFrame(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=np.ones((6, 6)))


@pytest.fixture(name='noise_map')
def make_noise_map():
    return ci_frame.CIFrame(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=2.0 * np.ones((6, 6)))


@pytest.fixture(name='ci_pre_cti')
def make_ci_pre_cti():
    return ci_frame.CIFrame(frame_geometry=MockGeometry(), ci_pattern=MockPattern(), array=3.0 * np.ones((6, 6)))


def test__image_is_output(image, mask, data_plotter_path, plot_patch):
    data_plotters.plot_image(image=image, mask=mask, extract_array_from_mask=True, output_path=data_plotter_path,
                             output_format='png')
    assert data_plotter_path + 'image.png' in plot_patch.paths


def test__noise_map_is_output(noise_map, mask, data_plotter_path, plot_patch):
    data_plotters.plot_noise_map(noise_map=noise_map, mask=mask, extract_array_from_mask=True,
                                 output_path=data_plotter_path, output_format='png')
    assert data_plotter_path + 'noise_map.png' in plot_patch.paths


def test__ci_pre_cti_is_output(ci_pre_cti, mask, data_plotter_path, plot_patch):
    data_plotters.plot_ci_pre_cti(ci_pre_cti=ci_pre_cti, mask=mask, extract_array_from_mask=True,
                                  output_path=data_plotter_path, output_format='png')
    assert data_plotter_path + 'ci_pre_cti.png' in plot_patch.paths


def test__signal_to_noise_map_is_output(image, noise_map, mask, data_plotter_path, plot_patch):
    data_plotters.plot_signal_to_noise_map(signal_to_noise_map=image / noise_map, mask=mask,
                                           extract_array_from_mask=True,
                                           output_path=data_plotter_path,
                                           output_format='png')
    assert data_plotter_path + 'signal_to_noise_map.png' in plot_patch.paths
