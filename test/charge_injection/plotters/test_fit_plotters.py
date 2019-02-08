import numpy as np

from autocti.charge_injection import ci_data
from autocti.charge_injection import ci_fit
from autocti.charge_injection import ci_frame
from autocti.charge_injection.plotters import fit_plotters
from autocti.data import mask as msk
from autocti.model import arctic_params
from autocti.model import arctic_settings
from test.charge_injection.plotters.fixtures import *
from test.mock.mock import MockGeometry, MockPattern, MockCIFrame


@pytest.fixture(name='fit_plotter_path')
def make_ci_data_plotter_setup():
    return "{}/../../test_files/plotting/fit/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(name='mask')
def make_mask():
    return msk.Mask.empty_for_shape(shape=(2, 2))


@pytest.fixture(name='image')
def make_image():
    return np.ones((2, 2))


@pytest.fixture(name='noise_map')
def make_noise_map():
    return 2.0 * np.ones((2, 2))


@pytest.fixture(name='ci_pre_cti')
def make_ci_pre_cti():
    return 3.0 * np.ones((2, 2))


@pytest.fixture(name='ci_data_fit')
def make_ci_data_fit(image, noise_map, mask, ci_pre_cti):
    return ci_data.CIDataFit(image=image, noise_map=noise_map, ci_pre_cti=ci_pre_cti, mask=mask,
                             ci_pattern=MockPattern(), ci_frame=MockCIFrame(value=3.0))


@pytest.fixture(name="cti_settings")
def make_cti_settings():
    parallel_settings = arctic_settings.Settings(well_depth=0, niter=1, express=1, n_levels=2000)
    return arctic_settings.ArcticSettings(neomode='NEO', parallel=parallel_settings)


@pytest.fixture(name="cti_params")
def make_cti_params():
    parallel_1_species = arctic_params.Species(trap_density=0.1, trap_lifetime=1.0)
    return arctic_params.ArcticParams(parallel_species=parallel_1_species)


@pytest.fixture(name="fit")
def make_fit(ci_data_fit, cti_params, cti_settings):
    return ci_fit.fit_ci_data_fit_with_cti_params_and_settings(ci_data_fit=ci_data_fit, cti_params=cti_params,
                                                               cti_settings=cti_settings)


def test__image_is_output(fit, mask, fit_plotter_path, plot_patch):
    fit_plotters.plot_image(fit=fit, mask=mask, extract_array_from_mask=True,
                            output_path=fit_plotter_path, output_format='png')
    assert fit_plotter_path + 'fit_image.png' in plot_patch.paths


def test__noise_map_is_output(fit, mask, fit_plotter_path, plot_patch):
    fit_plotters.plot_noise_map(fit=fit, mask=mask, extract_array_from_mask=True,
                                output_path=fit_plotter_path, output_format='png')
    assert fit_plotter_path + 'fit_noise_map.png' in plot_patch.paths


def test__signal_to_noise_map_is_output(fit, mask, fit_plotter_path, plot_patch):
    fit_plotters.plot_signal_to_noise_map(fit=fit, mask=mask, extract_array_from_mask=True,
                                          output_path=fit_plotter_path, output_format='png')
    assert fit_plotter_path + 'fit_signal_to_noise_map.png' in plot_patch.paths


def test__ci_pre_cti_is_output(fit, mask, fit_plotter_path, plot_patch):
    fit_plotters.plot_ci_pre_cti(fit=fit, mask=mask, extract_array_from_mask=True,
                                 output_path=fit_plotter_path, output_format='png')
    assert fit_plotter_path + 'fit_ci_pre_cti.png' in plot_patch.paths


def test__ci_post_cti_is_output(fit, mask, fit_plotter_path, plot_patch):
    fit_plotters.plot_ci_post_cti(fit=fit, mask=mask, extract_array_from_mask=True,
                                  output_path=fit_plotter_path, output_format='png')
    assert fit_plotter_path + 'fit_ci_post_cti.png' in plot_patch.paths


def test__residual_map_is_output(fit, mask, fit_plotter_path, plot_patch):
    fit_plotters.plot_residual_map(fit=fit, mask=mask, extract_array_from_mask=True,
                                   output_path=fit_plotter_path, output_format='png')
    assert fit_plotter_path + 'fit_residual_map.png' in plot_patch.paths


def test__chi_squared_map_is_output(fit, mask, fit_plotter_path, plot_patch):
    fit_plotters.plot_chi_squared_map(fit=fit, mask=mask, extract_array_from_mask=True,
                                      output_path=fit_plotter_path, output_format='png')
    assert fit_plotter_path + 'fit_chi_squared_map.png' in plot_patch.paths
