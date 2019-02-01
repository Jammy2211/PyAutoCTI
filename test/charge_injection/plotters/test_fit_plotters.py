import numpy as np

from autocti.charge_injection import ci_data
from autocti.charge_injection import ci_fit
from autocti.charge_injection import ci_frame
from autocti.charge_injection.plotters import fit_plotters
from autocti.data import mask as msk
from autocti.model import arctic_params
from autocti.model import arctic_settings
from test.charge_injection.plotters.fixtures import *
from test.mock.mock import MockGeometry, MockPattern


@pytest.fixture(name='fit_path')
def make_fit_setup():
    return "{}/../../test_files/plotting/fit/".format(os.path.dirname(os.path.realpath(__file__)))


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
    return ci_data.CIPreCTI(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=MockPattern(),
                            array=3.0 * np.ones((6, 6)))


@pytest.fixture(name='ci_data')
def make_ci_data(image, noise_map, ci_pre_cti):
    return ci_data.CIData(image=image, noise_map=noise_map, ci_pre_cti=ci_pre_cti, noise_scaling=None)


@pytest.fixture(name='ci_datas_fit')
def make_ci_datas_fit(ci_data, mask):
    ci_data.mask = mask
    return ci_data


@pytest.fixture(name='cti_params')
def make_cti_params():
    parallel_species = arctic_params.Species(trap_density=0.1, trap_lifetime=1.0)

    parallel_ccd = arctic_params.CCD(well_notch_depth=0.01, well_fill_alpha=0.2,
                                     well_fill_beta=0.8, well_fill_gamma=2.0)

    return arctic_params.ArcticParams(parallel_species=[parallel_species], parallel_ccd=parallel_ccd)


@pytest.fixture(name='cti_settings')
def make_cti_settings():
    parallel_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                                 charge_injection_mode=True, readout_offset=0)

    return arctic_settings.ArcticSettings(neomode='NEO', parallel=parallel_settings)


@pytest.fixture(name='ci_fit')
def make_ci_fit(ci_datas_fit, cti_params, cti_settings):
    return ci_fit.CIFit(ci_data_fit=ci_datas_fit, cti_params=cti_params, cti_settings=cti_settings)


def test__image_is_output(ci_fit, fit_path, plot_patch):
    fit_plotters.plot_image(fit=ci_fit, fit_index=0, output_path=fit_path, output_format='png')
    assert fit_path + 'fit_image.png' in plot_patch.paths
