from os import path
from os.path import dirname, realpath
import pytest
from matplotlib import pyplot
from autocti.mock import fixtures
from autofit import conf


class PlotPatch:
    def __init__(self):
        self.paths = []

    def __call__(self, path, *args, **kwargs):
        self.paths.append(path)


@pytest.fixture(name="plot_patch")
def make_plot_patch(monkeypatch):
    plot_patch = PlotPatch()
    monkeypatch.setattr(pyplot, "savefig", plot_patch)
    return plot_patch


directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path(request):

    conf.instance.push(
        new_path=path.join(directory, "config"),
        output_path=path.join(directory, "output"),
    )


### Arctic ###


@pytest.fixture(name="trap_0")
def make_trap_0():
    return fixtures.make_trap_0()


@pytest.fixture(name="trap_1")
def make_trap_1():
    return fixtures.make_trap_1()


@pytest.fixture(name="traps_x1")
def make_traps_x1():
    return fixtures.make_traps_x1()


@pytest.fixture(name="traps_x2")
def make_traps_x2():
    return fixtures.make_traps_x2()


@pytest.fixture(name="ccd")
def make_ccd():
    return fixtures.make_ccd()


@pytest.fixture(name="parallel_clocker")
def make_parallel_clocker():
    return fixtures.make_parallel_clocker()


@pytest.fixture(name="serial_clocker")
def make_serial_clocker():
    return fixtures.make_serial_clocker()


### MASK ###


@pytest.fixture(name="mask_7x7_unmasked")
def make_mask_7x7_unmasked():
    return fixtures.make_mask_7x7_unmasked()


### LINES ###


@pytest.fixture(name="data_7")
def make_data_7():
    return fixtures.make_data_7()


@pytest.fixture(name="noise_map_7")
def make_noise_map_7():
    return fixtures.make_noise_map_7()


@pytest.fixture(name="line_pre_cti_7")
def make_line_pre_cti():
    return fixtures.make_line_pre_cti_7()


@pytest.fixture(name="dataset_line_7")
def make_dataset_line_7():
    return fixtures.make_dataset_line_7()


### FRAMES ###


@pytest.fixture(name="scans_7x7")
def make_scans_7x7():
    return fixtures.make_scans_7x7()


@pytest.fixture(name="image_7x7_frame")
def make_image_7x7_frame():
    return fixtures.make_image_7x7_frame()


@pytest.fixture(name="noise_map_7x7_frame")
def make_noise_map_7x7_frame():
    return fixtures.make_noise_map_7x7_frame()


### IMAGING ###


@pytest.fixture(name="imaging_7x7_frame")
def make_imaging_7x7_frame():
    return fixtures.make_imaging_7x7_frame()


### LINEDATASET ###


@pytest.fixture(name="line_dataset_7")
def make_line_dataset_7():
    return fixtures.make_line_dataset_7()


### CHARGE INJECTION FRAMES ###


@pytest.fixture(name="ci_pattern_7x7")
def make_ci_pattern_7x7():
    return fixtures.make_ci_pattern_7x7()


@pytest.fixture(name="ci_image_7x7")
def make_ci_image_7x7():
    return fixtures.make_ci_image_7x7()


@pytest.fixture(name="ci_noise_map_7x7")
def make_ci_noise_map_7x7():
    return fixtures.make_ci_noise_map_7x7()


@pytest.fixture(name="ci_pre_cti_7x7")
def make_ci_pre_cti_7x7():
    return fixtures.make_ci_pre_cti_7x7()


@pytest.fixture(name="ci_cosmic_ray_map_7x7")
def make_ci_cosmic_ray_map_7x7():
    return fixtures.make_ci_cosmic_ray_map_7x7()


@pytest.fixture(name="ci_noise_scaling_maps_7x7")
def make_ci_noise_scaling_maps_7x7():

    return fixtures.make_ci_noise_scaling_maps_7x7()


### CHARGE INJECTION IMAGING ###


@pytest.fixture(name="ci_imaging_7x7")
def make_ci_imaging_7x7():

    return fixtures.make_ci_imaging_7x7()


### CHARGE INJECTION FITS ###


@pytest.fixture(name="hyper_noise_scalars")
def make_hyper_noise_scalars():
    return fixtures.make_hyper_noise_scalars()


@pytest.fixture(name="ci_fit_7x7")
def make_ci_fit_7x7():
    return fixtures.make_ci_fit_7x7()


# ### PHASES ###

from autofit.mapper.model import ModelInstance


@pytest.fixture(name="samples_with_result")
def make_samples_with_result():
    return fixtures.make_samples_with_result()


@pytest.fixture(name="analysis_ci_imaging_7x7")
def make_analysis_ci_imaging_7x7():
    return fixtures.make_analysis_ci_imaging_7x7()


# Datasets


@pytest.fixture(name="euclid_data")
def make_euclid_data():
    return fixtures.make_euclid_data()


@pytest.fixture(name="acs_ccd")
def make_acs_ccd():
    return fixtures.make_acs_ccd()


@pytest.fixture(name="acs_quadrant")
def make_acs_quadrant():
    return fixtures.make_acs_quadrant()
