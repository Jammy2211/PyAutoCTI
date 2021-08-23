from os import path
import pytest
from matplotlib import pyplot

from autofit import conf

from autocti.mock import fixtures


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


@pytest.fixture(name="clocker_1d")
def make_clocker_1d():
    return fixtures.make_clocker_1d()


@pytest.fixture(name="parallel_clocker_2d")
def make_parallel_clocker():
    return fixtures.make_parallel_clocker_2d()


@pytest.fixture(name="serial_clocker_2d")
def make_serial_clocker():
    return fixtures.make_serial_clocker_2d()


### MASK ###


@pytest.fixture(name="mask_1d_7_unmasked")
def make_mask_1d_7_unmasked():
    return fixtures.make_mask_1d_7_unmasked()


@pytest.fixture(name="mask_2d_7x7_unmasked")
def make_mask_2d_7x7_unmasked():
    return fixtures.make_mask_2d_7x7_unmasked()


### LINES ###


@pytest.fixture(name="layout_7")
def make_layout_7():
    return fixtures.make_layout_7()


@pytest.fixture(name="data_7")
def make_data_7():
    return fixtures.make_data_7()


@pytest.fixture(name="noise_map_7")
def make_noise_map_7():
    return fixtures.make_noise_map_7()


@pytest.fixture(name="pre_cti_data_7")
def make_pre_cti_data():
    return fixtures.make_pre_cti_data_7()


@pytest.fixture(name="dataset_line_7")
def make_dataset_line_7():
    return fixtures.make_dataset_line_7()


@pytest.fixture(name="fit_line_7")
def make_fit_line_7():
    return fixtures.make_fit_line_7()


### FRAMES ###


@pytest.fixture(name="image_7x7_native")
def make_image_7x7_native():
    return fixtures.make_image_7x7_native()


@pytest.fixture(name="noise_map_7x7_native")
def make_noise_map_7x7_native():
    return fixtures.make_noise_map_7x7_native()


### IMAGING ###


@pytest.fixture(name="imaging_7x7_frame")
def make_imaging_7x7_frame():
    return fixtures.make_imaging_7x7_frame()


### CHARGE INJECTION FRAMES ###


@pytest.fixture(name="layout_ci_7x7")
def make_layout_ci_7x7():
    return fixtures.make_layout_ci_7x7()


@pytest.fixture(name="ci_image_7x7")
def make_ci_image_7x7():
    return fixtures.make_ci_image_7x7()


@pytest.fixture(name="ci_noise_map_7x7")
def make_ci_noise_map_7x7():
    return fixtures.make_ci_noise_map_7x7()


@pytest.fixture(name="pre_cti_data_7x7")
def make_pre_cti_data_7x7():
    return fixtures.make_pre_cti_data_7x7()


@pytest.fixture(name="ci_cosmic_ray_map_7x7")
def make_ci_cosmic_ray_map_7x7():
    return fixtures.make_ci_cosmic_ray_map_7x7()


@pytest.fixture(name="ci_noise_scaling_map_list_7x7")
def make_ci_noise_scaling_map_list_7x7():

    return fixtures.make_ci_noise_scaling_map_list_7x7()


### CHARGE INJECTION IMAGING ###


@pytest.fixture(name="imaging_ci_7x7")
def make_imaging_ci_7x7():

    return fixtures.make_imaging_ci_7x7()


### CHARGE INJECTION FITS ###


@pytest.fixture(name="hyper_noise_scalars")
def make_hyper_noise_scalars():
    return fixtures.make_hyper_noise_scalars()


@pytest.fixture(name="fit_ci_7x7")
def make_fit_ci_7x7():
    return fixtures.make_fit_ci_7x7()


# ### PHASES ###

from autofit.mapper.model import ModelInstance


@pytest.fixture(name="samples_with_result")
def make_samples_with_result():
    return fixtures.make_samples_with_result()


@pytest.fixture(name="analysis_imaging_ci_7x7")
def make_analysis_imaging_ci_7x7():
    return fixtures.make_analysis_imaging_ci_7x7()


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
