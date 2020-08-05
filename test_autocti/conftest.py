from matplotlib import pyplot
import numpy as np
import os
import pytest

from autoconf import conf

import autocti as ac
from autocti.pipeline.phase.ci_imaging import PhaseCIImaging
from autocti.pipeline.phase.dataset import PhaseDataset

from test_autoarray.unit.conftest import (
    make_euclid_data,
    make_acs_ccd,
    make_acs_quadrant,
)
from test_autocti import mock

directory = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        os.path.join(directory, "unit/config"),
        os.path.join(directory, "pipeline/files/output"),
    )


class PlotPatch(object):
    def __init__(self):
        self.paths = []

    def __call__(self, path, *args, **kwargs):
        self.paths.append(path)


@pytest.fixture(name="plot_patch")
def make_plot_patch(monkeypatch):
    plot_patch = PlotPatch()
    monkeypatch.setattr(pyplot, "savefig", plot_patch)
    return plot_patch


### Arctic ###


@pytest.fixture(name="trap_0")
def make_trap_0():
    return ac.TrapInstantCapture(density=10, release_timescale=-1 / np.log(0.5))


@pytest.fixture(name="trap_1")
def make_trap_1():
    return ac.TrapInstantCapture(density=8, release_timescale=-1 / np.log(0.2))


@pytest.fixture(name="traps_x1")
def make_traps_x1(trap_0):
    return [trap_0]


@pytest.fixture(name="traps_x2")
def make_traps_x2(trap_0, trap_1):
    return [trap_0, trap_1]


@pytest.fixture(name="ccd")
def make_ccd():
    return ac.CCD(well_fill_power=0.5, full_well_depth=10000, well_notch_depth=1e-7)


@pytest.fixture(name="ccd_complex")
def make_ccd_complex():
    return ac.CCDComplex(
        well_fill_alpha=1.0,
        well_fill_power=0.5,
        full_well_depth=10000,
        well_notch_depth=1e-7,
    )


@pytest.fixture(name="parallel_clocker")
def make_parallel_clocker():
    return ac.Clocker(
        parallel_express=2, parallel_charge_injection_mode=False, parallel_offset=0
    )


@pytest.fixture(name="serial_clocker")
def make_serial_clocker():
    return ac.Clocker(serial_express=2, serial_offset=0)


### MASK ###


@pytest.fixture(name="mask_7x7")
def make_mask_7x7():
    return ac.Mask.unmasked(shape_2d=(7, 7), pixel_scales=(1.0, 1.0))


### FRAMES ###


@pytest.fixture(name="scans_7x7")
def make_scans_7x7():
    return ac.Scans(
        serial_overscan=(0, 6, 6, 7),
        serial_prescan=(0, 7, 0, 1),
        parallel_overscan=(6, 7, 1, 6),
    )


@pytest.fixture(name="image_7x7")
def make_image_7x7(scans_7x7):
    return ac.Frame.full(
        fill_value=1.0, shape_2d=(7, 7), scans=scans_7x7, pixel_scales=(1.0, 1.0)
    )


@pytest.fixture(name="noise_map_7x7")
def make_noise_map_7x7(scans_7x7):
    return ac.Frame.full(
        fill_value=2.0, shape_2d=(7, 7), pixel_scales=(1.0, 1.0), scans=scans_7x7
    )


### IMAGING ###


@pytest.fixture(name="imaging_7x7")
def make_imaging_7x7(image_7x7, noise_map_7x7):
    return ac.Imaging(image=image_7x7, noise_map=noise_map_7x7, name="mock_imaging_7x7")


### CHARGE INJECTION FRAMES ###


@pytest.fixture(name="ci_pattern_7x7")
def make_ci_pattern_7x7():
    return ac.ci.CIPatternUniform(normalization=10.0, regions=[(1, 5, 1, 5)])


@pytest.fixture(name="ci_image_7x7")
def make_ci_image_7x7(ci_pattern_7x7, scans_7x7):
    return ac.ci.CIFrame.full(
        fill_value=1.0,
        shape_2d=(7, 7),
        pixel_scales=(1.0, 1.0),
        ci_pattern=ci_pattern_7x7,
        roe_corner=(1, 0),
        scans=scans_7x7,
    )


@pytest.fixture(name="ci_noise_map_7x7")
def make_ci_noise_map_7x7(ci_pattern_7x7, scans_7x7):
    return ac.ci.CIFrame.full(
        fill_value=2.0,
        shape_2d=(7, 7),
        pixel_scales=(1.0, 1.0),
        roe_corner=(1, 0),
        ci_pattern=ci_pattern_7x7,
        scans=scans_7x7,
    )


@pytest.fixture(name="ci_pre_cti_7x7")
def make_ci_pre_cti_7x7(ci_pattern_7x7, scans_7x7):
    return ac.ci.CIFrame.full(
        shape_2d=(7, 7),
        fill_value=10.0,
        pixel_scales=(1.0, 1.0),
        roe_corner=(1, 0),
        ci_pattern=ci_pattern_7x7,
        scans=scans_7x7,
    )


@pytest.fixture(name="ci_cosmic_ray_map_7x7")
def make_ci_cosmic_ray_map_7x7(ci_pattern_7x7, scans_7x7):
    cosmic_ray_map = np.zeros(shape=(7, 7))

    return ac.ci.CIFrame.manual(
        array=cosmic_ray_map,
        pixel_scales=(1.0, 1.0),
        roe_corner=(1, 0),
        ci_pattern=ci_pattern_7x7,
        scans=scans_7x7,
    )


@pytest.fixture(name="ci_noise_scaling_maps_7x7")
def make_ci_noise_scaling_maps_7x7(ci_pattern_7x7, scans_7x7):

    return [
        ac.ci.CIFrame.ones(
            shape_2d=(7, 7),
            pixel_scales=(1.0, 1.0),
            roe_corner=(1, 0),
            scans=scans_7x7,
            ci_pattern=ci_pattern_7x7,
        ),
        ac.ci.CIFrame.full(
            shape_2d=(7, 7),
            roe_corner=(1, 0),
            fill_value=2.0,
            scans=scans_7x7,
            pixel_scales=(1.0, 1.0),
            ci_pattern=ci_pattern_7x7,
        ),
    ]


### CHARGE INJECTION IMAGING ###


@pytest.fixture(name="ci_imaging_7x7")
def make_ci_imaging_7x7(
    ci_image_7x7, ci_noise_map_7x7, ci_pre_cti_7x7, ci_cosmic_ray_map_7x7
):

    return ac.ci.CIImaging(
        image=ci_image_7x7,
        noise_map=ci_noise_map_7x7,
        ci_pre_cti=ci_pre_cti_7x7,
        cosmic_ray_map=ci_cosmic_ray_map_7x7,
    )


@pytest.fixture(name="masked_ci_imaging_7x7")
def make_masked_ci_imaging_7x7(ci_imaging_7x7, mask_7x7, ci_noise_scaling_maps_7x7):
    return ac.ci.MaskedCIImaging(
        ci_imaging=ci_imaging_7x7,
        mask=mask_7x7,
        noise_scaling_maps=ci_noise_scaling_maps_7x7,
    )


### CHARGE INJECTION FITS ###


@pytest.fixture(name="hyper_noise_scalars")
def make_hyper_noise_scalars():
    return [
        ac.ci.CIHyperNoiseScalar(scale_factor=1.0),
        ac.ci.CIHyperNoiseScalar(scale_factor=2.0),
    ]


@pytest.fixture(name="ci_fit_7x7")
def make_ci_fit_7x7(masked_ci_imaging_7x7, hyper_noise_scalars):
    return ac.ci.CIFitImaging(
        masked_ci_imaging=masked_ci_imaging_7x7,
        ci_post_cti=masked_ci_imaging_7x7.ci_pre_cti,
        hyper_noise_scalars=hyper_noise_scalars,
    )


# ### PHASES ###

from autofit.mapper.model import ModelInstance


@pytest.fixture(name="samples_with_result")
def make_samples_with_result(trap_0, ccd):

    instance = ModelInstance()

    instance.parallel_traps = [ac.TrapInstantCapture(density=0, release_timescale=1)]
    instance.parallel_ccd = ccd
    instance.serial_traps = [ac.TrapInstantCapture(density=0, release_timescale=1)]
    instance.serial_ccd = ccd

    instance.hyper_noise_scalar_of_ci_regions = None
    instance.hyper_noise_scalar_of_parallel_trails = None
    instance.hyper_noise_scalar_of_serial_trails = None
    instance.hyper_noise_scalar_of_serial_overscan_no_trails = None

    return mock.MockSamples(max_log_likelihood_instance=instance)


@pytest.fixture(name="phase_dataset_7x7")
def make_phase_data(mask_7x7):
    return PhaseDataset(phase_name="test_phase", search=mock.MockSearch())


@pytest.fixture(name="phase_ci_imaging_7x7")
def make_phase_ci_imaging_7x7():
    return PhaseCIImaging(phase_name="test_phase", search=mock.MockSearch())
