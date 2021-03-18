from autoarray.mock.fixtures import *
from autofit.mapper.model import ModelInstance
from autofit.mock.mock import MockSearch, MockSamples
from autocti import charge_injection as ci
from autocti.dataset import imaging
from autocti.structures import frames
from autocti.mask import mask as msk
from autocti.util import traps
from autocti.util.clocker import Clocker
from autocti.util import ccd
from autocti.analysis import analysis
from autocti.analysis import result as res
from autocti.pipeline.phase.ci_imaging import phase

import numpy as np

### Arctic ###


def make_trap_0():
    return traps.TrapInstantCapture(density=10, release_timescale=-1 / np.log(0.5))


def make_trap_1():
    return traps.TrapInstantCapture(density=8, release_timescale=-1 / np.log(0.2))


def make_traps_x1():
    return [make_trap_0()]


def make_traps_x2():
    return [make_trap_0(), make_trap_1()]


def make_ccd():
    return ccd.CCD(well_fill_power=0.5, full_well_depth=10000, well_notch_depth=1e-7)


def make_parallel_clocker():
    return Clocker(parallel_express=2, parallel_charge_injection_mode=False)


def make_serial_clocker():
    return Clocker(serial_express=2)


### MASK ###


def make_mask_7x7_unmasked():
    return msk.Mask2D.unmasked(shape_native=(7, 7), pixel_scales=(1.0, 1.0))


### FRAMES ###


def make_image_7x7_frame():
    return frames.Frame2D.full(
        fill_value=1.0,
        shape_native=(7, 7),
        scans=make_scans_7x7(),
        pixel_scales=(1.0, 1.0),
    )


def make_noise_map_7x7_frame():
    return frames.Frame2D.full(
        fill_value=2.0,
        shape_native=(7, 7),
        pixel_scales=(1.0, 1.0),
        scans=make_scans_7x7(),
    )


### IMAGING ###


def make_imaging_7x7_frame():
    return imaging.Imaging(
        image=make_image_7x7_frame(),
        noise_map=make_noise_map_7x7_frame(),
        name="mock_imaging_7x7_frame",
    )


### CHARGE INJECTION FRAMES ###


def make_ci_pattern_7x7():
    return ci.CIPatternUniform(normalization=10.0, regions=[(1, 5, 1, 5)])


def make_ci_image_7x7():
    return ci.CIFrame.full(
        fill_value=1.0,
        shape_native=(7, 7),
        pixel_scales=(1.0, 1.0),
        ci_pattern=make_ci_pattern_7x7(),
        roe_corner=(1, 0),
        scans=make_scans_7x7(),
    )


def make_ci_noise_map_7x7():
    return ci.CIFrame.full(
        fill_value=2.0,
        shape_native=(7, 7),
        pixel_scales=(1.0, 1.0),
        roe_corner=(1, 0),
        ci_pattern=make_ci_pattern_7x7(),
        scans=make_scans_7x7(),
    )


def make_ci_pre_cti_7x7():
    return ci.CIFrame.full(
        shape_native=(7, 7),
        fill_value=10.0,
        pixel_scales=(1.0, 1.0),
        roe_corner=(1, 0),
        ci_pattern=make_ci_pattern_7x7(),
        scans=make_scans_7x7(),
    )


def make_ci_cosmic_ray_map_7x7():
    cosmic_ray_map = np.zeros(shape=(7, 7))

    return ci.CIFrame.manual(
        array=cosmic_ray_map,
        pixel_scales=(1.0, 1.0),
        roe_corner=(1, 0),
        ci_pattern=make_ci_pattern_7x7(),
        scans=make_scans_7x7(),
    )


def make_ci_noise_scaling_maps_7x7():

    return [
        ci.CIFrame.ones(
            shape_native=(7, 7),
            pixel_scales=(1.0, 1.0),
            roe_corner=(1, 0),
            scans=make_scans_7x7(),
            ci_pattern=make_ci_pattern_7x7(),
        ),
        ci.CIFrame.full(
            shape_native=(7, 7),
            roe_corner=(1, 0),
            fill_value=2.0,
            scans=make_scans_7x7(),
            pixel_scales=(1.0, 1.0),
            ci_pattern=make_ci_pattern_7x7(),
        ),
    ]


### CHARGE INJECTION IMAGING ###


def make_ci_imaging_7x7():

    return ci.CIImaging(
        image=make_ci_image_7x7(),
        noise_map=make_ci_noise_map_7x7(),
        ci_pre_cti=make_ci_pre_cti_7x7(),
        cosmic_ray_map=make_ci_cosmic_ray_map_7x7(),
    )


def make_masked_ci_imaging_7x7():
    return ci.MaskedCIImaging(
        ci_imaging=make_ci_imaging_7x7(),
        mask=make_mask_7x7_unmasked(),
        noise_scaling_maps=make_ci_noise_scaling_maps_7x7(),
    )


### CHARGE INJECTION FITS ###


def make_hyper_noise_scalars():
    return [
        ci.CIHyperNoiseScalar(scale_factor=1.0),
        ci.CIHyperNoiseScalar(scale_factor=2.0),
    ]


def make_ci_fit_7x7():
    return ci.CIFitImaging(
        masked_ci_imaging=make_masked_ci_imaging_7x7(),
        ci_post_cti=make_masked_ci_imaging_7x7().ci_pre_cti,
        hyper_noise_scalars=make_hyper_noise_scalars(),
    )


# ### PHASES ###


def make_samples_with_result():

    instance = ModelInstance()

    instance.parallel_traps = [traps.TrapInstantCapture(density=0, release_timescale=1)]
    instance.parallel_ccd = make_ccd()
    instance.serial_traps = [traps.TrapInstantCapture(density=0, release_timescale=1)]
    instance.serial_ccd = make_ccd()

    instance.hyper_noise_scalar_of_ci_regions = None
    instance.hyper_noise_scalar_of_parallel_trails = None
    instance.hyper_noise_scalar_of_serial_trails = None
    instance.hyper_noise_scalar_of_serial_overscan_no_trails = None

    return MockSamples(max_log_likelihood_instance=instance)


def make_analysis_ci_imaging_7x7():
    return analysis.AnalysisCIImaging(
        ci_imagings=make_masked_ci_imaging_7x7(), clocker=make_parallel_clocker()
    )


def make_phase_data():
    from autocti.pipeline.phase.dataset.phase import PhaseDataset

    return PhaseDataset(search=MockSearch(name="test_phase"))


def make_phase_ci_imaging_7x7():
    return phase.PhaseCIImaging(search=MockSearch(name="test_phase"))


### EUCLID DATA ####


def make_euclid_data():
    return np.zeros((2086, 2128))


### ACS DATA ####


def make_acs_ccd():
    return np.zeros((2068, 4144))


def make_acs_quadrant():
    return np.zeros((2068, 2072))
