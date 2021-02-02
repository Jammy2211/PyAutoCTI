import autocti as ac
from autoarray.mock.fixtures import *
from autofit.mapper.model import ModelInstance
from autofit.mock.mock import MockSearch, MockSamples

import numpy as np

### Arctic ###


def make_trap_0():
    return ac.TrapInstantCapture(density=10, release_timescale=-1 / np.log(0.5))


def make_trap_1():
    return ac.TrapInstantCapture(density=8, release_timescale=-1 / np.log(0.2))


def make_traps_x1():
    return [make_trap_0()]


def make_traps_x2():
    return [make_trap_0(), make_trap_1()]


def make_ccd():
    return ac.CCD(well_fill_power=0.5, full_well_depth=10000, well_notch_depth=1e-7)


def make_ccd_complex():
    return ac.CCDComplex(
        well_fill_alpha=1.0,
        well_fill_power=0.5,
        full_well_depth=10000,
        well_notch_depth=1e-7,
    )


def make_parallel_clocker():
    return ac.Clocker(parallel_express=2, parallel_charge_injection_mode=False)


def make_serial_clocker():
    return ac.Clocker(serial_express=2)


### MASK ###


def make_mask_7x7():
    return ac.Mask2D.unmasked(shape_native=(7, 7), pixel_scales=(1.0, 1.0))


### FRAMES ###


def make_scans_7x7():
    return ac.Scans(
        serial_overscan=(0, 6, 6, 7),
        serial_prescan=(0, 7, 0, 1),
        parallel_overscan=(6, 7, 1, 6),
    )


def make_image_7x7():
    return ac.Frame2D.full(
        fill_value=1.0,
        shape_native=(7, 7),
        scans=make_scans_7x7(),
        pixel_scales=(1.0, 1.0),
    )


def make_noise_map_7x7():
    return ac.Frame2D.full(
        fill_value=2.0,
        shape_native=(7, 7),
        pixel_scales=(1.0, 1.0),
        scans=make_scans_7x7(),
    )


### IMAGING ###


def make_imaging_7x7():
    return ac.Imaging(
        image=make_image_7x7(), noise_map=make_noise_map_7x7(), name="mock_imaging_7x7"
    )


### CHARGE INJECTION FRAMES ###


def make_ci_pattern_7x7():
    return ac.ci.CIPatternUniform(normalization=10.0, regions=[(1, 5, 1, 5)])


def make_ci_image_7x7():
    return ac.ci.CIFrame.full(
        fill_value=1.0,
        shape_native=(7, 7),
        pixel_scales=(1.0, 1.0),
        ci_pattern=make_ci_pattern_7x7(),
        roe_corner=(1, 0),
        scans=make_scans_7x7(),
    )


def make_ci_noise_map_7x7():
    return ac.ci.CIFrame.full(
        fill_value=2.0,
        shape_native=(7, 7),
        pixel_scales=(1.0, 1.0),
        roe_corner=(1, 0),
        ci_pattern=make_ci_pattern_7x7(),
        scans=make_scans_7x7(),
    )


def make_ci_pre_cti_7x7():
    return ac.ci.CIFrame.full(
        shape_native=(7, 7),
        fill_value=10.0,
        pixel_scales=(1.0, 1.0),
        roe_corner=(1, 0),
        ci_pattern=make_ci_pattern_7x7(),
        scans=make_scans_7x7(),
    )


def make_ci_cosmic_ray_map_7x7():
    cosmic_ray_map = np.zeros(shape=(7, 7))

    return ac.ci.CIFrame.manual(
        array=cosmic_ray_map,
        pixel_scales=(1.0, 1.0),
        roe_corner=(1, 0),
        ci_pattern=make_ci_pattern_7x7(),
        scans=make_scans_7x7(),
    )


def make_ci_noise_scaling_maps_7x7():

    return [
        ac.ci.CIFrame.ones(
            shape_native=(7, 7),
            pixel_scales=(1.0, 1.0),
            roe_corner=(1, 0),
            scans=make_scans_7x7(),
            ci_pattern=make_ci_pattern_7x7(),
        ),
        ac.ci.CIFrame.full(
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

    return ac.ci.CIImaging(
        image=make_ci_image_7x7(),
        noise_map=make_ci_noise_map_7x7(),
        ci_pre_cti=make_ci_pre_cti_7x7(),
        cosmic_ray_map=make_ci_cosmic_ray_map_7x7(),
    )


def make_masked_ci_imaging_7x7():
    return ac.ci.MaskedCIImaging(
        ci_imaging=make_ci_imaging_7x7(),
        mask=make_mask_7x7(),
        noise_scaling_maps=make_ci_noise_scaling_maps_7x7(),
    )


### CHARGE INJECTION FITS ###


def make_hyper_noise_scalars():
    return [
        ac.ci.CIHyperNoiseScalar(scale_factor=1.0),
        ac.ci.CIHyperNoiseScalar(scale_factor=2.0),
    ]


def make_ci_fit_7x7():
    return ac.ci.CIFitImaging(
        masked_ci_imaging=make_masked_ci_imaging_7x7(),
        ci_post_cti=make_masked_ci_imaging_7x7().ci_pre_cti,
        hyper_noise_scalars=make_hyper_noise_scalars(),
    )


# ### PHASES ###


def make_samples_with_result():

    instance = ModelInstance()

    instance.parallel_traps = [ac.TrapInstantCapture(density=0, release_timescale=1)]
    instance.parallel_ccd = make_ccd()
    instance.serial_traps = [ac.TrapInstantCapture(density=0, release_timescale=1)]
    instance.serial_ccd = make_ccd()

    instance.hyper_noise_scalar_of_ci_regions = None
    instance.hyper_noise_scalar_of_parallel_trails = None
    instance.hyper_noise_scalar_of_serial_trails = None
    instance.hyper_noise_scalar_of_serial_overscan_no_trails = None

    return MockSamples(max_log_likelihood_instance=instance)


def make_phase_data():
    from autocti.pipeline.phase.dataset import PhaseDataset

    return PhaseDataset(search=MockSearch(name="test_phase"))


def make_phase_ci_imaging_7x7():
    return ac.PhaseCIImaging(search=MockSearch(name="test_phase"))


### EUCLID DATA ####


def make_euclid_data():
    return np.zeros((2086, 2128))


### ACS DATA ####


def make_acs_ccd():
    return np.zeros((2068, 4144))


def make_acs_quadrant():
    return np.zeros((2068, 2072))
