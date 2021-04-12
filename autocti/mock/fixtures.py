import autofit as af
from autoarray.mock.fixtures import *
from autofit.mapper.model import ModelInstance
from autofit.mock.mock import MockSearch, MockSamples
from autocti import charge_injection as ci
from autoarray.structures.arrays.one_d import array_1d
from autoarray.dataset import imaging
from autoarray.structures.frames import frames
from autocti.line import dataset_line
from autocti.mask import mask_2d
from autocti.util import traps
from autocti.util.clocker import Clocker
from autocti.util import ccd
from autocti.analysis import analysis
from autocti.analysis.model_util import CTI

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
    return mask_2d.Mask2D.unmasked(shape_native=(7, 7), pixel_scales=(1.0, 1.0))


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


### LINE DATASET ###


def make_data_7():
    return array_1d.Array1D.full(fill_value=1.0, shape_native=(7,), pixel_scales=1.0)


def make_noise_map_7():
    return array_1d.Array1D.full(fill_value=2.0, shape_native=(7,), pixel_scales=1.0)


def make_pre_cti_line_7():
    return array_1d.Array1D.full(fill_value=1.0, shape_native=(7,), pixel_scales=1.0)


def make_dataset_line_7():

    return dataset_line.DatasetLine(
        data=make_data_7(),
        noise_map=make_noise_map_7(),
        pre_cti_line=make_pre_cti_line_7(),
    )


### CHARGE INJECTION FRAMES ###


def make_pattern_ci_7x7():
    return ci.CIPatternUniform(normalization=10.0, regions=[(1, 5, 1, 5)])


def make_ci_image_7x7():
    return ci.CIFrame.full(
        fill_value=1.0,
        shape_native=(7, 7),
        pixel_scales=(1.0, 1.0),
        pattern_ci=make_pattern_ci_7x7(),
        roe_corner=(1, 0),
        scans=make_scans_7x7(),
    )


def make_ci_noise_map_7x7():
    return ci.CIFrame.full(
        fill_value=2.0,
        shape_native=(7, 7),
        pixel_scales=(1.0, 1.0),
        roe_corner=(1, 0),
        pattern_ci=make_pattern_ci_7x7(),
        scans=make_scans_7x7(),
    )


def make_pre_cti_ci_7x7():
    return ci.CIFrame.full(
        shape_native=(7, 7),
        fill_value=10.0,
        pixel_scales=(1.0, 1.0),
        roe_corner=(1, 0),
        pattern_ci=make_pattern_ci_7x7(),
        scans=make_scans_7x7(),
    )


def make_ci_cosmic_ray_map_7x7():
    cosmic_ray_map = np.zeros(shape=(7, 7))

    return ci.CIFrame.manual(
        array=cosmic_ray_map,
        pixel_scales=(1.0, 1.0),
        roe_corner=(1, 0),
        pattern_ci=make_pattern_ci_7x7(),
        scans=make_scans_7x7(),
    )


def make_ci_noise_scaling_maps_7x7():

    return [
        ci.CIFrame.ones(
            shape_native=(7, 7),
            pixel_scales=(1.0, 1.0),
            roe_corner=(1, 0),
            scans=make_scans_7x7(),
            pattern_ci=make_pattern_ci_7x7(),
        ),
        ci.CIFrame.full(
            shape_native=(7, 7),
            roe_corner=(1, 0),
            fill_value=2.0,
            scans=make_scans_7x7(),
            pixel_scales=(1.0, 1.0),
            pattern_ci=make_pattern_ci_7x7(),
        ),
    ]


### CHARGE INJECTION IMAGING ###


def make_imaging_ci_7x7():

    return ci.CIImaging(
        image=make_ci_image_7x7(),
        noise_map=make_ci_noise_map_7x7(),
        pre_cti_ci=make_pre_cti_ci_7x7(),
        cosmic_ray_map=make_ci_cosmic_ray_map_7x7(),
        noise_scaling_maps=make_ci_noise_scaling_maps_7x7(),
    )


### CHARGE INJECTION FITS ###


def make_hyper_noise_scalars():
    return [
        ci.CIHyperNoiseScalar(scale_factor=1.0),
        ci.CIHyperNoiseScalar(scale_factor=2.0),
    ]


def make_fit_ci_7x7():
    return ci.CIFitImaging(
        imaging_ci=make_imaging_ci_7x7(),
        post_cti_ci=make_imaging_ci_7x7().pre_cti_ci,
        hyper_noise_scalars=make_hyper_noise_scalars(),
    )


# ### PHASES ###


def make_samples_with_result():

    model = af.CollectionPriorModel(
        cti=af.Model(
            CTI,
            parallel_traps=[traps.TrapInstantCapture],
            parallel_ccd=make_ccd(),
            serial_traps=[traps.TrapInstantCapture],
            serial_ccd=make_ccd(),
        )
    )

    instance = model.instance_from_prior_medians()

    return MockSamples(max_log_likelihood_instance=instance)


def make_analysis_imaging_ci_7x7():
    return analysis.AnalysisCIImaging(
        dataset_list=[make_imaging_ci_7x7()], clocker=make_parallel_clocker()
    )


### EUCLID DATA ####


def make_euclid_data():
    return np.zeros((2086, 2128))


### ACS DATA ####


def make_acs_ccd():
    return np.zeros((2068, 4144))


def make_acs_quadrant():
    return np.zeros((2068, 2072))
