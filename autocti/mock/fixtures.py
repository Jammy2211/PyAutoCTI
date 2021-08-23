from arcticpy.src import traps, ccd, roe

from autofit.mapper.prior_model.prior_model import PriorModel
from autofit.mapper.prior_model.collection import CollectionPriorModel
from autofit.mock.mock import MockSamples

from autocti.line.fit import FitDatasetLine

from autoarray.mock.fixtures import *
from autoarray.mask.mask_1d import Mask1D
from autoarray.structures.arrays.one_d.array_1d import Array1D
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.dataset.imaging import Imaging

from autocti.line.dataset import DatasetLine
from autocti.line.layout import Layout1DLine
from autocti import charge_injection as ci
from autocti.charge_injection.model.analysis import AnalysisImagingCI
from autocti.mask.mask_2d import Mask2D
from autocti.util.clocker import Clocker1D
from autocti.util.clocker import Clocker2D
from autocti.model.model_util import CTI2D

import numpy as np

### Arctic ###


def make_trap_0():
    return traps.TrapInstantCapture(density=10.0, release_timescale=-1 / np.log(0.5))


def make_trap_1():
    return traps.TrapInstantCapture(density=8.0, release_timescale=-1 / np.log(0.2))


def make_traps_x1():
    return [make_trap_0()]


def make_traps_x2():
    return [make_trap_0(), make_trap_1()]


def make_ccd():
    return ccd.CCDPhase(
        well_fill_power=0.5, full_well_depth=10000.0, well_notch_depth=1e-7
    )


def make_clocker_1d():
    return Clocker1D(express=2)


def make_parallel_clocker_2d():
    return Clocker2D(
        parallel_express=2, parallel_roe=roe.ROE(empty_traps_for_first_transfers=True)
    )


def make_serial_clocker_2d():
    return Clocker2D(serial_express=2)


### MASK ###


def make_mask_1d_7_unmasked():
    return Mask1D.unmasked(shape_slim=(7,), pixel_scales=(1.0,))


def make_mask_2d_7x7_unmasked():
    return Mask2D.unmasked(shape_native=(7, 7), pixel_scales=(1.0, 1.0))


### FRAMES ###


def make_image_7x7_native():
    return Array2D.full(
        fill_value=1.0, shape_native=(7, 7), pixel_scales=(1.0, 1.0)
    ).native


def make_noise_map_7x7_native():
    return Array2D.full(
        fill_value=2.0, shape_native=(7, 7), pixel_scales=(1.0, 1.0)
    ).native


### IMAGING ###


def make_imaging_7x7_frame():
    return Imaging(
        image=make_image_7x7_native(),
        noise_map=make_noise_map_7x7_native(),
        name="mock_imaging_7x7_frame",
    )


### LINE DATASET ###


def make_layout_7():
    return Layout1DLine(
        shape_1d=(7,),
        normalization=10.0,
        region_list=[(1, 5)],
        prescan=(0, 1),
        overscan=(6, 7),
    )


def make_data_7():
    return Array1D.full(fill_value=1.0, shape_native=(7,), pixel_scales=1.0)


def make_noise_map_7():
    return Array1D.full(fill_value=2.0, shape_native=(7,), pixel_scales=1.0)


def make_pre_cti_data_7():
    return Array1D.full(fill_value=1.0, shape_native=(7,), pixel_scales=1.0)


def make_dataset_line_7():

    return DatasetLine(
        data=make_data_7(),
        noise_map=make_noise_map_7(),
        pre_cti_data=make_pre_cti_data_7(),
        layout=make_layout_7(),
    )


### CHARGE INJECTION FRAMES ###


def make_layout_ci_7x7():
    return ci.Layout2DCI(
        shape_2d=(7, 7),
        normalization=10.0,
        region_list=[(1, 5, 1, 5)],
        original_roe_corner=(1, 0),
        serial_overscan=(0, 6, 6, 7),
        serial_prescan=(0, 7, 0, 1),
        parallel_overscan=(6, 7, 1, 6),
    )


def make_ci_image_7x7():
    return Array2D.full(fill_value=1.0, shape_native=(7, 7), pixel_scales=(1.0, 1.0))


def make_ci_noise_map_7x7():
    return Array2D.full(fill_value=2.0, shape_native=(7, 7), pixel_scales=(1.0, 1.0))


def make_pre_cti_data_7x7():
    return Array2D.full(shape_native=(7, 7), fill_value=10.0, pixel_scales=(1.0, 1.0))


def make_ci_cosmic_ray_map_7x7():
    cosmic_ray_map = np.zeros(shape=(7, 7))

    return Array2D.manual(array=cosmic_ray_map, pixel_scales=(1.0, 1.0))


def make_ci_noise_scaling_map_list_7x7():

    return [
        Array2D.ones(shape_native=(7, 7), pixel_scales=(1.0, 1.0)),
        Array2D.full(shape_native=(7, 7), fill_value=2.0, pixel_scales=(1.0, 1.0)),
    ]


### LINE DATASET FITS ###


def make_fit_line_7():
    return FitDatasetLine(
        dataset_line=make_dataset_line_7(),
        post_cti_data=make_dataset_line_7().pre_cti_data + 1,
    )


### CHARGE INJECTION IMAGING ###


def make_imaging_ci_7x7():

    return ci.ImagingCI(
        image=make_ci_image_7x7(),
        noise_map=make_ci_noise_map_7x7(),
        pre_cti_data=make_pre_cti_data_7x7(),
        cosmic_ray_map=make_ci_cosmic_ray_map_7x7(),
        noise_scaling_map_list=make_ci_noise_scaling_map_list_7x7(),
        layout=make_layout_ci_7x7(),
    )


### CHARGE INJECTION FITS ###


def make_hyper_noise_scalars():
    return [
        ci.HyperCINoiseScalar(scale_factor=1.0),
        ci.HyperCINoiseScalar(scale_factor=2.0),
    ]


def make_fit_ci_7x7():
    return ci.FitImagingCI(
        imaging=make_imaging_ci_7x7(),
        post_cti_data=make_imaging_ci_7x7().pre_cti_data,
        hyper_noise_scalars=make_hyper_noise_scalars(),
    )


# ### PHASES ###


def make_samples_with_result():

    model = CollectionPriorModel(
        cti=PriorModel(
            CTI2D,
            parallel_traps=[traps.TrapInstantCapture],
            parallel_ccd=make_ccd(),
            serial_traps=[traps.TrapInstantCapture],
            serial_ccd=make_ccd(),
        )
    )

    instance = model.instance_from_prior_medians()

    return MockSamples(max_log_likelihood_instance=instance)


def make_analysis_imaging_ci_7x7():
    return AnalysisImagingCI(
        dataset_ci=make_imaging_ci_7x7(), clocker=make_parallel_clocker_2d()
    )


### EUCLID DATA ####


def make_euclid_data():
    return np.zeros((2086, 2128))


### ACS DATA ####


def make_acs_ccd():
    return np.zeros((2068, 4144))


def make_acs_quadrant():
    return np.zeros((2068, 2072))
