import numpy as np
import pytest

import autocti as ac
from autocti.charge_injection.fit import hyper_noise_map_from


def test__fit_figure_of_merit(imaging_ci_7x7):

    fit = ac.FitImagingCI(
        dataset=imaging_ci_7x7,
        post_cti_data=imaging_ci_7x7.pre_cti_data,
        hyper_noise_scalar_list=None,
    )

    assert fit.log_likelihood == pytest.approx(-575.11719997, 1e-4)

    hyper_noise_scalar_0 = ac.HyperCINoiseScalar(scale_factor=1.0)
    hyper_noise_scalar_1 = ac.HyperCINoiseScalar(scale_factor=2.0)

    fit = ac.FitImagingCI(
        dataset=imaging_ci_7x7,
        post_cti_data=imaging_ci_7x7.pre_cti_data,
        hyper_noise_scalar_list=[hyper_noise_scalar_0, hyper_noise_scalar_1],
    )

    assert fit.log_likelihood == pytest.approx(-180.877585, 1.0e-4)


def test__hyper_noise_map_from():
    noise_map = ac.Array2D.full(fill_value=2.0, shape_native=(2, 2), pixel_scales=1.0)
    noise_scaling_map_dict = [
        ac.Array2D.manual(array=[[0.0, 0.0], [0.0, 0.0]], pixel_scales=1.0)
    ]
    hyper_noise_scalar_list = [ac.HyperCINoiseScalar(scale_factor=1.0)]

    noise_map = hyper_noise_map_from(
        hyper_noise_scalar_list=hyper_noise_scalar_list,
        noise_map=noise_map,
        noise_scaling_map_dict=noise_scaling_map_dict,
    )

    assert (noise_map.native == (np.array([[2.0, 2.0], [2.0, 2.0]]))).all()

    noise_map = ac.Array2D.full(fill_value=2.0, shape_native=(2, 2), pixel_scales=1.0)
    noise_scaling_map_dict = [
        ac.Array2D.manual(array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0)
    ]
    hyper_noise_scalar_list = [ac.HyperCINoiseScalar(scale_factor=0.0)]

    noise_map = hyper_noise_map_from(
        hyper_noise_scalar_list=hyper_noise_scalar_list,
        noise_map=noise_map,
        noise_scaling_map_dict=noise_scaling_map_dict,
    )

    assert (noise_map.native == (np.array([[2.0, 2.0], [2.0, 2.0]]))).all()

    noise_map = ac.Array2D.full(fill_value=2.0, shape_native=(2, 2), pixel_scales=1.0)
    noise_scaling_map_dict = [
        ac.Array2D.manual(array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0)
    ]
    hyper_noise_scalar_list = [ac.HyperCINoiseScalar(scale_factor=1.0)]

    noise_map = hyper_noise_map_from(
        hyper_noise_scalar_list=hyper_noise_scalar_list,
        noise_map=noise_map,
        noise_scaling_map_dict=noise_scaling_map_dict,
    )

    assert (noise_map.native == (np.array([[3.0, 4.0], [5.0, 6.0]]))).all()

    noise_map = ac.Array2D.full(fill_value=2.0, shape_native=(2, 2), pixel_scales=1.0)
    noise_scaling_map_dict = [
        ac.Array2D.manual(array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0),
        ac.Array2D.manual(array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0),
    ]
    hyper_noise_scalar_list = [
        ac.HyperCINoiseScalar(scale_factor=1.0),
        ac.HyperCINoiseScalar(scale_factor=2.0),
    ]

    noise_map = hyper_noise_map_from(
        hyper_noise_scalar_list=hyper_noise_scalar_list,
        noise_map=noise_map,
        noise_scaling_map_dict=noise_scaling_map_dict,
    )

    assert (noise_map.native == (np.array([[5.0, 8.0], [11.0, 14.0]]))).all()


def test__chi_squared_map_of_regions_ci():

    layout = ac.Layout2DCI(shape_2d=(2, 2), region_list=[(0, 1, 0, 1)])

    image = 3.0 * ac.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)
    noise_map = ac.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)
    pre_cti_data = ac.Array2D.full(
        fill_value=1.0, shape_native=(2, 2), pixel_scales=1.0
    ).native

    imaging = ac.ImagingCI(
        image=image, noise_map=noise_map, pre_cti_data=pre_cti_data, layout=layout
    )

    mask = ac.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0)

    masked_imaging = imaging.apply_mask(mask=mask)

    fit = ac.FitImagingCI(dataset=masked_imaging, post_cti_data=pre_cti_data)

    assert (
        fit.chi_squared_map_of_regions_ci == np.array([[4.0, 0.0], [0.0, 0.0]])
    ).all()


def test__chi_squared_map_of_parallel_non_regions_ci():

    layout = ac.Layout2DCI(
        shape_2d=(2, 2),
        region_list=[(0, 1, 0, 1)],
        serial_prescan=(1, 2, 1, 2),
        serial_overscan=(0, 1, 1, 2),
    )

    image = 3.0 * ac.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)
    noise_map = ac.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)
    pre_cti_data = ac.Array2D.full(
        fill_value=1.0, shape_native=(2, 2), pixel_scales=1.0
    ).native

    imaging = ac.ImagingCI(
        image=image, noise_map=noise_map, pre_cti_data=pre_cti_data, layout=layout
    )

    mask = ac.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0)

    masked_imaging = imaging.apply_mask(mask=mask)

    fit = ac.FitImagingCI(dataset=masked_imaging, post_cti_data=pre_cti_data)

    assert (
        fit.chi_squared_map_of_parallel_epers == np.array([[0.0, 0.0], [4.0, 0.0]])
    ).all()


def test__chi_squared_map_of_serial_epers():

    layout = ac.Layout2DCI(
        shape_2d=(2, 2), region_list=[(0, 2, 0, 1)], serial_overscan=(1, 2, 0, 2)
    )

    image = 3.0 * ac.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)
    noise_map = ac.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)
    pre_cti_data = ac.Array2D.full(
        fill_value=1.0, shape_native=(2, 2), pixel_scales=1.0
    ).native

    imaging = ac.ImagingCI(
        image=image, noise_map=noise_map, pre_cti_data=pre_cti_data, layout=layout
    )

    mask = ac.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0)

    masked_imaging = imaging.apply_mask(mask=mask)

    fit = ac.FitImagingCI(dataset=masked_imaging, post_cti_data=pre_cti_data)

    assert (
        fit.chi_squared_map_of_serial_epers == np.array([[0.0, 4.0], [0.0, 4.0]])
    ).all()


def test__chi_squared_map_of_overscan_above_serial_eper():
    layout = ac.Layout2DCI(
        shape_2d=(2, 2), region_list=[(0, 1, 0, 1)], serial_overscan=(0, 2, 1, 2)
    )

    image = 3.0 * ac.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)
    noise_map = ac.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)
    pre_cti_data = ac.Array2D.full(
        fill_value=1.0, shape_native=(2, 2), pixel_scales=1.0
    ).native

    imaging = ac.ImagingCI(
        image=image, noise_map=noise_map, pre_cti_data=pre_cti_data, layout=layout
    )

    mask = ac.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0)

    masked_imaging = imaging.apply_mask(mask=mask)

    fit = ac.FitImagingCI(dataset=masked_imaging, post_cti_data=pre_cti_data)

    assert (
        fit.chi_squared_map_of_serial_overscan_no_epers
        == np.array([[0.0, 0.0], [0.0, 4.0]])
    ).all()
