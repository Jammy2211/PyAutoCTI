import autofit as af
import autocti as ac
from autocti.charge_injection.model.result import ResultImagingCI

import numpy as np
import pytest


def test__fits_to_extracted_and_full_datasets_available(
    imaging_ci_7x7, mask_2d_7x7_unmasked, parallel_clocker_2d, samples_with_result
):

    masked_imaging_ci = imaging_ci_7x7.apply_mask(mask=mask_2d_7x7_unmasked)
    masked_imaging_ci = masked_imaging_ci.apply_settings(
        settings=ac.SettingsImagingCI(parallel_pixels=(0, 1))
    )

    analysis = ac.AnalysisImagingCI(
        dataset=masked_imaging_ci, clocker=parallel_clocker_2d
    )

    result = ResultImagingCI(
        samples=samples_with_result, analysis=analysis, model=None, search=None
    )

    assert (
        result.max_log_likelihood_fit.mask == np.full(fill_value=False, shape=(7, 1))
    ).all()

    assert (
        result.max_log_likelihood_full_fit.mask
        == np.full(fill_value=False, shape=(7, 7))
    ).all()


def test__noise_scaling_map_list_is_list_of_result__are_correct(
    imaging_ci_7x7,
    mask_2d_7x7_unmasked,
    parallel_clocker_2d,
    layout_ci_7x7,
    samples_with_result,
    traps_x1,
    ccd,
):

    noise_scaling_map_list_list_of_regions_ci = [
        ac.Array2D.ones(shape_native=(7, 7), pixel_scales=1.0)
    ]
    noise_scaling_map_list_list_of_parallel_epers = [
        ac.Array2D.full(fill_value=2.0, shape_native=(7, 7), pixel_scales=1.0)
    ]
    noise_scaling_map_list_list_of_serial_eper = [
        ac.Array2D.full(fill_value=3.0, shape_native=(7, 7), pixel_scales=1.0)
    ]
    noise_scaling_map_list_list_of_serial_overscan_no_eper = [
        ac.Array2D.full(fill_value=4.0, shape_native=(7, 7), pixel_scales=1.0)
    ]

    imaging_ci_7x7.noise_scaling_map_list = [
        noise_scaling_map_list_list_of_regions_ci[0],
        noise_scaling_map_list_list_of_parallel_epers[0],
        noise_scaling_map_list_list_of_serial_eper[0],
        noise_scaling_map_list_list_of_serial_overscan_no_eper[0],
    ]

    masked_imaging_ci_7x7 = imaging_ci_7x7.apply_mask(mask=mask_2d_7x7_unmasked)

    analysis = ac.AnalysisImagingCI(
        dataset=masked_imaging_ci_7x7, clocker=parallel_clocker_2d
    )

    fit_analysis = analysis.fit_via_instance_from(
        instance=samples_with_result.max_log_likelihood_instance
    )

    result = ResultImagingCI(
        samples=samples_with_result, analysis=analysis, model=None, search=None
    )

    assert result.noise_scaling_map_of_regions_ci == pytest.approx(
        fit_analysis.chi_squared_map_of_regions_ci, 1.0e-2
    )
    assert result.noise_scaling_map_of_parallel_epers == pytest.approx(
        fit_analysis.chi_squared_map_of_parallel_epers, 1.0e-2
    )
    assert result.noise_scaling_map_of_serial_epers == pytest.approx(
        fit_analysis.chi_squared_map_of_serial_epers, 1.0e-2
    )
    assert result.noise_scaling_map_of_serial_overscan_no_trails == pytest.approx(
        fit_analysis.chi_squared_map_of_serial_overscan_no_trails, 1.0e-2
    )

    assert result.noise_scaling_map_of_regions_ci[1, 1] == pytest.approx(18.16, 1.0e-1)
    assert result.noise_scaling_map_of_parallel_epers[1, 1] == pytest.approx(
        0.0, 1.0e-4
    )
    assert result.noise_scaling_map_of_serial_epers[1, 1] == pytest.approx(0.0, 1.0e-4)
    assert result.noise_scaling_map_of_serial_overscan_no_trails[1, 1] == pytest.approx(
        0.0, 1.0e-4
    )

    model = af.CollectionPriorModel(
        cti=af.Model(
            ac.CTI2D,
            parallel_trap_list=traps_x1,
            parallel_ccd=ccd,
            serial_trap_list=traps_x1,
            serial_ccd=ccd,
        ),
        hyper_noise=af.Model(
            ac.HyperCINoiseCollection,
            regions_ci=ac.HyperCINoiseScalar(scale_factor=1.0),
            parallel_epers=ac.HyperCINoiseScalar(scale_factor=1.0),
            serial_eper=ac.HyperCINoiseScalar(scale_factor=1.0),
            serial_overscan_no_eper=ac.HyperCINoiseScalar(scale_factor=1.0),
        ),
    )

    instance = model.instance_from_prior_medians()

    fit_analysis = analysis.fit_via_instance_from(instance=instance)

    assert result.noise_scaling_map_of_regions_ci != pytest.approx(
        fit_analysis.chi_squared_map_of_regions_ci, 1.0e-2
    )
    assert result.noise_scaling_map_of_parallel_epers != pytest.approx(
        fit_analysis.chi_squared_map_of_parallel_epers, 1.0e-2
    )
    assert result.noise_scaling_map_of_serial_epers != pytest.approx(
        fit_analysis.chi_squared_map_of_serial_epers, 1.0e-2
    )
    assert result.noise_scaling_map_of_serial_overscan_no_trails != pytest.approx(
        fit_analysis.chi_squared_map_of_serial_overscan_no_trails, 1.0e-2
    )


def test__noise_scaling_map_list_is_setup_correctly(
    imaging_ci_7x7,
    mask_2d_7x7_unmasked,
    layout_ci_7x7,
    parallel_clocker_2d,
    samples_with_result,
):

    imaging_ci_7x7.cosmic_ray_map = None

    masked_imaging_ci_7x7 = imaging_ci_7x7.apply_mask(mask=mask_2d_7x7_unmasked)

    analysis = ac.AnalysisImagingCI(
        dataset=masked_imaging_ci_7x7, clocker=parallel_clocker_2d
    )

    result = ResultImagingCI(
        samples=samples_with_result, analysis=analysis, model=None, search=None
    )

    assert (
        result.noise_scaling_map_list[0] == result.noise_scaling_map_of_regions_ci
    ).all()

    assert (
        result.noise_scaling_map_list[1] == result.noise_scaling_map_of_parallel_epers
    ).all()

    assert (
        result.noise_scaling_map_list[2] == result.noise_scaling_map_of_serial_epers
    ).all()

    assert (
        result.noise_scaling_map_list[3]
        == result.noise_scaling_map_of_serial_overscan_no_trails
    ).all()
