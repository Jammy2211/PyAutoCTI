import copy
import numpy as np
import pytest

import autofit as af
import autocti as ac
from autocti.charge_injection.model.result import ResultImagingCI


def test__fits_to_extracted_and_full_datasets_available(
    imaging_ci_7x7, mask_2d_7x7_unmasked, parallel_clocker_2d, samples_with_result
):
    imaging_ci_full = copy.deepcopy(imaging_ci_7x7)

    masked_dataset = imaging_ci_7x7.apply_mask(mask=mask_2d_7x7_unmasked)
    masked_dataset = masked_dataset.apply_settings(
        settings=ac.SettingsImagingCI(parallel_pixels=(0, 1))
    )

    analysis = ac.AnalysisImagingCI(
        dataset=masked_dataset,
        clocker=parallel_clocker_2d,
        dataset_full=imaging_ci_full,
    )

    result = ac.ResultImagingCI(
        samples=samples_with_result,
        analysis=analysis,
    )

    assert (
        result.max_log_likelihood_fit.mask == np.full(fill_value=False, shape=(7, 1))
    ).all()

    assert (
        result.max_log_likelihood_full_fit.mask
        == np.full(fill_value=False, shape=(7, 7))
    ).all()


def test__noise_scaling_map_dict_is_list_of_result__are_correct(
    imaging_ci_7x7,
    mask_2d_7x7_unmasked,
    parallel_clocker_2d,
    layout_ci_7x7,
    samples_with_result,
    traps_x1,
    ccd,
):
    noise_scaling_map_dict_list_of_regions_ci = [
        ac.Array2D.ones(shape_native=(7, 7), pixel_scales=1.0)
    ]
    noise_scaling_map_dict_list_of_parallel_eper = [
        ac.Array2D.full(fill_value=2.0, shape_native=(7, 7), pixel_scales=1.0)
    ]
    noise_scaling_map_dict_list_of_serial_eper = [
        ac.Array2D.full(fill_value=3.0, shape_native=(7, 7), pixel_scales=1.0)
    ]
    noise_scaling_map_dict_list_of_serial_overscan_no_eper = [
        ac.Array2D.full(fill_value=4.0, shape_native=(7, 7), pixel_scales=1.0)
    ]

    imaging_ci_7x7.noise_scaling_map_dict = {
        "regions_ci": noise_scaling_map_dict_list_of_regions_ci[0],
        "parallel_eper": noise_scaling_map_dict_list_of_parallel_eper[0],
        "serial_eper": noise_scaling_map_dict_list_of_serial_eper[0],
        "serial_overscan_no_eper": noise_scaling_map_dict_list_of_serial_overscan_no_eper[
            0
        ],
    }

    imaging_ci_full = copy.deepcopy(imaging_ci_7x7)

    masked_imaging_ci_7x7 = imaging_ci_7x7.apply_mask(mask=mask_2d_7x7_unmasked)

    analysis = ac.AnalysisImagingCI(
        dataset=masked_imaging_ci_7x7,
        clocker=parallel_clocker_2d,
        dataset_full=imaging_ci_full,
    )

    fit_analysis = analysis.fit_via_instance_from(
        instance=samples_with_result.max_log_likelihood()
    )

    result = ac.ResultImagingCI(
        samples=samples_with_result,
        analysis=analysis,
    )

    assert result.noise_scaling_map_dict["regions_ci"] == pytest.approx(
        fit_analysis.chi_squared_map_of_regions_ci, 1.0e-2
    )
    assert result.noise_scaling_map_dict["parallel_eper"] == pytest.approx(
        fit_analysis.chi_squared_map_of_parallel_eper, 1.0e-2
    )
    assert result.noise_scaling_map_dict["serial_eper"] == pytest.approx(
        fit_analysis.chi_squared_map_of_serial_eper, 1.0e-2
    )
    assert result.noise_scaling_map_dict["serial_overscan_no_eper"] == pytest.approx(
        fit_analysis.chi_squared_map_of_serial_overscan_no_eper, 1.0e-2
    )

    assert result.noise_scaling_map_dict["regions_ci"][1, 1] == pytest.approx(
        18.168, 1.0e-1
    )
    assert result.noise_scaling_map_dict["parallel_eper"][1, 1] == pytest.approx(
        0.0, 1.0e-4
    )
    assert result.noise_scaling_map_dict["serial_eper"][1, 1] == pytest.approx(
        0.0, 1.0e-4
    )
    assert result.noise_scaling_map_dict["serial_overscan_no_eper"][
        1, 1
    ] == pytest.approx(0.0, 1.0e-4)

    model = af.Collection(
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
            parallel_eper=ac.HyperCINoiseScalar(scale_factor=1.0),
            serial_eper=ac.HyperCINoiseScalar(scale_factor=1.0),
            serial_overscan_no_eper=ac.HyperCINoiseScalar(scale_factor=1.0),
        ),
    )

    instance = model.instance_from_prior_medians()

    fit_analysis = analysis.fit_via_instance_from(instance=instance)

    assert result.noise_scaling_map_dict["regions_ci"] != pytest.approx(
        fit_analysis.chi_squared_map_of_regions_ci, 1.0e-2
    )
    assert result.noise_scaling_map_dict["parallel_eper"] != pytest.approx(
        fit_analysis.chi_squared_map_of_parallel_eper, 1.0e-2
    )
    assert result.noise_scaling_map_dict["serial_eper"] != pytest.approx(
        fit_analysis.chi_squared_map_of_serial_eper, 1.0e-2
    )
    assert result.noise_scaling_map_dict["serial_overscan_no_eper"] != pytest.approx(
        fit_analysis.chi_squared_map_of_serial_overscan_no_eper, 1.0e-2
    )
