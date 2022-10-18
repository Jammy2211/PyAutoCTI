import copy
import pytest
import autofit as af
import autocti as ac

from autofit.non_linear.mock.mock_search import MockSearch
from autocti.charge_injection.model.result import ResultImagingCI


def test__make_result__result_imaging_is_returned(
    imaging_ci_7x7, pre_cti_data_7x7, traps_x1, ccd, parallel_clocker_2d
):

    model = af.CollectionPriorModel(
        cti=af.Model(ac.CTI2D, parallel_trap_list=traps_x1, parallel_ccd=ccd),
        hyper_noise=af.Model(ac.HyperCINoiseCollection),
    )

    analysis = ac.AnalysisImagingCI(dataset=imaging_ci_7x7, clocker=parallel_clocker_2d)

    search = MockSearch(name="test_search")

    result = search.fit(model=model, analysis=analysis)

    assert isinstance(result, ResultImagingCI)


def test__log_likelihood_via_analysis__matches_manual_fit(
    imaging_ci_7x7, pre_cti_data_7x7, traps_x1, ccd, parallel_clocker_2d
):

    model = af.CollectionPriorModel(
        cti=af.Model(ac.CTI2D, parallel_trap_list=traps_x1, parallel_ccd=ccd),
        hyper_noise=af.Model(ac.HyperCINoiseCollection),
    )

    analysis = ac.AnalysisImagingCI(dataset=imaging_ci_7x7, clocker=parallel_clocker_2d)

    instance = model.instance_from_unit_vector([])

    log_likelihood_via_analysis = analysis.log_likelihood_function(instance=instance)

    cti = ac.CTI2D(parallel_trap_list=traps_x1, parallel_ccd=ccd)

    post_cti_data = parallel_clocker_2d.add_cti(data=pre_cti_data_7x7.native, cti=cti)

    fit = ac.FitImagingCI(dataset=analysis.dataset, post_cti_data=post_cti_data)

    assert fit.log_likelihood == log_likelihood_via_analysis


def test__log_likelihood_via_analysis__fast_settings_same_as_default(
    imaging_ci_7x7,
    pre_cti_data_7x7,
    traps_x1,
    ccd,
    parallel_clocker_2d,
    serial_clocker_2d,
    parallel_serial_clocker_2d,
):

    model = af.CollectionPriorModel(
        cti=af.Model(ac.CTI2D, parallel_trap_list=traps_x1, parallel_ccd=ccd),
        hyper_noise=af.Model(ac.HyperCINoiseCollection),
    )

    analysis = ac.AnalysisImagingCI(dataset=imaging_ci_7x7, clocker=parallel_clocker_2d)

    instance = model.instance_from_unit_vector([])

    log_likelihood_via_default = analysis.log_likelihood_function(instance=instance)

    parallel_clocker_2d = copy.copy(parallel_clocker_2d)
    parallel_clocker_2d.parallel_fast_mode = True

    analysis = ac.AnalysisImagingCI(dataset=imaging_ci_7x7, clocker=parallel_clocker_2d)

    log_likelihood_via_fast = analysis.log_likelihood_function(instance=instance)

    assert analysis.preloads.parallel_fast_index_list is not None
    assert analysis.preloads.parallel_fast_column_lists is not None

    assert log_likelihood_via_fast == log_likelihood_via_default

    model = af.CollectionPriorModel(
        cti=af.Model(ac.CTI2D, serial_trap_list=traps_x1, serial_ccd=ccd),
        hyper_noise=af.Model(ac.HyperCINoiseCollection),
    )

    analysis = ac.AnalysisImagingCI(dataset=imaging_ci_7x7, clocker=serial_clocker_2d)

    instance = model.instance_from_unit_vector([])

    log_likelihood_via_default = analysis.log_likelihood_function(instance=instance)

    serial_clocker_2d = copy.copy(serial_clocker_2d)
    serial_clocker_2d.serial_fast_mode = True

    analysis = ac.AnalysisImagingCI(dataset=imaging_ci_7x7, clocker=serial_clocker_2d)

    log_likelihood_via_fast = analysis.log_likelihood_function(instance=instance)

    assert analysis.preloads.serial_fast_index_list is not None
    assert analysis.preloads.serial_fast_row_lists is not None

    assert log_likelihood_via_fast == log_likelihood_via_default

    model = af.CollectionPriorModel(
        cti=af.Model(
            ac.CTI2D,
            parallel_trap_list=traps_x1,
            parallel_ccd=ccd,
            serial_trap_list=traps_x1,
            serial_ccd=ccd,
        ),
        hyper_noise=af.Model(ac.HyperCINoiseCollection),
    )

    analysis = ac.AnalysisImagingCI(
        dataset=imaging_ci_7x7, clocker=parallel_serial_clocker_2d
    )

    instance = model.instance_from_unit_vector([])

    log_likelihood_via_default = analysis.log_likelihood_function(instance=instance)

    parallel_serial_clocker_2d = copy.copy(parallel_serial_clocker_2d)
    parallel_serial_clocker_2d.parallel_fast_mode = True

    analysis = ac.AnalysisImagingCI(
        dataset=imaging_ci_7x7, clocker=parallel_serial_clocker_2d
    )

    log_likelihood_via_fast = analysis.log_likelihood_function(instance=instance)

    assert analysis.preloads.parallel_fast_index_list is not None
    assert analysis.preloads.parallel_fast_column_lists is not None
    assert analysis.preloads.serial_fast_index_list is None
    assert analysis.preloads.serial_fast_row_lists is None

    assert log_likelihood_via_fast == log_likelihood_via_default


def test__full_and_extracted_fits_from_instance_and_imaging_ci(
    imaging_ci_7x7, mask_2d_7x7_unmasked, traps_x1, ccd, parallel_clocker_2d
):

    model = af.CollectionPriorModel(
        cti=af.Model(ac.CTI2D, parallel_trap_list=traps_x1, parallel_ccd=ccd),
        hyper_noise=af.Model(ac.HyperCINoiseCollection),
    )

    masked_imaging_ci = imaging_ci_7x7.apply_mask(mask=mask_2d_7x7_unmasked)
    masked_imaging_ci = masked_imaging_ci.apply_settings(
        settings=ac.SettingsImagingCI(parallel_pixels=(0, 1))
    )

    cti = ac.CTI2D(parallel_trap_list=traps_x1, parallel_ccd=ccd)

    post_cti_data = parallel_clocker_2d.add_cti(
        data=masked_imaging_ci.pre_cti_data, cti=cti
    )

    analysis = ac.AnalysisImagingCI(
        dataset=masked_imaging_ci, clocker=parallel_clocker_2d
    )

    instance = model.instance_from_unit_vector([])

    fit_analysis = analysis.fit_via_instance_from(instance=instance)

    fit = ac.FitImagingCI(dataset=masked_imaging_ci, post_cti_data=post_cti_data)

    assert fit.image.shape == (7, 1)
    assert fit_analysis.log_likelihood == pytest.approx(fit.log_likelihood)

    fit_full_analysis = analysis.fit_full_dataset_via_instance_from(instance=instance)

    fit = ac.FitImagingCI(dataset=imaging_ci_7x7, post_cti_data=post_cti_data)

    assert fit.image.shape == (7, 7)
    assert fit_full_analysis.log_likelihood == pytest.approx(fit.log_likelihood)


def test__extracted_fits_from_instance_and_imaging_ci__include_noise_scaling(
    imaging_ci_7x7,
    mask_2d_7x7_unmasked,
    traps_x1,
    ccd,
    parallel_clocker_2d,
    layout_ci_7x7,
):

    model = af.CollectionPriorModel(
        cti=af.Model(ac.CTI2D, parallel_trap_list=traps_x1, parallel_ccd=ccd),
        hyper_noise=af.Model(
            ac.HyperCINoiseCollection, regions_ci=ac.HyperCINoiseScalar
        ),
    )

    noise_scaling_map_dict_list_of_regions_ci = [
        ac.Array2D.ones(shape_native=(7, 7), pixel_scales=1.0)
    ]

    imaging_ci_7x7.noise_scaling_map_dict = [
        noise_scaling_map_dict_list_of_regions_ci[0]
    ]

    masked_imaging_ci = imaging_ci_7x7.apply_mask(mask=mask_2d_7x7_unmasked)
    masked_imaging_ci = masked_imaging_ci.apply_settings(
        settings=ac.SettingsImagingCI(parallel_pixels=(0, 1))
    )

    analysis = ac.AnalysisImagingCI(
        dataset=masked_imaging_ci, clocker=parallel_clocker_2d
    )

    instance = model.instance_from_prior_medians()

    fit_analysis = analysis.fit_via_instance_from(
        instance=instance, hyper_noise_scale=True
    )

    cti = ac.CTI2D(parallel_trap_list=traps_x1, parallel_ccd=ccd)

    post_cti_data = parallel_clocker_2d.add_cti(
        data=masked_imaging_ci.pre_cti_data, cti=cti
    )

    fit = ac.FitImagingCI(
        dataset=masked_imaging_ci,
        post_cti_data=post_cti_data,
        hyper_noise_scalar_dict=[instance.hyper_noise.regions_ci],
    )

    assert fit.image.shape == (7, 1)
    assert fit.log_likelihood == pytest.approx(fit_analysis.log_likelihood, 1.0e-4)

    fit = ac.FitImagingCI(
        dataset=masked_imaging_ci,
        post_cti_data=post_cti_data,
        hyper_noise_scalar_dict=[ac.HyperCINoiseScalar(scale_factor=0.0)],
    )

    assert fit.log_likelihood != pytest.approx(fit_analysis.log_likelihood, 1.0e-4)

    fit_full_analysis = analysis.fit_full_dataset_via_instance_from(
        instance=instance, hyper_noise_scale=True
    )

    masked_imaging_ci = imaging_ci_7x7.apply_mask(mask=mask_2d_7x7_unmasked)

    fit = ac.FitImagingCI(
        dataset=masked_imaging_ci,
        post_cti_data=post_cti_data,
        hyper_noise_scalar_dict=[instance.hyper_noise.regions_ci],
    )

    assert fit.image.shape == (7, 7)
    assert fit.log_likelihood == pytest.approx(fit_full_analysis.log_likelihood, 1.0e-4)

    fit = ac.FitImagingCI(
        dataset=masked_imaging_ci,
        post_cti_data=post_cti_data,
        hyper_noise_scalar_dict=[ac.HyperCINoiseScalar(scale_factor=0.0)],
    )

    assert fit.log_likelihood != pytest.approx(fit_full_analysis.log_likelihood, 1.0e-4)
