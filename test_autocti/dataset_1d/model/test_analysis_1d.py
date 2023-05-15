import pytest

import autofit as af
import autocti as ac

from autofit.non_linear.mock.mock_search import MockSearch
from autocti.dataset_1d.model.result import ResultDataset1D


def test__make_result__result_line_is_returned(
    dataset_1d_7, pre_cti_data_7, traps_x1, ccd, clocker_1d
):
    model = af.Collection(
        cti=af.Model(ac.CTI1D, trap_list=traps_x1, ccd=ccd),
        hyper_noise=af.Model(ac.HyperCINoiseCollection),
    )

    analysis = ac.AnalysisDataset1D(dataset=dataset_1d_7, clocker=clocker_1d)

    search = MockSearch(name="test_search")

    result = search.fit(model=model, analysis=analysis)

    assert isinstance(result, ResultDataset1D)


def test__log_likelihood_via_analysis__matches_manual_fit(
    dataset_1d_7, pre_cti_data_7, traps_x1, ccd, clocker_1d
):
    model = af.Collection(
        cti=af.Model(ac.CTI1D, trap_list=traps_x1, ccd=ccd),
        hyper_noise=af.Model(ac.HyperCINoiseCollection),
    )

    analysis = ac.AnalysisDataset1D(dataset=dataset_1d_7, clocker=clocker_1d)

    instance = model.instance_from_unit_vector([])

    log_likelihood_via_analysis = analysis.log_likelihood_function(instance=instance)

    cti = ac.CTI1D(trap_list=traps_x1, ccd=ccd)

    post_cti_data = clocker_1d.add_cti(data=pre_cti_data_7.native, cti=cti)

    fit = ac.FitDataset1D(dataset=analysis.dataset, post_cti_data=post_cti_data)

    assert fit.log_likelihood == log_likelihood_via_analysis


def test__extracted_fits_from_instance_and_line_ci(
    dataset_1d_7, mask_1d_7_unmasked, traps_x1, ccd, clocker_1d
):
    model = af.Collection(
        cti=af.Model(ac.CTI1D, trap_list=traps_x1, ccd=ccd),
        hyper_noise=af.Model(ac.HyperCINoiseCollection),
    )

    masked_line_ci = dataset_1d_7.apply_mask(mask=mask_1d_7_unmasked)

    cti = ac.CTI1D(trap_list=traps_x1, ccd=ccd)

    post_cti_data = clocker_1d.add_cti(data=masked_line_ci.pre_cti_data, cti=cti)

    analysis = ac.AnalysisDataset1D(dataset=masked_line_ci, clocker=clocker_1d)

    instance = model.instance_from_unit_vector([])

    fit_analysis = analysis.fit_via_instance_from(instance=instance)

    fit = ac.FitDataset1D(dataset=masked_line_ci, post_cti_data=post_cti_data)

    assert fit.dataset.data.shape == (7,)
    assert fit_analysis.log_likelihood == pytest.approx(fit.log_likelihood)
