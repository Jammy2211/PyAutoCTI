import pytest

import autofit as af
import autocti as ac

from autofit.mock.mock import MockSearch
from autocti.line.model.result import ResultDatasetLine


class TestAnalysisDatasetLine:
    def test__make_result__result_line_is_returned(
        self, dataset_line_7, pre_cti_data_7, traps_x1, ccd, clocker_1d
    ):
        model = af.CollectionPriorModel(
            cti=af.Model(ac.CTI1D, traps=traps_x1, ccd=ccd),
            hyper_noise=af.Model(ac.ci.HyperCINoiseCollection),
        )

        analysis = ac.AnalysisDatasetLine(
            dataset_line=dataset_line_7, clocker=clocker_1d
        )

        search = MockSearch(name="test_search")

        result = search.fit(model=model, analysis=analysis)

        assert isinstance(result, ResultDatasetLine)

    def test__log_likelihood_via_analysis__matches_manual_fit(
        self, dataset_line_7, pre_cti_data_7, traps_x1, ccd, clocker_1d
    ):

        model = af.CollectionPriorModel(
            cti=af.Model(ac.CTI1D, traps=traps_x1, ccd=ccd),
            hyper_noise=af.Model(ac.ci.HyperCINoiseCollection),
        )

        analysis = ac.AnalysisDatasetLine(
            dataset_line=dataset_line_7, clocker=clocker_1d
        )

        instance = model.instance_from_unit_vector([])

        log_likelihood_via_analysis = analysis.log_likelihood_function(
            instance=instance
        )

        post_cti_data = clocker_1d.add_cti(
            data=pre_cti_data_7.native, trap_list=traps_x1, ccd=ccd
        )

        fit = ac.FitDatasetLine(
            dataset_line=analysis.dataset_line, post_cti_data=post_cti_data
        )

        assert fit.log_likelihood == log_likelihood_via_analysis

    def test__extracted_fits_from_instance_and_line_ci(
        self, dataset_line_7, mask_1d_7_unmasked, traps_x1, ccd, clocker_1d
    ):

        model = af.CollectionPriorModel(
            cti=af.Model(ac.CTI1D, traps=traps_x1, ccd=ccd),
            hyper_noise=af.Model(ac.ci.HyperCINoiseCollection),
        )

        masked_line_ci = dataset_line_7.apply_mask(mask=mask_1d_7_unmasked)

        post_cti_data = clocker_1d.add_cti(
            data=masked_line_ci.pre_cti_data, trap_list=traps_x1, ccd=ccd
        )

        analysis = ac.AnalysisDatasetLine(
            dataset_line=masked_line_ci, clocker=clocker_1d
        )

        instance = model.instance_from_unit_vector([])

        fit_analysis = analysis.fit_from_instance(instance=instance)

        fit = ac.FitDatasetLine(
            dataset_line=masked_line_ci, post_cti_data=post_cti_data
        )

        assert fit.dataset.data.shape == (7,)
        assert fit_analysis.log_likelihood == pytest.approx(fit.log_likelihood)
