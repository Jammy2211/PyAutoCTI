import autofit as af
import autocti as ac
from autocti.pipeline.phase.ci_imaging.phase import PhaseCIImaging
from autocti.analysis import result as res
from autocti.mock import mock

import numpy as np


class TestResult:
    def test__result_contains_instance_with_cti_model(
        self, analysis_ci_imaging_7x7, samples_with_result
    ):

        result = res.Result(
            samples=samples_with_result,
            analysis=analysis_ci_imaging_7x7,
            model=None,
            search=None,
        )

        assert isinstance(result.instance.cti.parallel_traps[0], ac.TrapInstantCapture)
        assert isinstance(result.instance.cti.parallel_ccd, ac.CCD)

    def test__clocker_passed_as_result_correctly(
        self, analysis_ci_imaging_7x7, samples_with_result, parallel_clocker
    ):

        result = res.Result(
            samples=samples_with_result,
            analysis=analysis_ci_imaging_7x7,
            model=None,
            search=None,
        )

        assert isinstance(result.clocker, ac.Clocker)
        assert result.clocker.parallel_express == parallel_clocker.parallel_express


class TestResultDataset:
    def test__masks_available_as_property(
        self,
        analysis_ci_imaging_7x7,
        samples_with_result,
        parallel_clocker,
        traps_x1,
        ccd,
    ):
        model = af.CollectionPriorModel(
            cti=af.Model(ac.CTI, parallel_traps=traps_x1, parallel_ccd=ccd)
        )
        result = res.ResultDataset(
            samples=samples_with_result,
            analysis=analysis_ci_imaging_7x7,
            model=model,
            search=None,
        )

        assert (result.masks[0] == np.full(fill_value=False, shape=(7, 7))).all()
