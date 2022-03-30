import autofit as af
import autocti as ac
from autocti.model import result as res

import numpy as np
import pytest


class TestResult:
    def test__result_contains_instance_with_cti_model(
        self, analysis_imaging_ci_7x7, samples_with_result
    ):

        result = res.Result(
            samples=samples_with_result,
            analysis=analysis_imaging_ci_7x7,
            model=None,
            search=None,
        )

        assert isinstance(
            result.instance.cti.parallel_trap_list[0], ac.TrapInstantCapture
        )
        assert isinstance(result.instance.cti.parallel_ccd, ac.CCDPhase)

    def test__clocker_passed_as_result_correctly(
        self, analysis_imaging_ci_7x7, samples_with_result, parallel_clocker_2d
    ):

        result = res.Result(
            samples=samples_with_result,
            analysis=analysis_imaging_ci_7x7,
            model=None,
            search=None,
        )

        assert isinstance(result.clocker, ac.Clocker2D)
        assert result.clocker.parallel_express == parallel_clocker_2d.parallel_express


class TestResultDataset:
    def test__masks_available_as_property(
        self,
        analysis_imaging_ci_7x7,
        samples_with_result,
        parallel_clocker_2d,
        traps_x1,
        ccd,
    ):
        model = af.CollectionPriorModel(
            cti=af.Model(ac.CTI2D, parallel_traps=traps_x1, parallel_ccd=ccd)
        )
        result = res.ResultDataset(
            samples=samples_with_result,
            analysis=analysis_imaging_ci_7x7,
            model=model,
            search=None,
        )

        assert (result.mask == np.full(fill_value=False, shape=(7, 7))).all()
