import autofit as af
import autocti as ac
from autocti.pipeline.phase.ci_imaging.phase import PhaseCIImaging
from autocti.analysis import result as res
from autocti.mock import mock


class TestResult:
    def test__result_contains_instance_with_cti_model(
        self, analysis_ci_imaging_7x7, samples_with_result
    ):

        model = af.Model(
            ac.CTI, parallel_traps=[ac.TrapInstantCapture], parallel_ccd=ac.CCD
        )

        result = res.Result(
            samples=samples_with_result,
            analysis=analysis_ci_imaging_7x7,
            model=model,
            search=None,
        )

        assert isinstance(result.instance.parallel_traps[0], ac.TrapInstantCapture)
        assert isinstance(result.instance.parallel_ccd, ac.CCD)

    def test__clocker_passed_as_result_correctly(
        self, analysis_ci_imaging_7x7, samples_with_result, parallel_clocker
    ):

        model = af.Model(
            ac.CTI, parallel_traps=[ac.TrapInstantCapture], parallel_ccd=ac.CCD
        )

        result = res.Result(
            samples=samples_with_result,
            analysis=analysis_ci_imaging_7x7,
            model=model,
            search=None,
        )

        assert isinstance(result.clocker, ac.Clocker)
        assert result.clocker.parallel_express == parallel_clocker.parallel_express
