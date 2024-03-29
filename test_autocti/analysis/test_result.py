import autocti as ac
from autocti.model import result as res

import numpy as np


def test__result_contains_instance_with_cti_model(
    analysis_imaging_ci_7x7, samples_summary_with_result
):
    result = res.Result(
        samples_summary=samples_summary_with_result,
        analysis=analysis_imaging_ci_7x7,
    )

    assert isinstance(result.instance.cti.parallel_trap_list[0], ac.TrapInstantCapture)
    assert isinstance(result.instance.cti.parallel_ccd, ac.CCDPhase)


def test__clocker_passed_as_result_correctly(
    analysis_imaging_ci_7x7, samples_summary_with_result, parallel_clocker_2d
):
    result = res.Result(
        samples_summary=samples_summary_with_result,
        analysis=analysis_imaging_ci_7x7,
    )

    assert isinstance(result.clocker, ac.Clocker2D)
    assert result.clocker.parallel_express == parallel_clocker_2d.parallel_express


def test__masks_available_as_property(
    analysis_imaging_ci_7x7,
    samples_summary_with_result,
    parallel_clocker_2d,
    traps_x1,
    ccd,
):
    result = res.ResultDataset(
        samples_summary=samples_summary_with_result,
        analysis=analysis_imaging_ci_7x7,
    )

    assert (result.mask == np.full(fill_value=False, shape=(7, 7))).all()
