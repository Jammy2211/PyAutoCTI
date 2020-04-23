from os import path

import numpy as np
import pytest

from autocti.pipeline.phase.dataset.result import Result
from autocti.pipeline.phase.ci_imaging import PhaseCIImaging
from test_autocti.mock import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestResult:
    def test__results_of_phase_are_available_as_properties(self, ci_imaging_7x7):

        phase_ci_imaging_7x7 = PhaseCIImaging(
            non_linear_class=mock_pipeline.MockNLO, phase_name="test_phase_2"
        )

        result = phase_ci_imaging_7x7.run(
            datasets=[ci_imaging_7x7], results=mock_pipeline.MockResults()
        )

        assert isinstance(result, Result)
        assert (result.mask == np.full(fill_value=False, shape=(7, 7))).all()

    def test__parallel_phase__noise_scaling_maps_list_of_result__are_correct(
        self, ci_data, clocker
    ):

        phase = ac.PhaseCI(
            parallel_traps=[af.PriorModel(ac.Trap)],
            parallel_ccd_volume=ac.CCDVolume,
            non_linear_class=NLO,
            phase_name="test_phase",
        )

        # The ci_region is [0, 1, 0, 1], therefore by changing the image at 0,0 to 2.0 there will be a residual of 1.0,
        # which for a noise_map entry of 2.0 gives a chi squared of 0.25..

        ci_data.image[0, 0] = 2.0
        ci_data.noise_map[0, 0] = 2.0

        result = phase.run(ci_datas=[ci_data], clocker=clocker)

        assert (
            result.noise_scaling_maps_list_of_ci_regions[0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (
            result.noise_scaling_maps_list_of_parallel_trails[0] == np.zeros((3, 3))
        ).all()
        assert (
            result.noise_scaling_maps_list_of_serial_trails[0] == np.zeros((3, 3))
        ).all()
        assert (
            result.noise_scaling_maps_list_of_serial_overscan_no_trails[0]
            == np.zeros((3, 3))
        ).all()
