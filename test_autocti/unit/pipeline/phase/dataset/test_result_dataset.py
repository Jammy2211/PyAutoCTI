from os import path

import numpy as np
import pytest
from autocti.pipeline.phase.ci_imaging import PhaseCIImaging
from autocti.pipeline.phase.dataset.result import Result
from autocti import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestResult:
    def test__results_of_phase_are_available_as_properties(
        self, ci_imaging_7x7, parallel_clocker, samples_with_result
    ):

        ci_imaging_7x7.cosmic_ray_map = None

        phase_ci_imaging_7x7 = PhaseCIImaging(
            search=mock.MockSearch(name="test_phase_2", samples=samples_with_result)
        )

        result = phase_ci_imaging_7x7.run(
            datasets=[ci_imaging_7x7],
            clocker=parallel_clocker,
            results=mock.MockResults(),
        )

        assert isinstance(result, Result)
        assert (result.masks[0] == np.full(fill_value=False, shape=(7, 7))).all()
