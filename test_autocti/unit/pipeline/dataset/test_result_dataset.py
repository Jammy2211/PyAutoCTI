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
            non_linear_class=mock_pipeline.MockNLO,
            phase_name="test_phase_2",
        )

        result = phase_ci_imaging_7x7.run(
            datasets=[ci_imaging_7x7], results=mock_pipeline.MockResults()
        )

        assert isinstance(result, Result)
        assert (result.mask == np.full(fill_value=False, shape=(7,7))).all()