from os import path

from autoconf import conf
import pytest
from autocti.pipeline.phase.ci_imaging import PhaseCIImaging
from test_autocti import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(scope="session", autouse=True)
def do_something():
    print("{}/config/".format(directory))

    conf.instance = conf.Config("{}/config/".format(directory))


class TestGeneric:
    def test__clocker_passed_as_result_correctly(
        self, ci_imaging_7x7, parallel_clocker
    ):

        phase_ci_imaging_7x7 = PhaseCIImaging(
            phase_name="test_phase", search=mock.MockSearch()
        )

        result = phase_ci_imaging_7x7.run(
            datasets=[ci_imaging_7x7],
            clocker=parallel_clocker,
            results=mock.MockResults(),
        )

        assert result.clocker == parallel_clocker
