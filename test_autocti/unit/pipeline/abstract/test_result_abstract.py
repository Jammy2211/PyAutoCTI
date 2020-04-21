import os
from os import path
import numpy as np

import pytest

import autofit as af
import arctic as ac
from autocti.pipeline.phase.abstract import AbstractPhase
from autocti.pipeline.phase.ci_imaging import PhaseCIImaging
from test_autocti.mock import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(scope="session", autouse=True)
def do_something():
    print("{}/config/".format(directory))

    af.conf.instance = af.conf.Config("{}/config/".format(directory))


class TestGeneric:
    def test__results_of_phase_are_available_as_properties(
        self, ci_imaging_7x7, mask_7x7
    ):

        phase_dataset_7x7 = PhaseCIImaging(
            parallel_traps=[ac.Trap],
            parallel_ccd_volume=ac.CCDVolume,
            serial_traps=[ac.Trap],
            serial_ccd_volume=ac.CCDVolume,
            non_linear_class=af.MultiNest,
            phase_name="test_phase",
        )

        result = phase_dataset_7x7.run(
            datasets=ci_imaging_7x7, masks=mask_7x7, results=mock_pipeline.MockResults()
        )

        assert isinstance(result, AbstractPhase.Result)
