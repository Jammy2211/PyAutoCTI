from os import path

import numpy as np
import pytest
import arctic as ac

from autocti.pipeline.phase.ci_imaging import PhaseCIImaging
from test_autocti.mock import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestSetup:

    def test__recognises_type_of_fit_correctly(self):

        phase_dataset_7x7 = PhaseCIImaging(
            phase_name="test_phase",
            parallel_traps=[ac.Trap],
            parallel_ccd_volume=ac.CCDVolume
        )

        assert phase_dataset_7x7.meta_dataset.is_parallel_fit is True
        assert phase_dataset_7x7.meta_dataset.is_serial_fit is False
        assert phase_dataset_7x7.meta_dataset.is_parallel_and_serial_fit is False

        phase_dataset_7x7 = PhaseCIImaging(
            phase_name="test_phase",
            serial_traps=[ac.Trap],
            serial_ccd_volume=ac.CCDVolume
        )

        assert phase_dataset_7x7.meta_dataset.is_parallel_fit is False
        assert phase_dataset_7x7.meta_dataset.is_serial_fit is True
        assert phase_dataset_7x7.meta_dataset.is_parallel_and_serial_fit is False

        phase_dataset_7x7 = PhaseCIImaging(
            phase_name="test_phase",
            parallel_traps=[ac.Trap],
            parallel_ccd_volume=ac.CCDVolume,
            serial_traps=[ac.Trap],
            serial_ccd_volume=ac.CCDVolume,
        )

        assert phase_dataset_7x7.meta_dataset.is_parallel_fit is False
        assert phase_dataset_7x7.meta_dataset.is_serial_fit is False
        assert phase_dataset_7x7.meta_dataset.is_parallel_and_serial_fit is True