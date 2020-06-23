from os import path

from autofit.mapper import model
import autocti as ac
import pytest
from autocti import exc
from autocti.pipeline.phase.ci_imaging import PhaseCIImaging
from test_autocti import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestCheckDensity:
    def test__parallel_and_serial_checks_raise_exception(self, ci_imaging_7x7):

        phase_ci_imaging_7x7 = PhaseCIImaging(
            phase_name="test_phase",
            settings=ac.PhaseSettingsCIImaging(parallel_total_density_range=(1.0, 2.0)),
            search=mock.MockSearch(),
        )

        analysis = phase_ci_imaging_7x7.make_analysis(
            datasets=[ci_imaging_7x7], clocker=None
        )

        instance = model.ModelInstance()
        instance.parallel_traps = [ac.Trap(density=0.75), ac.Trap(density=0.75)]
        instance.serial_traps = []

        analysis.check_total_density_within_range(instance=instance)

        instance = model.ModelInstance()
        instance.parallel_traps = [ac.Trap(density=1.1), ac.Trap(density=1.1)]

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(instance=instance)

        phase_ci_imaging_7x7 = PhaseCIImaging(
            phase_name="test_phase",
            settings=ac.PhaseSettingsCIImaging(serial_total_density_range=(1.0, 2.0)),
            search=mock.MockSearch(),
        )

        analysis = phase_ci_imaging_7x7.make_analysis(
            datasets=[ci_imaging_7x7], clocker=None
        )

        instance = model.ModelInstance()
        instance.parallel_traps = []
        instance.serial_traps = [ac.Trap(density=0.75), ac.Trap(density=0.75)]

        analysis.check_total_density_within_range(instance=instance)

        instance = model.ModelInstance()
        instance.serial_traps = [ac.Trap(density=1.1), ac.Trap(density=1.1)]

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(instance=instance)
