from os import path

import autocti as ac
import autofit as af
import pytest
from autocti import exc

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestCheckDensity:
    def test__parallel_and_serial_checks_raise_exception(
        self, phase_ci_imaging_7x7, ci_imaging_7x7
    ):

        phase_ci_imaging_7x7.meta_dataset.parallel_total_density_range = (1.0, 2.0)

        analysis = phase_ci_imaging_7x7.make_analysis(
            datasets=[ci_imaging_7x7], clocker=None
        )

        instance = af.ModelInstance()
        instance.parallel_traps = [ac.Trap(density=0.75), ac.Trap(density=0.75)]

        analysis.check_total_density_within_range(instance=instance)

        instance = af.ModelInstance()
        instance.parallel_traps = [ac.Trap(density=1.1), ac.Trap(density=1.1)]

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(instance=instance)

        phase_ci_imaging_7x7.meta_dataset.parallel_total_density_range = None
        phase_ci_imaging_7x7.meta_dataset.serial_total_density_range = (1.0, 2.0)

        analysis = phase_ci_imaging_7x7.make_analysis(
            datasets=[ci_imaging_7x7], clocker=None
        )

        instance = af.ModelInstance()
        instance.serial_traps = [ac.Trap(density=0.75), ac.Trap(density=0.75)]

        analysis.check_total_density_within_range(instance=instance)

        instance = af.ModelInstance()
        instance.serial_traps = [ac.Trap(density=1.1), ac.Trap(density=1.1)]

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(instance=instance)
