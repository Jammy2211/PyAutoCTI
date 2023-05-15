import autocti as ac
from autocti import exc

import pytest


class TestSettingsCTI1D:
    def test__density_checks_raise_exception(self):
        settings = ac.SettingsCTI1D(total_density_range=(1.0, 2.0))

        traps = [
            ac.TrapInstantCapture(density=0.75),
            ac.TrapInstantCapture(density=0.75),
        ]

        settings.check_total_density_within_range(traps=traps)

        traps = [ac.TrapInstantCapture(density=1.1), ac.TrapInstantCapture(density=1.1)]

        with pytest.raises(exc.PriorException):
            settings.check_total_density_within_range(traps=traps)


class TestSettingsCTI2D:
    def test__parallel_and_serial_checks_raise_exception(self):
        settings = ac.SettingsCTI2D(parallel_total_density_range=(1.0, 2.0))

        parallel_traps = [
            ac.TrapInstantCapture(density=0.75),
            ac.TrapInstantCapture(density=0.75),
        ]
        serial_traps = []

        settings.check_total_density_within_range(
            parallel_traps=parallel_traps, serial_traps=serial_traps
        )

        parallel_traps = [
            ac.TrapInstantCapture(density=1.1),
            ac.TrapInstantCapture(density=1.1),
        ]

        with pytest.raises(exc.PriorException):
            settings.check_total_density_within_range(
                parallel_traps=parallel_traps, serial_traps=serial_traps
            )

        settings = ac.SettingsCTI2D(serial_total_density_range=(1.0, 2.0))

        parallel_traps = []
        serial_traps = [
            ac.TrapInstantCapture(density=0.75),
            ac.TrapInstantCapture(density=0.75),
        ]

        settings.check_total_density_within_range(
            parallel_traps=parallel_traps, serial_traps=serial_traps
        )

        serial_traps = [
            ac.TrapInstantCapture(density=1.1),
            ac.TrapInstantCapture(density=1.1),
        ]

        with pytest.raises(exc.PriorException):
            settings.check_total_density_within_range(
                parallel_traps=parallel_traps, serial_traps=serial_traps
            )
