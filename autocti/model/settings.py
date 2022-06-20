from typing import List, Tuple

try:
    from arcticpy.src.traps import AbstractTrap
except ModuleNotFoundError:
    pass

from autocti import exc


class AbstractSettingsCTI:
    def check_total_density_within_range_of_traps(
        self, total_density_range: Tuple[float, float], traps: List[AbstractTrap]
    ):

        if total_density_range is not None:

            total_density = sum([trap.density for trap in traps])

            if (
                total_density < total_density_range[0]
                or total_density > total_density_range[1]
            ):
                raise exc.PriorException


class SettingsCTI1D(AbstractSettingsCTI):
    def __init__(self, total_density_range=None):
        self.total_density_range = total_density_range

    def check_total_density_within_range(self, traps):

        self.check_total_density_within_range_of_traps(
            total_density_range=self.total_density_range, traps=traps
        )


class SettingsCTI2D(AbstractSettingsCTI):
    def __init__(
        self, parallel_total_density_range=None, serial_total_density_range=None
    ):

        self.parallel_total_density_range = parallel_total_density_range
        self.serial_total_density_range = serial_total_density_range

    def check_total_density_within_range(self, parallel_traps, serial_traps):

        self.check_total_density_within_range_of_traps(
            total_density_range=self.parallel_total_density_range, traps=parallel_traps
        )

        self.check_total_density_within_range_of_traps(
            total_density_range=self.serial_total_density_range, traps=serial_traps
        )
