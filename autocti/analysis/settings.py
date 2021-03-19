from autoconf import conf

from autocti.mask import mask as msk
from autocti.charge_injection import ci_mask as ci_msk
from autocti.charge_injection import ci_imaging
from autocti import exc


class SettingsCTI:
    def __init__(
        self, parallel_total_density_range=None, serial_total_density_range=None
    ):

        self.parallel_total_density_range = parallel_total_density_range
        self.serial_total_density_range = serial_total_density_range

    def check_total_density_within_range(self, parallel_traps, serial_traps):

        if self.parallel_total_density_range is not None:

            total_density = sum([trap.density for trap in parallel_traps])

            if (
                total_density < self.parallel_total_density_range[0]
                or total_density > self.parallel_total_density_range[1]
            ):
                raise exc.PriorException

        if self.serial_total_density_range is not None:

            total_density = sum([trap.density for trap in serial_traps])

            if (
                total_density < self.serial_total_density_range[0]
                or total_density > self.serial_total_density_range[1]
            ):
                raise exc.PriorException
