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

    @property
    def tag(self):
        return (
            self.parallel_total_density_range_tag + self.serial_total_density_range_tag
        )

    @property
    def parallel_total_density_range_tag(self):
        """Generate a parallel_total_density_range tag, to customize phase names based on the range of values in total \
        density that are allowed for the non-linear search in the parallel direction.

        This changes the phase settings folder as follows:

        parallel_total_density_range = None -> settings
        parallel_total_density_range = (0, 10) -> settings__parallel_total_density_range_(0,10)
        parallel_total_density_range = (20, 60) -> settings__parallel_total_density_range_(20,60)
        """
        if self.parallel_total_density_range == None:
            return ""
        else:
            x0 = str(self.parallel_total_density_range[0])
            x1 = str(self.parallel_total_density_range[1])
            return f"__{conf.instance.tag.get('cti', 'parallel_total_density_range')}_({x0},{x1})"

    @property
    def serial_total_density_range_tag(self):
        """Generate a serial_total_density_range tag, to customize phase names based on the range of values in total \
        density that are allowed for the non-linear search in the serial direction.

        This changes the phase settings folder as follows:

        serial_total_density_range = None -> settings
        serial_total_density_range = (0, 10) -> settings__serial_total_density_range_(0,10)
        serial_total_density_range = (20, 60) -> settings__serial_total_density_range_(20,60)
        """
        if self.serial_total_density_range == None:
            return ""
        else:
            x0 = str(self.serial_total_density_range[0])
            x1 = str(self.serial_total_density_range[1])
            return f"__{conf.instance.tag.get('cti', 'serial_total_density_range')}_({x0},{x1})"

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


class SettingsPhaseCIImaging:
    def __init__(
        self,
        cti=SettingsCTI(),
        mask=msk.SettingsMask(),
        ci_mask=ci_msk.SettingsCIMask(),
        masked_ci_imaging=ci_imaging.SettingsMaskedCIImaging(),
    ):

        self.cti = cti
        self.mask = mask
        self.ci_mask = ci_mask
        self.masked_ci_imaging = masked_ci_imaging

    @property
    def phase_tag(self):

        return (
            conf.instance.tag.get("phase", "settings")
            + self.cti.tag
            + self.mask.tag
            + self.ci_mask.tag
            + self.masked_ci_imaging.tag
        )
