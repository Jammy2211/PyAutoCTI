from autoconf import conf

from autocti.mask import mask as msk
from autocti.charge_injection import ci_mask as ci_msk
from autocti.charge_injection import ci_imaging


class SettingsPhaseCIImaging:
    def __init__(
        self,
        mask=msk.SettingsMask(),
        ci_mask=ci_msk.SettingsCIMask(),
        masked_ci_imaging=ci_imaging.SettingsMaskedCIImaging(),
        parallel_total_density_range=None,
        serial_total_density_range=None,
    ):

        self.mask = mask
        self.ci_mask = ci_mask
        self.masked_ci_imaging = masked_ci_imaging

        self.parallel_total_density_range = parallel_total_density_range
        self.serial_total_density_range = serial_total_density_range

    @property
    def phase_tag(self):

        return (
            conf.instance.tag.get("phase", "settings")
            + self.parallel_total_density_range_tag
            + self.serial_total_density_range_tag
            + self.mask.mask_tag
            + self.ci_mask.ci_mask_tag
            + self.masked_ci_imaging.masked_ci_imaging_tag
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
            return f"__{conf.instance.tag.get('phase', 'parallel_total_density_range')}_({x0},{x1})"

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
            return f"__{conf.instance.tag.get('phase', 'serial_total_density_range')}_({x0},{x1})"
