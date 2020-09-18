import numpy as np
from autoconf import conf
from autocti.mask.mask import Mask


class SettingsCIMask:
    def __init__(
        self,
        parallel_front_edge_rows=None,
        parallel_trails_rows=None,
        serial_front_edge_columns=None,
        serial_trails_columns=None,
    ):

        self.parallel_front_edge_rows = parallel_front_edge_rows
        self.parallel_trails_rows = parallel_trails_rows
        self.serial_front_edge_columns = serial_front_edge_columns
        self.serial_trails_columns = serial_trails_columns

    @property
    def tag(self):
        return (
            f"{conf.instance.settings_tag.get('ci_mask', 'ci_mask')}["
            f"{self.parallel_front_edge_rows_tag}"
            f"{self.parallel_trails_rows_tag}"
            f"{self.serial_front_edge_columns_tag}"
            f"{self.serial_trails_columns_tag}]"
        )

    @property
    def parallel_front_edge_rows_tag(self):
        """Generate a parallel_front_edge_rows tag, to customize phase names based on the number of rows in the charge
        injection region at the front edge of the parallel clocking direction are masked during the fit,

        This changes the phase settings folder as follows:

        parallel_front_edge_rows = None -> settings
        parallel_front_edge_rows = (0, 10) -> settings__parallel_front_edge_rows_(0,10)
        parallel_front_edge_rows = (20, 60) -> settings__parallel_front_edge_rows_(20,60)
        """
        if self.parallel_front_edge_rows == None:
            return ""
        else:
            x0 = str(self.parallel_front_edge_rows[0])
            x1 = str(self.parallel_front_edge_rows[1])
            return f"__{conf.instance.settings_tag.get('ci_mask', 'parallel_front_edge_rows')}_({x0},{x1})"

    @property
    def parallel_trails_rows_tag(self):
        """Generate a parallel_trails_rows tag, to customize phase names based on the number of rows in the charge
        injection region in the trails of the parallel clocking direction are masked during the fit,

        This changes the phase settings folder as follows:

        parallel_trails_rows = None -> settings
        parallel_trails_rows = (0, 10) -> settings__parallel_trails_rows_(0,10)
        parallel_trails_rows = (20, 60) -> settings__parallel_trails_rows_(20,60)
        """
        if self.parallel_trails_rows == None:
            return ""
        else:
            x0 = str(self.parallel_trails_rows[0])
            x1 = str(self.parallel_trails_rows[1])
            return f"__{conf.instance.settings_tag.get('ci_mask', 'parallel_trails_rows')}_({x0},{x1})"

    @property
    def serial_front_edge_columns_tag(self):
        """Generate a serial_front_edge_columns tag, to customize phase names based on the number of columns in the
        charge  injection region at the front edge of the serial clocking direction are masked during the fit,

        This changes the phase settings folder as follows:

        serial_front_edge_columns = None -> settings
        serial_front_edge_columns = (0, 10) -> settings__serial_front_edge_columns_(0,10)
        serial_front_edge_columns = (20, 60) -> settings__serial_front_edge_columns_(20,60)
        """
        if self.serial_front_edge_columns == None:
            return ""
        else:
            x0 = str(self.serial_front_edge_columns[0])
            x1 = str(self.serial_front_edge_columns[1])
            return f"__{conf.instance.settings_tag.get('ci_mask', 'serial_front_edge_mask_rows')}_({x0},{x1})"

    @property
    def serial_trails_columns_tag(self):
        """Generate a serial_trails_columns tag, to customize phase names based on the number of columns in the charge
        injection region in the trails of the serial clocking direction are masked during the fit,

        This changes the phase settings folder as follows:

        serial_trails_columns = None -> settings
        serial_trails_columns = (0, 10) -> settings__serial_trails_columns_(0,10)
        serial_trails_columns = (20, 60) -> settings__serial_trails_columns_(20,60)
        """
        if self.serial_trails_columns == None:
            return ""
        else:
            x0 = str(self.serial_trails_columns[0])
            x1 = str(self.serial_trails_columns[1])
            return f"__{conf.instance.settings_tag.get('ci_mask', 'serial_trails_mask_rows')}_({x0},{x1})"


class CIMask(Mask):
    @classmethod
    def masked_front_edges_and_trails_from_ci_frame(cls, mask, ci_frame, settings):

        if settings.parallel_front_edge_rows is not None:

            parallel_front_edge_mask = cls.masked_parallel_front_edge_from_ci_frame(
                ci_frame=ci_frame, settings=settings
            )

            mask = mask + parallel_front_edge_mask

        if settings.parallel_trails_rows is not None:

            parallel_trails_mask = cls.masked_parallel_trails_from_ci_frame(
                ci_frame=ci_frame, settings=settings
            )

            mask = mask + parallel_trails_mask

        if settings.serial_front_edge_columns is not None:

            serial_front_edge_mask = cls.masked_serial_front_edge_from_ci_frame(
                ci_frame=ci_frame, settings=settings
            )

            mask = mask + serial_front_edge_mask

        if settings.serial_trails_columns is not None:

            serial_trails_mask = cls.masked_serial_trails_from_ci_frame(
                ci_frame=ci_frame, settings=settings
            )

            mask = mask + serial_trails_mask

        return mask

    @classmethod
    def masked_parallel_front_edge_from_ci_frame(cls, ci_frame, settings, invert=False):

        front_edge_regions = ci_frame.parallel_front_edge_regions(
            rows=settings.parallel_front_edge_rows
        )
        mask = np.full(ci_frame.shape_2d, False)

        for region in front_edge_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return CIMask(mask=mask.astype("bool"))

    @classmethod
    def masked_parallel_trails_from_ci_frame(cls, ci_frame, settings, invert=False):

        trails_regions = ci_frame.parallel_trails_regions(
            rows=settings.parallel_trails_rows
        )
        mask = np.full(ci_frame.shape_2d, False)

        for region in trails_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return CIMask(mask=mask.astype("bool"))

    @classmethod
    def masked_serial_front_edge_from_ci_frame(cls, ci_frame, settings, invert=False):

        front_edge_regions = ci_frame.serial_front_edge_regions(
            columns=settings.serial_front_edge_columns
        )
        mask = np.full(ci_frame.shape_2d, False)

        for region in front_edge_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return CIMask(mask=mask.astype("bool"))

    @classmethod
    def masked_serial_trails_from_ci_frame(cls, ci_frame, settings, invert=False):

        trails_regions = ci_frame.serial_trails_regions(
            columns=settings.serial_trails_columns
        )
        mask = np.full(ci_frame.shape_2d, False)

        for region in trails_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return CIMask(mask=mask.astype("bool"))
