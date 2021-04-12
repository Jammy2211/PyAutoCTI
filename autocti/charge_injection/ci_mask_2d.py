import numpy as np
from autocti.mask.mask_2d import Mask2D


class SettingsCIMask2D:
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


class CIMask2D(Mask2D):
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
        mask = np.full(ci_frame.shape_native, False)

        for region in front_edge_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return CIMask2D(mask=mask.astype("bool"), pixel_scales=ci_frame.pixel_scales)

    @classmethod
    def masked_parallel_trails_from_ci_frame(cls, ci_frame, settings, invert=False):

        trails_regions = ci_frame.parallel_trails_regions(
            rows=settings.parallel_trails_rows
        )
        mask = np.full(ci_frame.shape_native, False)

        for region in trails_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return CIMask2D(mask=mask.astype("bool"), pixel_scales=ci_frame.pixel_scales)

    @classmethod
    def masked_serial_front_edge_from_ci_frame(cls, ci_frame, settings, invert=False):

        front_edge_regions = ci_frame.serial_front_edge_regions(
            columns=settings.serial_front_edge_columns
        )
        mask = np.full(ci_frame.shape_native, False)

        for region in front_edge_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return CIMask2D(mask=mask.astype("bool"), pixel_scales=ci_frame.pixel_scales)

    @classmethod
    def masked_serial_trails_from_ci_frame(cls, ci_frame, settings, invert=False):

        trails_regions = ci_frame.serial_trails_regions(
            columns=settings.serial_trails_columns
        )
        mask = np.full(ci_frame.shape_native, False)

        for region in trails_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return CIMask2D(mask=mask.astype("bool"), pixel_scales=ci_frame.pixel_scales)
