import numpy as np
from autocti.mask.mask_2d import Mask2D


class SettingsMask2DCI:
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


class Mask2DCI(Mask2D):
    @classmethod
    def masked_front_edges_and_trails_from_frame_ci(cls, mask, frame_ci, settings):

        if settings.parallel_front_edge_rows is not None:

            parallel_front_edge_mask = cls.masked_parallel_front_edge_from_frame_ci(
                frame_ci=frame_ci, settings=settings
            )

            mask = mask + parallel_front_edge_mask

        if settings.parallel_trails_rows is not None:

            parallel_trails_mask = cls.masked_parallel_trails_from_frame_ci(
                frame_ci=frame_ci, settings=settings
            )

            mask = mask + parallel_trails_mask

        if settings.serial_front_edge_columns is not None:

            serial_front_edge_mask = cls.masked_serial_front_edge_from_frame_ci(
                frame_ci=frame_ci, settings=settings
            )

            mask = mask + serial_front_edge_mask

        if settings.serial_trails_columns is not None:

            serial_trails_mask = cls.masked_serial_trails_from_frame_ci(
                frame_ci=frame_ci, settings=settings
            )

            mask = mask + serial_trails_mask

        return mask

    @classmethod
    def masked_parallel_front_edge_from_frame_ci(cls, frame_ci, settings, invert=False):

        front_edge_regions = frame_ci.region_list(
            rows=settings.parallel_front_edge_rows
        )
        mask = np.full(frame_ci.shape_native, False)

        for region in front_edge_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return Mask2DCI(mask=mask.astype("bool"), pixel_scales=frame_ci.pixel_scales)

    @classmethod
    def masked_parallel_trails_from_frame_ci(cls, frame_ci, settings, invert=False):

        trails_regions = frame_ci.region_list_from(rows=settings.parallel_trails_rows)
        mask = np.full(frame_ci.shape_native, False)

        for region in trails_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return Mask2DCI(mask=mask.astype("bool"), pixel_scales=frame_ci.pixel_scales)

    @classmethod
    def masked_serial_front_edge_from_frame_ci(cls, frame_ci, settings, invert=False):

        front_edge_regions = frame_ci.serial_front_edge_region_list(
            columns=settings.serial_front_edge_columns
        )
        mask = np.full(frame_ci.shape_native, False)

        for region in front_edge_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return Mask2DCI(mask=mask.astype("bool"), pixel_scales=frame_ci.pixel_scales)

    @classmethod
    def masked_serial_trails_from_frame_ci(cls, frame_ci, settings, invert=False):

        trails_regions = frame_ci.serial_trails_region_list_from(
            columns=settings.serial_trails_columns
        )
        mask = np.full(frame_ci.shape_native, False)

        for region in trails_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return Mask2DCI(mask=mask.astype("bool"), pixel_scales=frame_ci.pixel_scales)
