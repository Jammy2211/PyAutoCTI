import numpy as np
from autocti.mask.mask_2d import Mask2D


class SettingsMask2DCI:
    def __init__(
        self,
        parallel_front_edge_rows=None,
        parallel_epers_rows=None,
        serial_front_edge_columns=None,
        serial_trails_columns=None,
    ):

        self.parallel_front_edge_rows = parallel_front_edge_rows
        self.parallel_epers_rows = parallel_epers_rows
        self.serial_front_edge_columns = serial_front_edge_columns
        self.serial_trails_columns = serial_trails_columns


class Mask2DCI(Mask2D):
    @classmethod
    def masked_front_edges_and_epers_from_layout(
        cls, mask, layout, settings, pixel_scales
    ):

        if settings.parallel_front_edge_rows is not None:

            parallel_front_edge_mask = cls.masked_parallel_front_edge_from_layout(
                layout=layout, settings=settings, pixel_scales=pixel_scales
            )

            mask = mask + parallel_front_edge_mask

        if settings.parallel_epers_rows is not None:

            parallel_epers_mask = cls.masked_parallel_epers_from_layout(
                layout=layout, settings=settings, pixel_scales=pixel_scales
            )

            mask = mask + parallel_epers_mask

        if settings.serial_front_edge_columns is not None:

            serial_front_edge_mask = cls.masked_serial_front_edge_from_layout(
                layout=layout, settings=settings, pixel_scales=pixel_scales
            )

            mask = mask + serial_front_edge_mask

        if settings.serial_trails_columns is not None:

            serial_trails_mask = cls.masked_serial_trails_from_layout(
                layout=layout, settings=settings, pixel_scales=pixel_scales
            )

            mask = mask + serial_trails_mask

        return mask

    @classmethod
    def masked_parallel_front_edge_from_layout(
        cls, layout, settings, pixel_scales, invert=False
    ):

        front_edge_regions = layout.extractor_parallel_front_edge.region_list_from(
            rows=settings.parallel_front_edge_rows
        )
        mask = np.full(layout.shape_2d, False)

        for region in front_edge_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return Mask2DCI(mask=mask.astype("bool"), pixel_scales=pixel_scales)

    @classmethod
    def masked_parallel_epers_from_layout(
        cls, layout, settings, pixel_scales, invert=False
    ):

        trails_regions = layout.extractor_parallel_epers.region_list_from(
            rows=settings.parallel_epers_rows
        )

        mask = np.full(layout.shape_2d, False)

        for region in trails_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return Mask2DCI(mask=mask.astype("bool"), pixel_scales=pixel_scales)

    @classmethod
    def masked_serial_front_edge_from_layout(
        cls, layout, settings, pixel_scales, invert=False
    ):

        front_edge_regions = layout.extractor_serial_front_edge.region_list_from(
            columns=settings.serial_front_edge_columns
        )
        mask = np.full(layout.shape_2d, False)

        for region in front_edge_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return Mask2DCI(mask=mask.astype("bool"), pixel_scales=pixel_scales)

    @classmethod
    def masked_serial_trails_from_layout(
        cls, layout, settings, pixel_scales, invert=False
    ):

        trails_regions = layout.extractor_serial_trails.region_list_from(
            columns=settings.serial_trails_columns
        )
        mask = np.full(layout.shape_2d, False)

        for region in trails_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return Mask2DCI(mask=mask.astype("bool"), pixel_scales=pixel_scales)
