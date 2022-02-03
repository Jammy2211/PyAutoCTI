import numpy as np

import autoarray as aa


class SettingsMask1DLine:
    def __init__(self, front_edge_pixels=None, trails_pixels=None):

        self.front_edge_pixels = front_edge_pixels
        self.trails_pixels = trails_pixels


class Mask1DLine(aa.Mask1D):
    @classmethod
    def masked_front_edges_and_epers_from_layout(
        cls, mask, layout, settings, pixel_scales
    ):

        if settings.front_edge_pixels is not None:

            front_edge_mask = cls.masked_front_edge_from_layout(
                layout=layout, settings=settings, pixel_scales=pixel_scales
            )

            mask = mask + front_edge_mask

        if settings.trails_pixels is not None:

            trails_mask = cls.masked_trails_from_layout(
                layout=layout, settings=settings, pixel_scales=pixel_scales
            )

            mask = mask + trails_mask

        return mask

    @classmethod
    def masked_front_edge_from_layout(
        cls, layout, settings, pixel_scales, invert=False
    ):

        front_edge_regions = layout.extractor_front_edge.region_list_from(
            pixels=settings.front_edge_pixels
        )
        mask = np.full(layout.shape_1d, False)

        for region in front_edge_regions:
            mask[region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return Mask1DLine(mask=mask.astype("bool"), pixel_scales=pixel_scales)

    @classmethod
    def masked_trails_from_layout(cls, layout, settings, pixel_scales, invert=False):

        trails_regions = layout.extractor_trails.region_list_from(
            pixels=settings.trails_pixels
        )

        mask = np.full(layout.shape_1d, False)

        for region in trails_regions:
            mask[region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return Mask1DLine(mask=mask.astype("bool"), pixel_scales=pixel_scales)
