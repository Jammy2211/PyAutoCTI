import numpy as np
from typing import Tuple

import autoarray as aa

from autocti.layout.one_d import Layout1D


class SettingsMask1D:
    def __init__(
        self, fpr_pixels: Tuple[int, int] = None, eper_pixels: Tuple[int, int] = None
    ):

        self.fpr_pixels = fpr_pixels
        self.eper_pixels = eper_pixels


class Mask1D(aa.Mask1D):
    @classmethod
    def masked_fprs_and_epers_from(
        cls,
        mask: "Mask1D",
        layout: Layout1D,
        settings: "SettingsMask1D",
        pixel_scales: aa.type.PixelScales,
    ) -> "Mask1D":

        if settings.fpr_pixels is not None:

            fpr_mask = cls.masked_fpr_from_layout(
                layout=layout, settings=settings, pixel_scales=pixel_scales
            )

            mask = mask + fpr_mask

        if settings.eper_pixels is not None:

            eper_mask = cls.masked_epers_from_layout(
                layout=layout, settings=settings, pixel_scales=pixel_scales
            )

            mask = mask + eper_mask

        return mask

    @classmethod
    def masked_fpr_from_layout(
        cls,
        layout: Layout1D,
        settings: "SettingsMask1D",
        pixel_scales: aa.type.PixelScales,
        invert: bool = False,
    ) -> "Mask1D":

        fpr_regions = layout.extract.fpr.region_list_from(pixels=settings.fpr_pixels)

        mask = np.full(layout.shape_1d, False)

        for region in fpr_regions:
            mask[region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return Mask1D(mask=mask.astype("bool"), pixel_scales=pixel_scales)

    @classmethod
    def masked_epers_from_layout(
        cls,
        layout: Layout1D,
        settings: "SettingsMask1D",
        pixel_scales: aa.type.PixelScales,
        invert: bool = False,
    ) -> "Mask1D":

        eper_regions = layout.extract.eper.region_list_from(pixels=settings.eper_pixels)

        mask = np.full(layout.shape_1d, False)

        for region in eper_regions:
            mask[region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return Mask1D(mask=mask.astype("bool"), pixel_scales=pixel_scales)
