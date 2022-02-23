import numpy as np
from typing import Tuple

import autoarray as aa

from autocti.mask.mask_2d import Mask2D


class SettingsMask2DCI:
    def __init__(
        self,
        parallel_fpr_pixels: Tuple[int, int] = None,
        parallel_epers_pixels: Tuple[int, int] = None,
        serial_fpr_pixels: Tuple[int, int] = None,
        serial_eper_pixels: Tuple[int, int] = None,
    ):

        self.parallel_fpr_pixels = parallel_fpr_pixels
        self.parallel_epers_pixels = parallel_epers_pixels
        self.serial_fpr_pixels = serial_fpr_pixels
        self.serial_eper_pixels = serial_eper_pixels


class Mask2DCI(Mask2D):
    @classmethod
    def masked_fprs_and_epers_from(
        cls,
        mask: "Mask2D",
        layout: "Layout2DCI",
        settings: "SettingsMask2DCI",
        pixel_scales: aa.type.PixelScales,
    ) -> "Mask2DCI":

        if settings.parallel_fpr_pixels is not None:

            parallel_fpr_mask = cls.masked_parallel_fpr_from(
                layout=layout, settings=settings, pixel_scales=pixel_scales
            )

            mask = mask + parallel_fpr_mask

        if settings.parallel_epers_pixels is not None:

            parallel_epers_mask = cls.masked_parallel_epers_from(
                layout=layout, settings=settings, pixel_scales=pixel_scales
            )

            mask = mask + parallel_epers_mask

        if settings.serial_fpr_pixels is not None:

            serial_fpr_mask = cls.masked_serial_fpr_from(
                layout=layout, settings=settings, pixel_scales=pixel_scales
            )

            mask = mask + serial_fpr_mask

        if settings.serial_eper_pixels is not None:

            serial_eper_mask = cls.masked_serial_epers_from(
                layout=layout, settings=settings, pixel_scales=pixel_scales
            )

            mask = mask + serial_eper_mask

        return mask

    @classmethod
    def masked_parallel_fpr_from(
        cls,
        layout: "Layout2DCI",
        settings: "SettingsMask2DCI",
        pixel_scales: aa.type.PixelScales,
        invert: bool = False,
    ) -> "Mask2DCI":

        fpr_regions = layout.extract_parallel_fpr.region_list_from(
            pixels=settings.parallel_fpr_pixels
        )
        mask = np.full(layout.shape_2d, False)

        for region in fpr_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return Mask2DCI(mask=mask.astype("bool"), pixel_scales=pixel_scales)

    @classmethod
    def masked_parallel_epers_from(
        cls,
        layout: "Layout2DCI",
        settings: "SettingsMask2DCI",
        pixel_scales: aa.type.PixelScales,
        invert: bool = False,
    ) -> "Mask2DCI":

        eper_regions = layout.extract_parallel_eper.region_list_from(
            pixels=settings.parallel_epers_pixels
        )

        mask = np.full(layout.shape_2d, False)

        for region in eper_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return Mask2DCI(mask=mask.astype("bool"), pixel_scales=pixel_scales)

    @classmethod
    def masked_serial_fpr_from(
        cls,
        layout: "Layout2DCI",
        settings: "SettingsMask2DCI",
        pixel_scales: aa.type.PixelScales,
        invert: bool = False,
    ) -> "Mask2DCI":

        fpr_regions = layout.extract_serial_fpr.region_list_from(
            pixels=settings.serial_fpr_pixels
        )
        mask = np.full(layout.shape_2d, False)

        for region in fpr_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return Mask2DCI(mask=mask.astype("bool"), pixel_scales=pixel_scales)

    @classmethod
    def masked_serial_epers_from(
        cls,
        layout: "Layout2DCI",
        settings: "SettingsMask2DCI",
        pixel_scales: aa.type.PixelScales,
        invert: bool = False,
    ) -> "Mask2DCI":

        eper_regions = layout.extract_serial_eper.region_list_from(
            pixels=settings.serial_eper_pixels
        )
        mask = np.full(layout.shape_2d, False)

        for region in eper_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return Mask2DCI(mask=mask.astype("bool"), pixel_scales=pixel_scales)
