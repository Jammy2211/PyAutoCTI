import numpy as np
from typing import Tuple

import autoarray as aa

from autoarray import exc

from autocti.extract.settings import SettingsExtract
from autocti.layout.two_d import Layout2D


class SettingsMask2D:
    def __init__(
        self,
        parallel_fpr_pixels: Tuple[int, int] = None,
        parallel_eper_pixels: Tuple[int, int] = None,
        serial_fpr_pixels: Tuple[int, int] = None,
        serial_eper_pixels: Tuple[int, int] = None,
        cosmic_ray_parallel_buffer: int = 10,
        cosmic_ray_serial_buffer: int = 10,
        cosmic_ray_diagonal_buffer: int = 3,
    ):
        self.parallel_fpr_pixels = parallel_fpr_pixels
        self.parallel_eper_pixels = parallel_eper_pixels
        self.serial_fpr_pixels = serial_fpr_pixels
        self.serial_eper_pixels = serial_eper_pixels

        self.cosmic_ray_parallel_buffer = cosmic_ray_parallel_buffer
        self.cosmic_ray_serial_buffer = cosmic_ray_serial_buffer
        self.cosmic_ray_diagonal_buffer = cosmic_ray_diagonal_buffer


class Mask2D(aa.Mask2D):
    @classmethod
    def manual(cls, mask, pixel_scales, origin=(0.0, 0.0), invert=False):
        """
        Create a Mask2D (see *Mask2D.__new__*) by inputting the array values in 2D, for example:

        mask=np.array([[False, False],
                       [True, False]])

        mask=[[False, False],
               [True, False]]

        Parameters
        ----------
        mask
            The bool values of the mask input as an ndarray of shape [total_y_pixels, total_x_pixels ]or a list of
            lists.
        pixel_scales
            The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        invert
            If `True`, the ``bool``'s of the input ``mask`` are inverted, for example `False`'s become `True`
            and visa versa.
        """
        if type(mask) is list:
            mask = np.asarray(mask).astype("bool")

        if invert:
            mask = np.invert(mask)

        pixel_scales = aa.util.geometry.convert_pixel_scales_2d(
            pixel_scales=pixel_scales
        )

        if len(mask.shape) != 2:
            raise exc.MaskException("The input mask is not a two dimensional array")

        return cls(mask=mask, pixel_scales=pixel_scales, origin=origin)

    @classmethod
    def all_false(cls, shape_native, pixel_scales, origin=(0.0, 0.0), invert=False):
        """Create a mask where all pixels are `False` and therefore unmasked.

        Parameters
        ----------
        mask
            The bool values of the mask input as an ndarray of shape [total_y_pixels, total_x_pixels ]or a list of
            lists.
        pixel_scales
            The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        invert
            If `True`, the ``bool``'s of the input ``mask`` are inverted, for example `False`'s become `True`
            and visa versa.
        """
        return cls.manual(
            mask=np.full(shape=shape_native, fill_value=False),
            pixel_scales=pixel_scales,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def from_masked_regions(cls, shape_native, pixel_scales, masked_regions):
        mask = cls.all_false(shape_native=shape_native, pixel_scales=pixel_scales)
        masked_regions = list(
            map(lambda region: aa.Region2D(region=region), masked_regions)
        )
        for region in masked_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        return mask

    @classmethod
    def from_cosmic_ray_map_buffed(cls, cosmic_ray_map, settings=SettingsMask2D()):
        """
        Returns the mask used for CTI Calibration, which is all `False` unless specific regions are input for masking.

        Parameters
        ----------
        cosmic_ray_map : array_2d.Array2D
            2D arrays flagging where cosmic rays on the image.
        cosmic_ray_parallel_buffer
            The number of pixels from each ray pixels are masked in the parallel direction.
        cosmic_ray_serial_buffer
            The number of pixels from each ray pixels are masked in the serial direction.
        cosmic_ray_diagonal_buffer
            The number of pixels from each ray pixels are masked in the digonal up from the parallel + serial direction.
        """
        mask = cls.all_false(
            shape_native=cosmic_ray_map.shape_native,
            pixel_scales=cosmic_ray_map.pixel_scales,
        )

        cosmic_ray_mask = (cosmic_ray_map.native > 0.0).astype("bool")

        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if cosmic_ray_mask[y, x]:
                    x0 = int(x)

                    y0 = y
                    y1 = y + 1 + settings.cosmic_ray_parallel_buffer

                    y1 = mask.shape[0] if y1 > mask.shape[0] else y1

                    mask[y0:y1, x] = True

                    x1 = int(x + 1 + settings.cosmic_ray_serial_buffer)

                    x1 = mask.shape[1] if x1 > mask.shape[1] else x1

                    mask[y, x0:x1] = True

                    y0 = y
                    y1 = y + 1 + settings.cosmic_ray_diagonal_buffer
                    x1 = int(x + 1 + settings.cosmic_ray_diagonal_buffer)

                    y1 = mask.shape[0] if y1 > mask.shape[0] else y1
                    x1 = mask.shape[1] if x1 > mask.shape[1] else x1

                    mask[y0:y1, x0:x1] = True

        return mask

    @classmethod
    def from_fits(
        cls, file_path, pixel_scales, hdu=0, origin=(0.0, 0.0), resized_mask_shape=None
    ):
        """
        Loads the image from a .fits file.

        Parameters
        ----------
        file_path
            The full path of the fits file.
        hdu
            The HDU number in the fits file containing the image image.
        pixel_scales or (float, float)
            The arc-second to pixel conversion factor of each pixel.
        """

        if type(pixel_scales) is not tuple:
            if type(pixel_scales) is float or int:
                pixel_scales = (float(pixel_scales), float(pixel_scales))

        mask = cls.manual(
            mask=aa.util.array_2d.numpy_array_2d_via_fits_from(
                file_path=file_path, hdu=hdu
            ),
            pixel_scales=pixel_scales,
            origin=origin,
        )

        if resized_mask_shape is not None:
            mask = mask.derive_mask.resized_from(new_shape=resized_mask_shape)

        return mask

    @classmethod
    def masked_fpr_and_eper_from(
        cls,
        mask: "Mask2D",
        layout: Layout2D,
        settings: "SettingsMask2D",
        pixel_scales: aa.type.PixelScales,
    ) -> "Mask2D":
        if settings.parallel_fpr_pixels is not None:
            parallel_fpr_mask = cls.masked_parallel_fpr_from(
                layout=layout, settings=settings, pixel_scales=pixel_scales
            )

            mask = mask + parallel_fpr_mask

        if settings.parallel_eper_pixels is not None:
            parallel_eper_mask = cls.masked_parallel_eper_from(
                layout=layout, settings=settings, pixel_scales=pixel_scales
            )

            mask = mask + parallel_eper_mask

        if settings.serial_fpr_pixels is not None:
            serial_fpr_mask = cls.masked_serial_fpr_from(
                layout=layout, settings=settings, pixel_scales=pixel_scales
            )

            mask = mask + serial_fpr_mask

        if settings.serial_eper_pixels is not None:
            serial_eper_mask = cls.masked_serial_eper_from(
                layout=layout, settings=settings, pixel_scales=pixel_scales
            )

            mask = mask + serial_eper_mask

        return mask

    @classmethod
    def masked_parallel_fpr_from(
        cls,
        layout: Layout2D,
        settings: "SettingsMask2D",
        pixel_scales: aa.type.PixelScales,
        invert: bool = False,
    ) -> "Mask2D":
        fpr_regions = layout.extract.parallel_fpr.region_list_from(
            settings=SettingsExtract(pixels=settings.parallel_fpr_pixels)
        )
        mask = np.full(layout.shape_2d, False)

        for region in fpr_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return Mask2D(mask=mask.astype("bool"), pixel_scales=pixel_scales)

    @classmethod
    def masked_parallel_eper_from(
        cls,
        layout: Layout2D,
        settings: "SettingsMask2D",
        pixel_scales: aa.type.PixelScales,
        invert: bool = False,
    ) -> "Mask2D":
        eper_regions = layout.extract.parallel_eper.region_list_from(
            settings=SettingsExtract(pixels=settings.parallel_eper_pixels)
        )

        mask = np.full(layout.shape_2d, False)

        for region in eper_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return Mask2D(mask=mask.astype("bool"), pixel_scales=pixel_scales)

    @classmethod
    def masked_serial_fpr_from(
        cls,
        layout: Layout2D,
        settings: "SettingsMask2D",
        pixel_scales: aa.type.PixelScales,
        invert: bool = False,
    ) -> "Mask2D":
        fpr_regions = layout.extract.serial_fpr.region_list_from(
            settings=SettingsExtract(pixels=settings.serial_fpr_pixels)
        )
        mask = np.full(layout.shape_2d, False)

        for region in fpr_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return Mask2D(mask=mask.astype("bool"), pixel_scales=pixel_scales)

    @classmethod
    def masked_serial_eper_from(
        cls,
        layout: Layout2D,
        settings: "SettingsMask2D",
        pixel_scales: aa.type.PixelScales,
        invert: bool = False,
    ) -> "Mask2D":
        eper_regions = layout.extract.serial_eper.region_list_from(
            settings=SettingsExtract(pixels=settings.serial_eper_pixels)
        )
        mask = np.full(layout.shape_2d, False)

        for region in eper_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return Mask2D(mask=mask.astype("bool"), pixel_scales=pixel_scales)
