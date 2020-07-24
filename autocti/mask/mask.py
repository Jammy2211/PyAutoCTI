from autoarray.mask import abstract_mask
from autoarray.structures import abstract_structure
from autoarray import exc
from autoarray.structures import region as reg
from autoarray.util import array_util

import numpy as np


class Mask(abstract_mask.AbstractMask):
    @classmethod
    def manual(cls, mask, pixel_scales=None, origin=(0.0, 0.0), invert=False):
        """Create a Mask (see *Mask.__new__*) by inputting the array values in 2D, for example:

        mask=np.array([[False, False],
                       [True, False]])

        mask=[[False, False],
               [True, False]]

        Parameters
        ----------
        mask : np.ndarray or list
            The bool values of the mask input as an ndarray of shape [total_y_pixels, total_x_pixels ]or a list of
            lists.
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        origin : (float, float)
            The origin of the array's mask.
        invert : bool
            If True, the input bools of the mask array are inverted such that previously unmasked entries containing
            *False* become masked entries with *True*, and visa versa.
        """
        if type(mask) is list:
            mask = np.asarray(mask).astype("bool")

        if invert:
            mask = np.invert(mask)

        pixel_scales = abstract_structure.convert_pixel_scales(
            pixel_scales=pixel_scales
        )

        if len(mask.shape) != 2:
            raise exc.MaskException("The input mask is not a two dimensional array")

        return cls(mask=mask, pixel_scales=pixel_scales, origin=origin)

    @classmethod
    def unmasked(cls, shape_2d, pixel_scales=None, origin=(0.0, 0.0), invert=False):
        """Create a mask where all pixels are *False* and therefore unmasked.

        Parameters
        ----------
        mask : np.ndarray or list
            The bool values of the mask input as an ndarray of shape [total_y_pixels, total_x_pixels ]or a list of
            lists.
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        origin : (float, float)
            The origin of the array's mask.
        invert : bool
            If True, the input bools of the mask array are inverted such that previously unmasked entries containing
            *False* become masked entries with *True*, and visa versa.
        """
        return cls.manual(
            mask=np.full(shape=shape_2d, fill_value=False),
            pixel_scales=pixel_scales,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def from_masked_regions(cls, shape_2d, masked_regions):

        mask = cls.unmasked(shape_2d=shape_2d)
        masked_regions = list(
            map(lambda region: reg.Region(region=region), masked_regions)
        )
        for region in masked_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        return mask

    @classmethod
    def from_cosmic_ray_map(
        cls,
        cosmic_ray_map,
        cosmic_ray_parallel_buffer=0,
        cosmic_ray_serial_buffer=0,
        cosmic_ray_diagonal_buffer=0,
    ):
        """
        Create the mask used for CTI Calibration, which is all False unless specific regions are input for masking.

        Parameters
        ----------
        cosmic_ray_map : arrays.Array
            2D arrays flagging where cosmic rays on the image.
        cosmic_ray_parallel_buffer : int
            The number of pixels from each ray pixels are masked in the parallel direction.
        cosmic_ray_serial_buffer : int
            The number of pixels from each ray pixels are masked in the serial direction.
        cosmic_ray_diagonal_buffer : int
            The number of pixels from each ray pixels are masked in the digonal up from the parallel + serial direction.
        """
        mask = cls.unmasked(shape_2d=cosmic_ray_map.shape_2d)

        cosmic_ray_mask = (cosmic_ray_map > 0.0).astype("bool")

        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if cosmic_ray_mask[y, x]:
                    y0, y1 = cosmic_ray_map.parallel_trail_from_y(
                        y=y, dy=cosmic_ray_parallel_buffer
                    )
                    mask[y0:y1, x] = True
                    x0, x1 = cosmic_ray_map.serial_trail_from_x(
                        x=x, dx=cosmic_ray_serial_buffer
                    )
                    mask[y, x0:x1] = True
                    y0, y1 = cosmic_ray_map.parallel_trail_from_y(
                        y=y, dy=cosmic_ray_diagonal_buffer
                    )
                    x0, x1 = cosmic_ray_map.serial_trail_from_x(
                        x=x, dx=cosmic_ray_diagonal_buffer
                    )
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
        file_path : str
            The full path of the fits file.
        hdu : int
            The HDU number in the fits file containing the image image.
        pixel_scales : float or (float, float)
            The arc-second to pixel conversion factor of each pixel.
        """

        if type(pixel_scales) is not tuple:
            if type(pixel_scales) is float or int:
                pixel_scales = (float(pixel_scales), float(pixel_scales))

        mask = cls.manual(
            mask=array_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu),
            pixel_scales=pixel_scales,
            origin=origin,
        )

        if resized_mask_shape is not None:
            mask = mask.resized_mask_from_new_shape(new_shape=resized_mask_shape)

        return mask
