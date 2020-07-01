import logging

import numpy as np
from autoarray.structures import arrays
from autocti import exc
from autocti.structures import mask as msk
from autocti.util import array_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class Array(arrays.Array):
    @classmethod
    def manual_2d(
        cls, array, pixel_scales=None, sub_size=1, origin=(0.0, 0.0), store_in_1d=False
    ):
        """Create an Array (see *Array.__new__*) by inputting the array values in 1D, for example:

        array=np.array([1.0, 2.0, 3.0, 4.0])

        array=[1.0, 2.0, 3.0, 4.0]

        From 1D input the method cannot determine the 2D shape of the array and its mask, thus the shape_2d must be
        input into this method. The mask is setup as a unmasked *Mask* of shape_2d.

        Parameters
        ----------
        array : np.ndarray or list
            The values of the array input as an ndarray of shape [total_unmasked_pixels*(sub_size**2)] or a list of
            lists.
        shape_2d : (float, float)
            The 2D shape of the mask the array is paired with.
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The origin of the array's mask.
        store_in_1d : bool
            If True, the array is stored in 1D as an ndarray of shape [total_unmasked_pixels]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels].
        """
        arr = arrays.Array.manual_2d(
            array=array,
            pixel_scales=pixel_scales,
            origin=origin,
            store_in_1d=store_in_1d,
        )
        return Array(array=arr, mask=arr.mask, store_in_1d=store_in_1d)

    @classmethod
    def full(
        cls,
        fill_value,
        shape_2d,
        pixel_scales=None,
        sub_size=1,
        origin=(0.0, 0.0),
        store_in_1d=False,
    ):

        return cls.manual_2d(
            array=np.full(fill_value=fill_value, shape=shape_2d),
            pixel_scales=pixel_scales,
            origin=origin,
        )

    @classmethod
    def ones(
        cls,
        shape_2d,
        pixel_scales=None,
        sub_size=1,
        origin=(0.0, 0.0),
        store_in_1d=False,
    ):
        return cls.full(
            fill_value=1.0, shape_2d=shape_2d, pixel_scales=pixel_scales, origin=origin
        )

    @classmethod
    def zeros(
        cls,
        shape_2d,
        pixel_scales=None,
        sub_size=1,
        origin=(0.0, 0.0),
        store_in_1d=False,
    ):
        return cls.full(
            fill_value=0.0, shape_2d=shape_2d, pixel_scales=pixel_scales, origin=origin
        )

    @classmethod
    def from_fits(
        cls,
        file_path,
        hdu=0,
        pixel_scales=None,
        sub_size=1,
        origin=(0.0, 0.0),
        store_in_1d=False,
    ):
        array_2d = array_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu)
        return cls.manual_2d(array=array_2d, pixel_scales=pixel_scales, origin=origin)


class MaskedArray(arrays.MaskedArray):
    @classmethod
    def manual_2d(cls, array, mask):
        arr = arrays.MaskedArray.manual_2d(array=array, mask=mask)
        return MaskedArray(array=arr, mask=mask)

    @classmethod
    def full(cls, fill_value, mask):
        return cls.manual_2d(
            array=np.full(fill_value=fill_value, shape=mask.shape_2d), mask=mask
        )

    @classmethod
    def ones(cls, mask):
        return cls.full(fill_value=1.0, mask=mask)

    @classmethod
    def zeros(cls, mask):
        return cls.full(fill_value=0.0, mask=mask)

    @classmethod
    def from_fits(cls, file_path, hdu, mask):
        array_2d = array_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu)
        return cls.manual_2d(array=array_2d, mask=mask)
