import logging

import numpy as np
from autoarray.structures.arrays import abstract_array
from autoarray.structures import arrays
from autoarray import exc
from autoarray.util import array_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class AbstractArray2D(abstract_array.AbstractArray2D):
    def __new__(cls, array, mask, *args, **kwargs):
        """An array of values, which are paired to a uniform 2D mask of pixels and sub-pixels. Each entry
        on the array corresponds to a value at the centre of a sub-pixel in an unmasked pixel.

        An *Array2D* is ordered such that pixels begin from the top-row of the corresponding mask and go right and down.
        The positive y-axis is upwards and positive x-axis to the right.

        The array can be stored in 1D or 2D, as detailed below.

        Case 1: [sub-size=1, store_slim = True]:
        -----------------------------------------

        The Array2D is an ndarray of shape [total_unmasked_pixels].

        The first element of the ndarray corresponds to the pixel index, for example:

        - array[3] = the 4th unmasked pixel's value.
        - array[6] = the 7th unmasked pixel's value.

        Below is a visual illustration of a array, where a total of 10 pixels are unmasked and are included in \
        the array.

        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     This is an example mask.Mask2D, where:
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|o|o|x|x|x|x|     x = `True` (Pixel is masked and excluded from the array)
        |x|x|x|o|o|o|o|x|x|x|     o = `False` (Pixel is not masked and included in the array)
        |x|x|x|o|o|o|o|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|

        The mask pixel index's will come out like this (and the direction of scaled values is highlighted
        around the mask.

        pixel_scales = 1.0"

        <--- -ve  x  +ve -->
                                                        y      x
        |x|x|x|x|x|x|x|x|x|x|  ^   array[0] = 0
        |x|x|x|x|x|x|x|x|x|x|  |   array[1] = 1
        |x|x|x|x|x|x|x|x|x|x|  |   array[2] = 2
        |x|x|x|x|0|1|x|x|x|x| +ve  array[3] = 3
        |x|x|x|2|3|4|5|x|x|x|  y   array[4] = 4
        |x|x|x|6|7|8|9|x|x|x| -ve  array[5] = 5
        |x|x|x|x|x|x|x|x|x|x|  |   array[6] = 6
        |x|x|x|x|x|x|x|x|x|x|  |   array[7] = 7
        |x|x|x|x|x|x|x|x|x|x| \/   array[8] = 8
        |x|x|x|x|x|x|x|x|x|x|      array[9] = 9

        Case 2: [sub-size>1, store_slim=True]:
        ------------------

        If the masks's sub size is > 1, the array is defined as a sub-array where each entry corresponds to the values
        at the centre of each sub-pixel of an unmasked pixel.

        The sub-array indexes are ordered such that pixels begin from the first (top-left) sub-pixel in the first
        unmasked pixel. Indexes then go over the sub-pixels in each unmasked pixel, for every unmasked pixel.
        Therefore, the sub-array is an ndarray of shape [total_unmasked_pixels*(sub_array_shape)**2]. For example:

        - array[9] - using a 2x2 sub-array, gives the 3rd unmasked pixel's 2nd sub-pixel value.
        - array[9] - using a 3x3 sub-array, gives the 2nd unmasked pixel's 1st sub-pixel value.
        - array[27] - using a 3x3 sub-array, gives the 4th unmasked pixel's 1st sub-pixel value.

        Below is a visual illustration of a sub array. Indexing of each sub-pixel goes from the top-left corner. In
        contrast to the array above, our illustration below restricts the mask to just 2 pixels, to keep the
        illustration brief.

        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     This is an example mask.Mask2D, where:
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     x = `True` (Pixel is masked and excluded from lens)
        |x|x|x|x|o|o|x|x|x|x|     o = `False` (Pixel is not masked and included in lens)
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|

        Our array with a sub-size looks like it did before:

        pixel_scales = 1.0"

        <--- -ve  x  +ve -->

        |x|x|x|x|x|x|x|x|x|x|  ^
        |x|x|x|x|x|x|x|x|x|x|  |
        |x|x|x|x|x|x|x|x|x|x|  |
        |x|x|x|x|x|x|x|x|x|x| +ve
        |x|x|x|0|1|x|x|x|x|x|  y
        |x|x|x|x|x|x|x|x|x|x| -ve
        |x|x|x|x|x|x|x|x|x|x|  |
        |x|x|x|x|x|x|x|x|x|x|  |
        |x|x|x|x|x|x|x|x|x|x| \/
        |x|x|x|x|x|x|x|x|x|x|

        However, if the sub-size is 2,each unmasked pixel has a set of sub-pixels with values. For example, for pixel 0,
        if *sub_size=2*, it has 4 values on a 2x2 sub-array:

        Pixel 0 - (2x2):

               array[0] = value of first sub-pixel in pixel 0.
        |0|1|  array[1] = value of first sub-pixel in pixel 1.
        |2|3|  array[2] = value of first sub-pixel in pixel 2.
               array[3] = value of first sub-pixel in pixel 3.

        If we used a sub_size of 3, for the first pixel we we would create a 3x3 sub-array:


                 array[0] = value of first sub-pixel in pixel 0.
                 array[1] = value of first sub-pixel in pixel 1.
                 array[2] = value of first sub-pixel in pixel 2.
        |0|1|2|  array[3] = value of first sub-pixel in pixel 3.
        |3|4|5|  array[4] = value of first sub-pixel in pixel 4.
        |6|7|8|  array[5] = value of first sub-pixel in pixel 5.
                 array[6] = value of first sub-pixel in pixel 6.
                 array[7] = value of first sub-pixel in pixel 7.
                 array[8] = value of first sub-pixel in pixel 8.

        Case 3: [sub_size=1 store_slim=False]
        --------------------------------------

        The Array2D has the same properties as Case 1, but is stored as an an ndarray of shape
        [total_y_values, total_x_values].

        All masked entries on the array have values of 0.0.

        For the following example mask:

        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     This is an example mask.Mask2D, where:
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|o|o|x|x|x|x|     x = `True` (Pixel is masked and excluded from the array)
        |x|x|x|o|o|o|o|x|x|x|     o = `False` (Pixel is not masked and included in the array)
        |x|x|x|o|o|o|o|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|

        - array[0,0] = 0.0 (it is masked, thus zero)
        - array[0,0] = 0.0 (it is masked, thus zero)
        - array[3,3] = 0.0 (it is masked, thus zero)
        - array[3,3] = 0.0 (it is masked, thus zero)
        - array[3,4] = 0
        - array[3,4] = -1

        Case 4: [sub_size>1 store_slim=False]
        --------------------------------------

        The properties of this array can be derived by combining Case's 2 and 3 above, whereby the array is stored as
        an ndarray of shape [total_y_values*sub_size, total_x_values*sub_size].

        All sub-pixels in masked pixels have values 0.0.

        Parameters
        ----------
        array : np.ndarray
            The values of the array.
        mask : msk.Mask2D
            The 2D mask associated with the array, defining the pixels each array value is paired with and
            originates from.
        store_slim : bool
            If True, the array is stored in 1D as an ndarray of shape [total_unmasked_pixels]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels].
        """
        if len(array.shape) != 2:
            raise exc.ArrayException(
                "An array input into the arrays.Array2D.__new__ method has store_slim = `True` but"
                "the input shape of the array is not 1."
            )

        obj = array.view(cls)
        obj.mask = mask
        obj.store_slim = False
        return obj


class Array2D(AbstractArray2D):
    @classmethod
    def manual_native(cls, array, pixel_scales, origin=(0.0, 0.0)):
        """Create an Array2D (see *AbstractArray2D.__new__*) by inputting the array values in 2D, for example:

        array=np.ndarray([[1.0, 2.0],
                         [3.0, 4.0]])

        array=[[1.0, 2.0],
              [3.0, 4.0]]

        The 2D shape of the array and its mask are determined from the input array and the mask is setup as an
        unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        array : np.ndarray or list
            The values of the array input as an ndarray of shape [total_y_pixels, total_x_pixel] or a
             list of lists.
        pixel_scales: (float, float) or float
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        """
        arr = arrays.Array2D.manual_native(
            array=array, pixel_scales=pixel_scales, origin=origin, store_slim=False
        )
        return Array2D(array=arr, mask=arr.mask)

    @classmethod
    def manual_mask(cls, array, mask, exposure_info=None):
        """Create a Array2D (see *AbstractArray2D.__new__*) by inputting the array values in 1D or 2D with its mask,
        for example:

        mask = Mask2D([[True, False, False, False])
        array=np.array([1.0, 2.0, 3.0])

        Parameters
        ----------
        array : np.ndarray or list
            The values of the array input as an ndarray of shape [total_unmasked_pixels*(sub_size**2)] or a list of
            lists.
        mask : Mask2D
            The mask whose masked pixels are used to setup the sub-pixel grid.
        store_slim : bool
            If True, the array is stored in 1D as an ndarray of shape [total_unmasked_pixels]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels].
        """
        array = abstract_array.convert_manual_array(
            array=array, mask=mask, store_slim=False
        )
        return Array2D(array=array, mask=mask, exposure_info=exposure_info)

    @classmethod
    def full(cls, fill_value, shape_native, pixel_scales, origin=(0.0, 0.0)):
        """Create a Array2D (see *AbstractArray2D.__new__*) where all values are filled with an input fill value, analogous to
         the method numpy ndarray.full.

        Parameters
        ----------
        fill_value : float
            The value all array elements are filled with.
        shape_native : (float, float)
            The 2D shape of the mask the array is paired with.
        pixel_scales: (float, float) or float
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        """
        return cls.manual_native(
            array=np.full(fill_value=fill_value, shape=shape_native),
            pixel_scales=pixel_scales,
            origin=origin,
        )

    @classmethod
    def ones(cls, shape_native, pixel_scales, origin=(0.0, 0.0)):
        """Create an Array2D (see *AbstractArray2D.__new__*) where all values are filled with ones, analogous to the method
        numpy ndarray.ones.

        Parameters
        ----------
        shape_native : (float, float)
            The 2D shape of the mask the array is paired with.
        pixel_scales: (float, float) or float
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        """
        return cls.full(
            fill_value=1.0,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
        )

    @classmethod
    def zeros(cls, shape_native, pixel_scales, origin=(0.0, 0.0)):
        """Create an Array2D (see *AbstractArray2D.__new__*) where all values are filled with zeros, analogous to the method numpy
        ndarray.ones.

        Parameters
        ----------
        shape_native : (float, float)
            The 2D shape of the mask the array is paired with.
        pixel_scales: (float, float) or float
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        """
        return cls.full(
            fill_value=0.0,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
        )

    @classmethod
    def from_fits(cls, file_path, pixel_scales, hdu=0, origin=(0.0, 0.0)):
        """Create an Array2D (see *AbstractArray2D.__new__*) by loading the array values from a .fits file.

        Parameters
        ----------
        file_path : str
            The path the file is output to, including the filename and the ``.fits`` extension,
            e.g. '/path/to/filename.fits'
        hdu : int
            The Header-Data Unit of the .fits file the array data is loaded from.
        pixel_scales: (float, float) or float
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        origin : (float, float)
            The (y,x) scaled units origin of the mask's coordinate system.
        """
        array_2d = array_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu)
        return cls.manual_native(
            array=array_2d, pixel_scales=pixel_scales, origin=origin
        )
