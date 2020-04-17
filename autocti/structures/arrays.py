import logging

import numpy as np

from autocti.structures import abstract_structure
from autocti.structures import mask as msk
from autocti.util import array_util

from autocti.util import exc

logging.basicConfig()
logger = logging.getLogger(__name__)


class AbstractArray(abstract_structure.AbstractStructure):

    # noinspection PyUnusedLocal
    def __new__(cls, array, mask, *args, **kwargs):
        """ A hyper array with square-pixels.

        Parameters
        ----------
        array: ndarray
            An array representing image (e.g. an image, noise map, etc.)
        pixel_scales: (float, float)
            The arc-second to pixel conversion factor of each pixel.
        origin : (float, float)
            The arc-second origin of the hyper array's coordinate system.
        """

        obj = super(AbstractArray, cls).__new__(
            cls=cls, structure=array, mask=mask,
        )
        return obj

    def new_with_array(self, array):
        """
        Parameters
        ----------
        array: ndarray
            An ndarray

        Returns
        -------
        new_array: Array
            A new instance of this class that shares all of this instances attributes with a new ndarray.
        """
        arguments = vars(self)
        arguments.update({"array": array})
        return self.__class__(**arguments)

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(AbstractArray, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        class_dict = {}
        for key, value in self.__dict__.items():
            class_dict[key] = value
        new_state = pickled_state[2] + (class_dict,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return pickled_state[0], pickled_state[1], new_state

    # noinspection PyMethodOverriding
    def __setstate__(self, state):

        for key, value in state[-1].items():
            setattr(self, key, value)
        super(AbstractArray, self).__setstate__(state[0:-1])

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __eq__(self, other):
        super_result = super(AbstractArray, self).__eq__(other)
        try:
            return super_result.all()
        except AttributeError:
            return super_result

    def map(self, func):
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                func(y, x)

    def output_to_fits(self, file_path, overwrite=False):

        array_util.numpy_array_2d_to_fits(
            array_2d=self, file_path=file_path, overwrite=overwrite
        )


class Array(AbstractArray):

    @classmethod
    def manual_2d(
        cls, array, pixel_scales=None, origin=(0.0, 0.0),
    ):

        if type(array) is list:
            array = np.asarray(array)

        if type(pixel_scales) is float:
            pixel_scales = (pixel_scales, pixel_scales)

        mask = msk.Mask.unmasked(
            shape_2d=array.shape,
            pixel_scales=pixel_scales,
            origin=origin,
        )

        return Array(array=array, mask=mask)

    @classmethod
    def full(
        cls,
        fill_value,
        shape_2d,
        pixel_scales=None,
        origin=(0.0, 0.0),
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
        origin=(0.0, 0.0),
    ):
        return cls.full(
            fill_value=1.0,
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            origin=origin,
        )

    @classmethod
    def zeros(
        cls,
        shape_2d,
        pixel_scales=None,
        origin=(0.0, 0.0),
    ):
        return cls.full(
            fill_value=0.0,
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            origin=origin,
        )

    @classmethod
    def from_fits(
        cls,
        file_path,
        hdu=0,
        pixel_scales=None,
        origin=(0.0, 0.0),
    ):
        array_2d = array_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu)
        return cls.manual_2d(
            array=array_2d,
            pixel_scales=pixel_scales,
            origin=origin,
        )


class MaskedArray(AbstractArray):

    @classmethod
    def manual_2d(cls, array, mask):

        if type(array) is list:
            array = np.asarray(array)

        if array.shape != mask.shape_2d:
            raise exc.ArrayException(
                "The input array is 2D but not the same dimensions as the sub-mask "
                "(e.g. the mask 2D shape multipled by its sub size."
            )

        return AbstractArray(array=array * np.invert(mask), mask=mask)

    @classmethod
    def full(cls, fill_value, mask):
        return cls.manual_2d(
            array=np.full(fill_value=fill_value, shape=mask.shape_2d),
            mask=mask,
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
