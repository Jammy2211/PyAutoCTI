import os
from os import path
import shutil

import numpy as np
import pytest
import autocti as ac
from autocti import exc


test_data_path = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "array"
)


class TestAPI:
    def test__manual__makes_array_with_pixel_scale(self):

        arr = ac.Array2D.manual_native(array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0)

        assert isinstance(arr, ac.Array2D)
        assert (arr == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert arr.pixel_scales == (1.0, 1.0)

        arr = ac.Array2D.manual_native(
            array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0, origin=(0.0, 1.0)
        )

        assert isinstance(arr, ac.Array2D)
        assert (arr == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 1.0)

        arr = ac.Array2D.manual_native(
            array=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], pixel_scales=(2.0, 3.0)
        )

        assert isinstance(arr, ac.Array2D)
        assert (arr == np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).all()
        assert arr.pixel_scales == (2.0, 3.0)

    def test__manual_mask__makes_array_with_pixel_scale(self):
        mask = ac.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0)
        arr = ac.Array2D.manual_mask(array=[[1.0, 2.0], [3.0, 4.0]], mask=mask)

        assert (arr == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert arr.pixel_scales == (1.0, 1.0)

        mask = ac.Mask2D.manual(
            mask=[[False, False], [True, False]], pixel_scales=1.0, origin=(0.0, 1.0)
        )
        arr = ac.Array2D.manual_mask(array=[[1.0, 2.0], [3.0, 4.0]], mask=mask)

        assert (arr == np.array([[1.0, 2.0], [0.0, 4.0]])).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 1.0)

    def test__manual_native__exception_raised_if_input_array_is_2d_and_not_shape_of_mask(
        self,
    ):
        with pytest.raises(exc.ArrayException):
            mask = ac.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0)
            ac.Array2D.manual_mask(array=[[1.0], [3.0]], mask=mask)

        with pytest.raises(exc.ArrayException):
            mask = ac.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0)
            ac.Array2D.manual_mask(
                array=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], mask=mask
            )

    def test__full__makes_scaled_array_with_pixel_scale(self):

        arr = ac.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)

        assert isinstance(arr, ac.Array2D)
        assert (arr == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
        assert arr.pixel_scales == (1.0, 1.0)

        arr = ac.Array2D.full(
            fill_value=2.0, shape_native=(2, 2), pixel_scales=1.0, origin=(0.0, 1.0)
        )

        assert isinstance(arr, ac.Array2D)
        assert (arr == np.array([[2.0, 2.0], [2.0, 2.0]])).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 1.0)

    def test__ones_zeros__makes_scaled_array_with_pixel_scale(self):

        arr = ac.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)

        assert isinstance(arr, ac.Array2D)
        assert (arr == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
        assert arr.pixel_scales == (1.0, 1.0)

        arr = ac.Array2D.zeros(shape_native=(2, 2), pixel_scales=1.0, origin=(0.0, 1.0))

        assert isinstance(arr, ac.Array2D)
        assert (arr == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 1.0)

    def test__from_fits__makes_array_without_other_inputs(self):

        arr = ac.Array2D.from_fits(
            file_path=path.join(test_data_path, "3x3_ones.fits"),
            hdu=0,
            pixel_scales=1.0,
        )

        assert isinstance(arr, ac.Array2D)
        assert (arr == np.ones((3, 3))).all()

        arr = ac.Array2D.from_fits(
            file_path=path.join(test_data_path, "4x3_ones.fits"),
            hdu=0,
            pixel_scales=1.0,
        )

        assert isinstance(arr, ac.Array2D)
        assert (arr == np.ones((4, 3))).all()

    def test__from_fits__makes_scaled_array_with_pixel_scale(self):

        arr = ac.Array2D.from_fits(
            file_path=path.join(test_data_path, "3x3_ones.fits"),
            hdu=0,
            pixel_scales=1.0,
        )

        assert isinstance(arr, ac.Array2D)
        assert (arr == np.ones((3, 3))).all()
        assert arr.pixel_scales == (1.0, 1.0)

        arr = ac.Array2D.from_fits(
            file_path=path.join(test_data_path, "4x3_ones.fits"),
            hdu=0,
            pixel_scales=1.0,
            origin=(0.0, 1.0),
        )

        assert isinstance(arr, ac.Array2D)
        assert (arr == np.ones((4, 3))).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 1.0)

    def test__output_to_fits(self):

        arr = ac.Array2D.from_fits(
            file_path=path.join(test_data_path, "3x3_ones.fits"),
            hdu=0,
            pixel_scales=1.0,
        )

        output_data_dir = path.join(test_data_path, "output_test")

        if path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        arr.output_to_fits(file_path=path.join(output_data_dir, "array.fits"))

        array_from_out = ac.Array2D.from_fits(
            file_path=path.join(output_data_dir, "array.fits"), hdu=0, pixel_scales=1.0
        )

        assert (array_from_out == np.ones((3, 3))).all()

    def test__output_to_fits__shapes_of_arrays_are_2d(self):

        arr = ac.Array2D.from_fits(
            file_path=path.join(test_data_path, "3x3_ones.fits"),
            hdu=0,
            pixel_scales=1.0,
        )

        output_data_dir = path.join(test_data_path, "output_test")

        if path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        arr.output_to_fits(file_path=path.join(output_data_dir, "array.fits"))

        array_from_out = ac.util.array.numpy_array_2d_from_fits(
            file_path=path.join(output_data_dir, "array.fits"), hdu=0
        )

        assert (array_from_out == np.ones((3, 3))).all()
