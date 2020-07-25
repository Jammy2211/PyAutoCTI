import os
import shutil

import numpy as np
import pytest
import autocti as ac
from autocti import exc


test_data_path = "{}/files/array/".format(os.path.dirname(os.path.realpath(__file__)))


class TestManual:
    def test__array__makes_array_without_other_inputs(self):

        arr = ac.Array.manual_2d(array=[[1.0, 2.0], [3.0, 4.0]])

        assert isinstance(arr, ac.Array)
        assert (arr == np.array([[1.0, 2.0], [3.0, 4.0]])).all()

        arr = ac.Array.manual_2d(array=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        assert isinstance(arr, ac.Array)
        assert (arr == np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).all()

        arr = ac.Array.manual_2d(array=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        assert isinstance(arr, ac.Array)
        assert (arr == np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])).all()

    def test__array__makes_array_with_pixel_scale(self):

        arr = ac.Array.manual_2d(array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0)

        assert isinstance(arr, ac.Array)
        assert (arr == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert arr.pixel_scales == (1.0, 1.0)

        arr = ac.Array.manual_2d(
            array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0, origin=(0.0, 1.0)
        )

        assert isinstance(arr, ac.Array)
        assert (arr == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 1.0)

        arr = ac.Array.manual_2d(
            array=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], pixel_scales=(2.0, 3.0)
        )

        assert isinstance(arr, ac.Array)
        assert (arr == np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).all()
        assert arr.pixel_scales == (2.0, 3.0)


class TestManualMask:
    def test__array__makes_array_with_pixel_scale(self):
        mask = ac.Mask.unmasked(shape_2d=(2, 2), pixel_scales=1.0)
        arr = ac.Array.manual_mask(array=[[1.0, 2.0], [3.0, 4.0]], mask=mask)

        assert (arr == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert arr.pixel_scales == (1.0, 1.0)

        mask = ac.Mask.manual(
            mask=[[False, False], [True, False]], pixel_scales=1.0, origin=(0.0, 1.0)
        )
        arr = ac.Array.manual_mask(array=[[1.0, 2.0], [3.0, 4.0]], mask=mask)

        assert (arr == np.array([[1.0, 2.0], [0.0, 4.0]])).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 1.0)

    def test__manual_2d__exception_raised_if_input_array_is_2d_and_not_shape_of_mask(
        self
    ):
        with pytest.raises(exc.ArrayException):
            mask = ac.Mask.unmasked(shape_2d=(2, 2), pixel_scales=1.0)
            ac.Array.manual_mask(array=[[1.0], [3.0]], mask=mask)

        with pytest.raises(exc.ArrayException):
            mask = ac.Mask.unmasked(shape_2d=(2, 2), pixel_scales=1.0)
            ac.Array.manual_mask(array=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], mask=mask)


class TestFull:
    def test__array__makes_array_without_other_inputs(self):

        arr = ac.Array.ones(shape_2d=(2, 2))

        assert isinstance(arr, ac.Array)
        assert (arr == np.array([[1.0, 1.0], [1.0, 1.0]])).all()

        arr = ac.Array.full(fill_value=2.0, shape_2d=(2, 2))

        assert isinstance(arr, ac.Array)
        assert (arr == np.array([[2.0, 2.0], [2.0, 2.0]])).all()

    def test__array__makes_scaled_array_with_pixel_scale(self):

        arr = ac.Array.ones(shape_2d=(2, 2), pixel_scales=1.0)

        assert isinstance(arr, ac.Array)
        assert (arr == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
        assert arr.pixel_scales == (1.0, 1.0)

        arr = ac.Array.full(
            fill_value=2.0, shape_2d=(2, 2), pixel_scales=1.0, origin=(0.0, 1.0)
        )

        assert isinstance(arr, ac.Array)
        assert (arr == np.array([[2.0, 2.0], [2.0, 2.0]])).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 1.0)


class TestOnesZeros:
    def test__array__makes_array_without_other_inputs(self):

        arr = ac.Array.ones(shape_2d=(2, 2))

        assert isinstance(arr, ac.Array)
        assert (arr == np.array([[1.0, 1.0], [1.0, 1.0]])).all()

        arr = ac.Array.zeros(shape_2d=(2, 2))

        assert isinstance(arr, ac.Array)
        assert (arr == np.array([[0.0, 0.0], [0.0, 0.0]])).all()

    def test__array__makes_scaled_array_with_pixel_scale(self):

        arr = ac.Array.ones(shape_2d=(2, 2), pixel_scales=1.0)

        assert isinstance(arr, ac.Array)
        assert (arr == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
        assert arr.pixel_scales == (1.0, 1.0)

        arr = ac.Array.zeros(shape_2d=(2, 2), pixel_scales=1.0, origin=(0.0, 1.0))

        assert isinstance(arr, ac.Array)
        assert (arr == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 1.0)


class TestFromFits:
    def test__array__makes_array_without_other_inputs(self):

        arr = ac.Array.from_fits(file_path=test_data_path + "3x3_ones.fits", hdu=0)

        assert isinstance(arr, ac.Array)
        assert (arr == np.ones((3, 3))).all()

        arr = ac.Array.from_fits(file_path=test_data_path + "4x3_ones.fits", hdu=0)

        assert isinstance(arr, ac.Array)
        assert (arr == np.ones((4, 3))).all()

    def test__array__makes_scaled_array_with_pixel_scale(self):

        arr = ac.Array.from_fits(
            file_path=test_data_path + "3x3_ones.fits", hdu=0, pixel_scales=1.0
        )

        assert isinstance(arr, ac.Array)
        assert (arr == np.ones((3, 3))).all()
        assert arr.pixel_scales == (1.0, 1.0)

        arr = ac.Array.from_fits(
            file_path=test_data_path + "4x3_ones.fits",
            hdu=0,
            pixel_scales=1.0,
            origin=(0.0, 1.0),
        )

        assert isinstance(arr, ac.Array)
        assert (arr == np.ones((4, 3))).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 1.0)


class TestOutputToFits:
    def test__output_to_fits(self):

        arr = ac.Array.from_fits(file_path=test_data_path + "3x3_ones.fits", hdu=0)

        output_data_dir = "{}/files/array/output_test/".format(
            os.path.dirname(os.path.realpath(__file__))
        )
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        arr.output_to_fits(file_path=output_data_dir + "array.fits")

        array_from_out = ac.Array.from_fits(
            file_path=output_data_dir + "array.fits", hdu=0
        )

        assert (array_from_out == np.ones((3, 3))).all()

    def test__output_to_fits__shapes_of_arrays_are_2d(self):

        arr = ac.Array.from_fits(file_path=test_data_path + "3x3_ones.fits", hdu=0)

        output_data_dir = "{}/files/array/output_test/".format(
            os.path.dirname(os.path.realpath(__file__))
        )
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        arr.output_to_fits(file_path=output_data_dir + "array.fits")

        array_from_out = ac.util.array.numpy_array_2d_from_fits(
            file_path=output_data_dir + "array.fits", hdu=0
        )

        assert (array_from_out == np.ones((3, 3))).all()
