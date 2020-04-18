import os

import numpy as np
import shutil
import pytest

from autocti import structures as struct
from autocti.structures import arrays
from autocti import util
from autocti.util import exc

test_data_path = "{}/files/array/".format(os.path.dirname(os.path.realpath(__file__)))


class TestArrayAPI:
    class TestManual:
        def test__array__makes_array_without_other_inputs(self):

            arr = struct.Array.manual_2d(array=[[1.0, 2.0], [3.0, 4.0]])

            assert isinstance(arr, struct.Array)
            assert (arr == np.array([[1.0, 2.0], [3.0, 4.0]])).all()

            arr = struct.Array.manual_2d(array=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

            assert isinstance(arr, struct.Array)
            assert (arr == np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).all()

            arr = struct.Array.manual_2d(array=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

            assert isinstance(arr, struct.Array)
            assert (arr == np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])).all()

        def test__array__makes_array_with_pixel_scale(self):

            arr = struct.Array.manual_2d(
                array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0
            )

            assert isinstance(arr, struct.Array)
            assert (arr == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert arr.pixel_scales == (1.0, 1.0)

            arr = struct.Array.manual_2d(
                array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0, origin=(0.0, 1.0)
            )

            assert isinstance(arr, struct.Array)
            assert (arr == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.origin == (0.0, 1.0)

            arr = struct.Array.manual_2d(
                array=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], pixel_scales=(2.0, 3.0)
            )

            assert isinstance(arr, struct.Array)
            assert (arr == np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).all()
            assert arr.pixel_scales == (2.0, 3.0)

    class TestFull:
        def test__array__makes_array_without_other_inputs(self):

            arr = struct.Array.ones(shape_2d=(2, 2))

            assert isinstance(arr, struct.Array)
            assert (arr == np.array([[1.0, 1.0], [1.0, 1.0]])).all()

            arr = struct.Array.full(fill_value=2.0, shape_2d=(2, 2))

            assert isinstance(arr, struct.Array)
            assert (arr == np.array([[2.0, 2.0], [2.0, 2.0]])).all()

        def test__array__makes_scaled_array_with_pixel_scale(self):

            arr = struct.Array.ones(shape_2d=(2, 2), pixel_scales=1.0)

            assert isinstance(arr, struct.Array)
            assert (arr == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert arr.pixel_scales == (1.0, 1.0)

            arr = struct.Array.full(
                fill_value=2.0, shape_2d=(2, 2), pixel_scales=1.0, origin=(0.0, 1.0)
            )

            assert isinstance(arr, struct.Array)
            assert (arr == np.array([[2.0, 2.0], [2.0, 2.0]])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.origin == (0.0, 1.0)

    class TestOnesZeros:
        def test__array__makes_array_without_other_inputs(self):

            arr = struct.Array.ones(shape_2d=(2, 2))

            assert isinstance(arr, struct.Array)
            assert (arr == np.array([[1.0, 1.0], [1.0, 1.0]])).all()

            arr = struct.Array.zeros(shape_2d=(2, 2))

            assert isinstance(arr, struct.Array)
            assert (arr == np.array([[0.0, 0.0], [0.0, 0.0]])).all()

        def test__array__makes_scaled_array_with_pixel_scale(self):

            arr = struct.Array.ones(shape_2d=(2, 2), pixel_scales=1.0)

            assert isinstance(arr, struct.Array)
            assert (arr == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert arr.pixel_scales == (1.0, 1.0)

            arr = struct.Array.zeros(
                shape_2d=(2, 2), pixel_scales=1.0, origin=(0.0, 1.0)
            )

            assert isinstance(arr, struct.Array)
            assert (arr == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.origin == (0.0, 1.0)

    class TestFromFits:
        def test__array__makes_array_without_other_inputs(self):

            arr = struct.Array.from_fits(
                file_path=test_data_path + "3x3_ones.fits", hdu=0
            )

            assert isinstance(arr, struct.Array)
            assert (arr == np.ones((3, 3))).all()

            arr = struct.Array.from_fits(
                file_path=test_data_path + "4x3_ones.fits", hdu=0
            )

            assert isinstance(arr, struct.Array)
            assert (arr == np.ones((4, 3))).all()

        def test__array__makes_scaled_array_with_pixel_scale(self):

            arr = struct.Array.from_fits(
                file_path=test_data_path + "3x3_ones.fits", hdu=0, pixel_scales=1.0
            )

            assert isinstance(arr, struct.Array)
            assert (arr == np.ones((3, 3))).all()
            assert arr.pixel_scales == (1.0, 1.0)

            arr = struct.Array.from_fits(
                file_path=test_data_path + "4x3_ones.fits",
                hdu=0,
                pixel_scales=1.0,
                origin=(0.0, 1.0),
            )

            assert isinstance(arr, struct.Array)
            assert (arr == np.ones((4, 3))).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.origin == (0.0, 1.0)


class TestMaskedArrayAPI:
    class TestManual:
        def test__array__makes_array_with_pixel_scale(self):

            mask = struct.Mask.unmasked(shape_2d=(2, 2), pixel_scales=1.0)
            arr = struct.MaskedArray.manual_2d(
                array=[[1.0, 2.0], [3.0, 4.0]], mask=mask
            )

            assert isinstance(arr, arrays.AbstractArray)
            assert (arr == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert arr.pixel_scales == (1.0, 1.0)

            mask = struct.Mask.manual(
                mask_2d=[[False, False], [True, False]],
                pixel_scales=1.0,
                origin=(0.0, 1.0),
            )
            arr = struct.MaskedArray.manual_2d(
                array=[[1.0, 2.0], [3.0, 4.0]], mask=mask
            )

            assert isinstance(arr, arrays.AbstractArray)
            assert (arr == np.array([[1.0, 2.0], [0.0, 4.0]])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.origin == (0.0, 1.0)

        def test__manual_2d__exception_raised_if_input_array_is_2d_and_not_shape_of_mask(
            self
        ):

            with pytest.raises(exc.ArrayException):
                mask = struct.Mask.unmasked(shape_2d=(2, 2), pixel_scales=1.0)
                struct.MaskedArray.manual_2d(array=[[1.0], [3.0]], mask=mask)

            with pytest.raises(exc.ArrayException):
                mask = struct.Mask.unmasked(shape_2d=(2, 2), pixel_scales=1.0)
                struct.MaskedArray.manual_2d(
                    array=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], mask=mask
                )

    class TestFull:
        def test__makes_array_using_mask(self):

            mask = struct.Mask.unmasked(shape_2d=(2, 2), pixel_scales=1.0)
            arr = struct.MaskedArray.ones(mask=mask)

            assert isinstance(arr, arrays.AbstractArray)
            assert (arr == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert arr.pixel_scales == (1.0, 1.0)

            mask = struct.Mask.manual(
                mask_2d=[[False, False], [True, False]],
                pixel_scales=1.0,
                origin=(0.0, 1.0),
            )
            arr = struct.MaskedArray.full(fill_value=2.0, mask=mask)

            assert isinstance(arr, arrays.AbstractArray)
            assert (arr == np.array([[2.0, 2.0], [0.0, 2.0]])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.origin == (0.0, 1.0)

    class TestOnesZeros:
        def test__makes_array_using_mask(self):

            mask = struct.Mask.unmasked(shape_2d=(2, 2), pixel_scales=1.0)
            arr = struct.MaskedArray.ones(mask=mask)

            assert isinstance(arr, arrays.AbstractArray)
            assert (arr == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert arr.pixel_scales == (1.0, 1.0)

            mask = struct.Mask.manual(
                mask_2d=[[False, False], [True, False]],
                pixel_scales=1.0,
                origin=(0.0, 1.0),
            )
            arr = struct.MaskedArray.zeros(mask=mask)

            assert isinstance(arr, arrays.AbstractArray)
            assert (arr == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.origin == (0.0, 1.0)

    class TestFromFits:
        def test__array_from_fits_uses_mask(self):

            mask = struct.Mask.unmasked(shape_2d=(3, 3), pixel_scales=1.0)
            arr = struct.MaskedArray.from_fits(
                file_path=test_data_path + "3x3_ones.fits", hdu=0, mask=mask
            )

            assert isinstance(arr, arrays.AbstractArray)
            assert (arr == np.ones((3, 3))).all()
            assert arr.pixel_scales == (1.0, 1.0)

            mask = struct.Mask.manual(
                [
                    [False, False, False],
                    [False, False, False],
                    [True, False, True],
                    [False, False, False],
                ],
                pixel_scales=1.0,
                origin=(0.0, 1.0),
            )
            arr = struct.MaskedArray.from_fits(
                file_path=test_data_path + "4x3_ones.fits", hdu=0, mask=mask
            )

            assert isinstance(arr, arrays.AbstractArray)
            assert (
                arr
                == np.array(
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
                )
            ).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.origin == (0.0, 1.0)

    class TestOutputToFits:
        def test__output_to_fits__shapes_of_arrays_are_2d(self):

            arr = arrays.Array.from_fits(
                file_path=test_data_path + "3x3_ones.fits", hdu=0
            )

            output_data_dir = "{}/files/array/output_test/".format(
                os.path.dirname(os.path.realpath(__file__))
            )
            if os.path.exists(output_data_dir):
                shutil.rmtree(output_data_dir)

            os.makedirs(output_data_dir)

            mask = struct.Mask.unmasked(shape_2d=(3, 3), pixel_scales=0.1)

            masked_array = struct.MaskedArray(array=arr, mask=mask)

            masked_array.output_to_fits(file_path=output_data_dir + "masked_array.fits")

            masked_array_from_out = util.array.numpy_array_2d_from_fits(
                file_path=output_data_dir + "masked_array.fits", hdu=0
            )

            assert (masked_array_from_out == np.ones((3, 3))).all()


class TestOutputToFits:
    def test__output_to_fits(self):

        arr = struct.Array.from_fits(file_path=test_data_path + "3x3_ones.fits", hdu=0)

        output_data_dir = "{}/files/array/output_test/".format(
            os.path.dirname(os.path.realpath(__file__))
        )
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        arr.output_to_fits(file_path=output_data_dir + "array.fits")

        array_from_out = struct.Array.from_fits(
            file_path=output_data_dir + "array.fits", hdu=0
        )

        assert (array_from_out == np.ones((3, 3))).all()

    def test__output_to_fits__shapes_of_arrays_are_2d(self):

        arr = struct.Array.from_fits(file_path=test_data_path + "3x3_ones.fits", hdu=0)

        output_data_dir = "{}/files/array/output_test/".format(
            os.path.dirname(os.path.realpath(__file__))
        )
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        arr.output_to_fits(file_path=output_data_dir + "array.fits")

        array_from_out = util.array.numpy_array_2d_from_fits(
            file_path=output_data_dir + "array.fits", hdu=0
        )

        assert (array_from_out == np.ones((3, 3))).all()
