import os

import numpy as np
import pytest
import autocti as ac

test_data_path = "{}/files/array/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(name="memoizer")
def make_memoizer():
    return ac.util.array.Memoizer()


class TestMemoizer:
    def test_storing(self, memoizer):
        @memoizer
        def func(arg):
            return "result for {}".format(arg)

        func(1)
        func(2)
        func(1)

        assert memoizer.results == {
            "('arg', 1)": "result for 1",
            "('arg', 2)": "result for 2",
        }
        assert memoizer.calls == 2

    def test_multiple_arguments(self, memoizer):
        @memoizer
        def func(arg1, arg2):
            return arg1 * arg2

        func(1, 2)
        func(2, 1)
        func(1, 2)

        assert memoizer.results == {
            "('arg1', 1), ('arg2', 2)": 2,
            "('arg1', 2), ('arg2', 1)": 2,
        }
        assert memoizer.calls == 2

    def test_key_word_arguments(self, memoizer):
        @memoizer
        def func(arg1=0, arg2=0):
            return arg1 * arg2

        func(arg1=1)
        func(arg2=1)
        func(arg1=1)
        func(arg1=1, arg2=1)

        assert memoizer.results == {
            "('arg1', 1)": 0,
            "('arg2', 1)": 0,
            "('arg1', 1), ('arg2', 1)": 1,
        }
        assert memoizer.calls == 3

    def test_key_word_for_positional(self, memoizer):
        @memoizer
        def func(arg):
            return "result for {}".format(arg)

        func(1)
        func(arg=2)
        func(arg=1)

        assert memoizer.calls == 2

    def test_methods(self, memoizer):
        class Class:
            def __init__(self, value):
                self.value = value

            @memoizer
            def method(self):
                return self.value

        one = Class(1)
        two = Class(2)

        assert one.method() == 1
        assert two.method() == 2


class TestFits:
    def test__numpy_array_1d_from_fits(self):
        arr = ac.util.array.numpy_array_1d_from_fits(
            file_path=test_data_path + "3_ones.fits", hdu=0
        )

        assert (arr == np.ones((3))).all()

    def test__numpy_array_1d_to_fits__output_and_load(self):

        if os.path.exists(test_data_path + "array_out.fits"):
            os.remove(test_data_path + "array_out.fits")

        arr = np.array([10.0, 30.0, 40.0, 92.0, 19.0, 20.0])

        ac.util.array.numpy_array_1d_to_fits(
            arr, file_path=test_data_path + "array_out.fits"
        )

        array_load = ac.util.array.numpy_array_1d_from_fits(
            file_path=test_data_path + "array_out.fits", hdu=0
        )

        assert (arr == array_load).all()

    def test__numpy_array_2d_from_fits(self):
        arr = ac.util.array.numpy_array_2d_from_fits(
            file_path=test_data_path + "3x3_ones.fits", hdu=0
        )

        assert (arr == np.ones((3, 3))).all()

        arr = ac.util.array.numpy_array_2d_from_fits(
            file_path=test_data_path + "4x3_ones.fits", hdu=0
        )

        assert (arr == np.ones((4, 3))).all()

    def test__numpy_array_2d_to_fits__output_and_load(self):

        if os.path.exists(test_data_path + "array_out.fits"):
            os.remove(test_data_path + "array_out.fits")

        arr = np.array([[10.0, 30.0, 40.0], [92.0, 19.0, 20.0]])

        ac.util.array.numpy_array_2d_to_fits(
            arr, file_path=test_data_path + "array_out.fits"
        )

        array_load = ac.util.array.numpy_array_2d_from_fits(
            file_path=test_data_path + "array_out.fits", hdu=0
        )

        assert (arr == array_load).all()
