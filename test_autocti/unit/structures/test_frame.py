import os

import numpy as np
import pytest

from autocti import exc
import autocti as ac
from autocti.structures import frame


@pytest.fixture(scope="function")
def frame_data():
    frame_data = np.array([[9, 0, 0], [1, 1, 14], [25, -6, 2]])

    return frame_data


@pytest.fixture(scope="class")
def image_data():
    image_data = np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]])

    return image_data


@pytest.fixture(scope="class")
def euclid_data():
    euclid_data = np.zeros((2048, 2066))
    return euclid_data


path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))


class TestFrameAPI:
    def test__frame__makes_frame_using_inputs(self):

        frame = ac.frame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            corner=(0, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(2, 3, 2, 3),
        )

        assert (frame == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert (frame.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert (frame.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert frame.corner == (0, 0)
        assert frame.parallel_overscan == (0, 1, 0, 1)
        assert frame.serial_prescan == (1, 2, 1, 2)
        assert frame.serial_overscan == (2, 3, 2, 3)
        assert (frame.mask == np.array([[False, False], [False, False]])).all()

    def test__euclid_frame_for_four_quandrants__loads_data_and_dimensions(
        self, euclid_data
    ):

        euclid_frame = ac.euclid_frame.top_left(array=euclid_data)

        assert euclid_frame.corner == (0, 0)
        assert euclid_frame.shape_2d == (2048, 2066)
        assert (euclid_frame.in_2d == np.zeros((2048, 2066))).all()
        assert euclid_frame.parallel_overscan == (0, 20, 51, 2099)
        assert euclid_frame.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.serial_overscan == (0, 2086, 2099, 2119)

        euclid_frame = ac.euclid_frame.top_right(array=euclid_data)

        assert euclid_frame.corner == (0, 1)
        assert euclid_frame.shape_2d == (2048, 2066)
        assert (euclid_frame.in_2d == np.zeros((2048, 2066))).all()
        assert euclid_frame.parallel_overscan == (0, 20, 20, 2068)
        assert euclid_frame.serial_prescan == (0, 2086, 2068, 2119)
        assert euclid_frame.serial_overscan == (0, 2086, 0, 20)

        euclid_frame = ac.euclid_frame.bottom_left(array=euclid_data)

        assert euclid_frame.corner == (1, 0)
        assert euclid_frame.shape_2d == (2048, 2066)
        assert (euclid_frame.in_2d == np.zeros((2048, 2066))).all()
        assert euclid_frame.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_frame.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.serial_overscan == (0, 2086, 2099, 2119)

        euclid_frame = ac.euclid_frame.bottom_right(array=euclid_data)

        assert euclid_frame.corner == (1, 1)
        assert euclid_frame.shape_2d == (2048, 2066)
        assert (euclid_frame.in_2d == np.zeros((2048, 2066))).all()
        assert euclid_frame.parallel_overscan == (2066, 2086, 20, 2068)
        assert euclid_frame.serial_prescan == (0, 2086, 2068, 2119)
        assert euclid_frame.serial_overscan == (0, 2086, 0, 20)

    def test__left_side__chooses_correct_frame_given_input(self, frame_data):
        frame = ac.euclid_frame.ccd_and_quadrant_id(
            array=frame_data, ccd_id="text1", quad_id="E"
        )

        assert frame.corner == (1, 0)

        frame = ac.euclid_frame.ccd_and_quadrant_id(
            array=frame_data, ccd_id="text2", quad_id="E"
        )

        assert frame.corner == (1, 0)

        frame = ac.euclid_frame.ccd_and_quadrant_id(
            array=frame_data, ccd_id="text3", quad_id="E"
        )

        assert frame.corner == (1, 0)

        frame = ac.euclid_frame.ccd_and_quadrant_id(
            array=frame_data, ccd_id="text1", quad_id="F"
        )

        assert frame.corner == (1, 1)

        frame = ac.euclid_frame.ccd_and_quadrant_id(
            array=frame_data, ccd_id="text2", quad_id="F"
        )

        assert frame.corner == (1, 1)

        frame = ac.euclid_frame.ccd_and_quadrant_id(
            array=frame_data, ccd_id="text3", quad_id="F"
        )

        assert frame.corner == (1, 1)

        frame = ac.euclid_frame.ccd_and_quadrant_id(
            array=frame_data, ccd_id="text1", quad_id="G"
        )

        assert frame.corner == (0, 1)

        frame = ac.euclid_frame.ccd_and_quadrant_id(
            array=frame_data, ccd_id="text2", quad_id="G"
        )

        assert frame.corner == (0, 1)

        frame = ac.euclid_frame.ccd_and_quadrant_id(
            array=frame_data, ccd_id="text3", quad_id="G"
        )

        assert frame.corner == (0, 1)

        frame = ac.euclid_frame.ccd_and_quadrant_id(
            array=frame_data, ccd_id="text1", quad_id="H"
        )

        assert frame.corner == (0, 0)

        frame = ac.euclid_frame.ccd_and_quadrant_id(
            array=frame_data, ccd_id="text2", quad_id="H"
        )

        assert frame.corner == (0, 0)

        frame = ac.euclid_frame.ccd_and_quadrant_id(
            array=frame_data, ccd_id="text3", quad_id="H"
        )

        assert frame.corner == (0, 0)

    def test__right_side__chooses_correct_frame_given_input(self, frame_data):
        frame = ac.euclid_frame.ccd_and_quadrant_id(
            array=frame_data, ccd_id="text4", quad_id="E"
        )

        assert frame.corner == (0, 1)

        frame = ac.euclid_frame.ccd_and_quadrant_id(
            array=frame_data, ccd_id="text5", quad_id="E"
        )

        assert frame.corner == (0, 1)

        frame = ac.euclid_frame.ccd_and_quadrant_id(
            array=frame_data, ccd_id="text6", quad_id="E"
        )

        assert frame.corner == (0, 1)

        frame = ac.euclid_frame.ccd_and_quadrant_id(
            array=frame_data, ccd_id="text4", quad_id="F"
        )

        assert frame.corner == (0, 0)

        frame = ac.euclid_frame.ccd_and_quadrant_id(
            array=frame_data, ccd_id="text5", quad_id="F"
        )

        assert frame.corner == (0, 0)

        frame = ac.euclid_frame.ccd_and_quadrant_id(
            array=frame_data, ccd_id="text6", quad_id="F"
        )

        assert frame.corner == (0, 0)

        frame = ac.euclid_frame.ccd_and_quadrant_id(
            array=frame_data, ccd_id="text4", quad_id="G"
        )

        assert frame.corner == (1, 0)

        frame = ac.euclid_frame.ccd_and_quadrant_id(
            array=frame_data, ccd_id="text5", quad_id="G"
        )

        assert frame.corner == (1, 0)

        frame = ac.euclid_frame.ccd_and_quadrant_id(
            array=frame_data, ccd_id="text6", quad_id="G"
        )

        assert frame.corner == (1, 0)

        frame = ac.euclid_frame.ccd_and_quadrant_id(
            array=frame_data, ccd_id="text4", quad_id="H"
        )

        assert frame.corner == (1, 1)

        frame = ac.euclid_frame.ccd_and_quadrant_id(
            array=frame_data, ccd_id="text5", quad_id="H"
        )

        assert frame.corner == (1, 1)

        frame = ac.euclid_frame.ccd_and_quadrant_id(
            array=frame_data, ccd_id="text6", quad_id="H"
        )

        assert frame.corner == (1, 1)


class TestMaskedFrameAPI:
    def test__frame__makes_frame_using_inputs(self):

        mask = ac.mask.manual(mask_2d=[[False, True], [False, False]])

        frame = ac.masked.frame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            mask=mask,
            corner=(0, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(2, 3, 2, 3),
        )

        assert (frame == np.array([[1.0, 0.0], [3.0, 4.0]])).all()
        assert (frame.in_2d == np.array([[1.0, 0.0], [3.0, 4.0]])).all()
        assert (frame.in_1d == np.array([1.0, 3.0, 4.0])).all()
        assert frame.corner == (0, 0)
        assert frame.parallel_overscan == (0, 1, 0, 1)
        assert frame.serial_prescan == (1, 2, 1, 2)
        assert frame.serial_overscan == (2, 3, 2, 3)
        assert (frame.mask == np.array([[False, True], [False, False]])).all()

    def test__euclid_frame_for_four_quandrants__loads_data_and_dimensions(
        self, euclid_data
    ):

        mask = np.full(shape=(2048, 2066), fill_value=False)
        mask[0, 0] = True

        euclid_frame = ac.masked.euclid_frame.top_left(array=euclid_data, mask=mask)

        assert euclid_frame.corner == (0, 0)
        assert euclid_frame.shape_2d == (2048, 2066)
        assert (euclid_frame.in_2d == np.zeros((2048, 2066))).all()
        assert euclid_frame.parallel_overscan == (0, 20, 51, 2099)
        assert euclid_frame.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.serial_overscan == (0, 2086, 2099, 2119)
        assert euclid_frame.mask[0, 0] == True
        assert euclid_frame.mask[0, 1] == False

        euclid_frame = ac.masked.euclid_frame.top_right(array=euclid_data, mask=mask)

        assert euclid_frame.corner == (0, 1)
        assert euclid_frame.shape_2d == (2048, 2066)
        assert (euclid_frame.in_2d == np.zeros((2048, 2066))).all()
        assert euclid_frame.parallel_overscan == (0, 20, 20, 2068)
        assert euclid_frame.serial_prescan == (0, 2086, 2068, 2119)
        assert euclid_frame.serial_overscan == (0, 2086, 0, 20)
        assert euclid_frame.mask[0, 0] == True
        assert euclid_frame.mask[0, 1] == False

        euclid_frame = ac.masked.euclid_frame.bottom_left(array=euclid_data, mask=mask)

        assert euclid_frame.corner == (1, 0)
        assert euclid_frame.shape_2d == (2048, 2066)
        assert (euclid_frame.in_2d == np.zeros((2048, 2066))).all()
        assert euclid_frame.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_frame.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.serial_overscan == (0, 2086, 2099, 2119)
        assert euclid_frame.mask[0, 0] == True
        assert euclid_frame.mask[0, 1] == False

        euclid_frame = ac.masked.euclid_frame.bottom_right(array=euclid_data, mask=mask)

        assert euclid_frame.corner == (1, 1)
        assert euclid_frame.shape_2d == (2048, 2066)
        assert (euclid_frame.in_2d == np.zeros((2048, 2066))).all()
        assert euclid_frame.parallel_overscan == (2066, 2086, 20, 2068)
        assert euclid_frame.serial_prescan == (0, 2086, 2068, 2119)
        assert euclid_frame.serial_overscan == (0, 2086, 0, 20)
        assert euclid_frame.mask[0, 0] == True
        assert euclid_frame.mask[0, 1] == False

    def test__left_side__chooses_correct_frame_given_input(self, euclid_data):

        mask = np.full(shape=(2048, 2066), fill_value=False)
        mask[0, 0] = True

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text1", quad_id="E"
        )

        assert frame.corner == (1, 0)
        assert frame.mask[0, 0] == True
        assert frame.mask[0, 1] == False

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text2", quad_id="E"
        )

        assert frame.corner == (1, 0)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text3", quad_id="E"
        )

        assert frame.corner == (1, 0)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text1", quad_id="F"
        )

        assert frame.corner == (1, 1)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text2", quad_id="F"
        )

        assert frame.corner == (1, 1)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text3", quad_id="F"
        )

        assert frame.corner == (1, 1)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text1", quad_id="G"
        )

        assert frame.corner == (0, 1)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text2", quad_id="G"
        )

        assert frame.corner == (0, 1)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text3", quad_id="G"
        )

        assert frame.corner == (0, 1)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text1", quad_id="H"
        )

        assert frame.corner == (0, 0)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text2", quad_id="H"
        )

        assert frame.corner == (0, 0)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text3", quad_id="H"
        )

        assert frame.corner == (0, 0)

    def test__right_side__chooses_correct_frame_given_input(self, euclid_data):

        mask = np.full(shape=(2048, 2066), fill_value=False)
        mask[0, 0] = True

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text4", quad_id="E"
        )

        assert frame.corner == (0, 1)
        assert frame.mask[0, 0] == True
        assert frame.mask[0, 1] == False

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text5", quad_id="E"
        )

        assert frame.corner == (0, 1)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text6", quad_id="E"
        )

        assert frame.corner == (0, 1)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text4", quad_id="F"
        )

        assert frame.corner == (0, 0)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text5", quad_id="F"
        )

        assert frame.corner == (0, 0)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text6", quad_id="F"
        )

        assert frame.corner == (0, 0)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text4", quad_id="G"
        )

        assert frame.corner == (1, 0)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text5", quad_id="G"
        )

        assert frame.corner == (1, 0)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text6", quad_id="G"
        )

        assert frame.corner == (1, 0)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text4", quad_id="H"
        )

        assert frame.corner == (1, 1)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text5", quad_id="H"
        )

        assert frame.corner == (1, 1)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text6", quad_id="H"
        )

        assert frame.corner == (1, 1)


class TestBinnedAcross:
    def test__parallel__different_arrays__gives_frame_binned(self):

        frame = ac.frame.manual(array=np.ones((3, 3)))

        assert (frame.binned_across_parallel == np.array([1.0, 1.0, 1.0])).all()

        frame = ac.frame.manual(array=np.ones((4, 3)))

        assert (frame.binned_across_parallel == np.array([1.0, 1.0, 1.0])).all()

        frame = ac.frame.manual(array=np.ones((3, 4)))

        assert (frame.binned_across_parallel == np.array([1.0, 1.0, 1.0, 1.0])).all()

        frame = ac.frame.manual(
            array=np.array([[1.0, 6.0, 9.0], [2.0, 6.0, 9.0], [3.0, 6.0, 9.0]])
        )

        assert (frame.binned_across_parallel == np.array([2.0, 6.0, 9.0])).all()

    def test__parallel__same_as_above_but_with_mask(self):

        mask = np.ma.array(
            [[False, False, False], [False, False, False], [True, False, False]]
        )

        frame = ac.masked.frame.manual(
            array=np.array([[1.0, 6.0, 9.0], [2.0, 6.0, 9.0], [3.0, 6.0, 9.0]]),
            mask=mask,
        )

        assert (frame.binned_across_parallel == np.array([1.5, 6.0, 9.0])).all()

    def test__serial__different_arrays__gives_frame_binned(self):

        frame = ac.frame.manual(array=np.ones((3, 3)))

        assert (frame.binned_across_serial == np.array([1.0, 1.0, 1.0])).all()

        frame = ac.frame.manual(array=np.ones((4, 3)))

        assert (frame.binned_across_serial == np.array([1.0, 1.0, 1.0, 1.0])).all()

        frame = ac.frame.manual(array=np.ones((3, 4)))

        assert (frame.binned_across_serial == np.array([1.0, 1.0, 1.0])).all()

        frame = ac.frame.manual(
            array=np.array([[1.0, 2.0, 3.0], [6.0, 6.0, 6.0], [9.0, 9.0, 9.0]])
        )

        assert (frame.binned_across_serial == np.array([2.0, 6.0, 9.0])).all()

    def test__serial__same_as_above_but_with_mask(self):

        mask = np.ma.array(
            [[False, False, True], [False, False, False], [False, False, False]]
        )

        frame = ac.masked.frame.manual(
            array=np.array([[1.0, 2.0, 3.0], [6.0, 6.0, 6.0], [9.0, 9.0, 9.0]]),
            mask=mask,
        )

        assert (frame.binned_across_serial == np.array([1.5, 6.0, 9.0])).all()


class TestRegion(object):
    class TestConstructor:
        def test__constructor__converts_region_to_cartesians(self):
            region = ac.Region(region=(0, 1, 2, 3))

            assert region == (0, 1, 2, 3)

            assert region.y0 == 0
            assert region.y1 == 1
            assert region.x0 == 2
            assert region.x1 == 3
            assert region.total_rows == 1
            assert region.total_columns == 1

        def test__first_row_or_column_equal_too_or_bigger_than_second__raise_errors(
            self
        ):
            with pytest.raises(exc.RegionException):
                ac.Region(region=(2, 2, 1, 2))

            with pytest.raises(exc.RegionException):
                ac.Region(region=(2, 1, 2, 2))

            with pytest.raises(exc.RegionException):
                ac.Region(region=(2, 1, 1, 2))

            with pytest.raises(exc.RegionException):
                ac.Region(region=(0, 1, 3, 2))

        def test__negative_coordinates_raise_errors(self):
            with pytest.raises(exc.RegionException):
                ac.Region(region=(-1, 0, 1, 2))

            with pytest.raises(exc.RegionException):
                ac.Region(region=(0, -1, 1, 2))

            with pytest.raises(exc.RegionException):
                ac.Region(region=(0, 0, -1, 2))

            with pytest.raises(exc.RegionException):
                ac.Region(region=(0, 1, 2, -1))

    class TestExtractRegionFromArray:
        def test__extracts_2x2_region_of_3x3_array(self):
            frame = ac.frame.manual(
                array=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
            )

            region = ac.Region(region=(0, 2, 0, 2))

            new_frame = frame[region.slice]

            assert (new_frame == np.array([[1.0, 2.0], [4.0, 5.0]])).all()

        def test__extracts_2x3_region_of_3x3_array(self):
            frame = ac.frame.manual(
                array=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
            )

            region = ac.Region(region=(1, 3, 0, 3))

            new_frame = frame[region.slice]

            assert (new_frame == np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])).all()

    class TestAddRegionToArrayFromImage:
        def test__array_is_all_zeros__image_goes_into_correct_region_of_array(self):
            frame = ac.frame.manual(array=np.zeros((2, 2)))
            image = np.ones((2, 2))
            region = ac.Region(region=(0, 1, 0, 1))

            frame[region.slice] += image[region.slice]

            assert (frame == np.array([[1.0, 0.0], [0.0, 0.0]])).all()

        def test__array_is_all_1s__image_goes_into_correct_region_of_array_and_adds_to_it(
            self
        ):
            frame = ac.frame.manual(array=np.ones((2, 2)))
            image = np.ones((2, 2))
            region = ac.Region(region=(0, 1, 0, 1))

            frame[region.slice] += image[region.slice]

            assert (frame == np.array([[2.0, 1.0], [1.0, 1.0]])).all()

        def test__different_region(self):
            frame = ac.frame.manual(array=np.ones((3, 3)))
            image = np.ones((3, 3))
            region = ac.Region(region=(1, 3, 2, 3))

            frame[region.slice] += image[region.slice]

            assert (
                frame == np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 2.0], [1.0, 1.0, 2.0]])
            ).all()

    class TestSetRegionToZeros:
        def test__region_is_corner__sets_to_0(self):
            frame = ac.frame.manual(array=np.ones((2, 2)))

            region = ac.Region(region=(0, 1, 0, 1))

            frame[region.slice] = 0

            assert (frame == np.array([[0.0, 1.0], [1.0, 1.0]])).all()

        def test__different_region___sets_to_0(self):
            frame = ac.frame.manual(array=np.ones((3, 3)))

            region = ac.Region(region=(1, 3, 2, 3))

            frame[region.slice] = 0

            assert (
                frame == np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
            ).all()


class TestParallelFrontEdgeOfRegion:

    def test__top_left__extracts_rows_from_top_of_region(self):
        frame = ac.frame.manual(array=np.ones((3, 3)), corner=(0, 0))

        region = ac.Region(
            (0, 3, 0, 3)
        )  # The front edge is closest to 3, so for 1 edge we extract row 3-> 4

        front_edge = frame.parallel_front_edge_of_region(
            region=region, rows=(0, 1)
        )

        assert front_edge == (2, 3, 0, 3)

         # The front edge is closest to 3, so for these 2 rows we extract 2 & 3

        front_edge = frame.parallel_front_edge_of_region(
            region=region, rows=(0, 2)
        )

        assert front_edge == (1, 3, 0, 3)

        # The front edge is closest to 3, so for these 2 rows we extract 1 & 2

        front_edge = frame.parallel_front_edge_of_region(
            region=region, rows=(1, 3)
        )

        assert front_edge == (0, 2, 0, 3)

    def test__top_right__same_extraction_as_above(
            self
    ):
        frame = ac.frame.manual(array=np.ones((3, 3)), corner=(0, 1))

        region = ac.Region(
            (0, 3, 0, 3)
        )  # The front edge is closest to 3, so for these 2 rows we extract 1 & 2

        front_edge = frame.parallel_front_edge_of_region(
            region=region, rows=(1, 3)
        )

        assert front_edge == (0, 2, 0, 3)

    def test__bottom_left__extracts_rows_from_bottom_of_region(self):

        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 0))

        region = ac.Region(
            region=(0, 3, 0, 3)
        )

        # Front edge is row 0, so for 1 row we extract 0 -> 1

        front_edge = frame.parallel_front_edge_of_region(
            region=region, rows=(0, 1)
        )

        assert front_edge == (0, 1, 0, 3)

         # Front edge is row 0, so for 2 rows we extract 0 -> 2

        front_edge = frame.parallel_front_edge_of_region(
            region=region, rows=(0, 2)
        )

        assert front_edge == (0, 2, 0, 3)

        # Front edge is row 0, so for these 2 rows we extract 1 ->2

        front_edge = frame.parallel_front_edge_of_region(
            region=region, rows=(1, 3)
        )

        assert front_edge == (1, 3, 0, 3)

    def test__bottom_right__same_extraction_as_above(
        self
    ):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 1))

        region = ac.Region(
            region=(0, 3, 0, 3)
        )  # Front edge is row 0, so for these 2 rows we extract 1 ->2

        front_edge = frame.parallel_front_edge_of_region(
            region=region, rows=(1, 3)
        )

        assert front_edge == (1, 3, 0, 3)


class TestParallelTrailsOfRegion:

    def test__top_left__extracts_rows_behind_region(self):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(0, 0))

        region = ac.Region(
            (3, 5, 0, 3)
        )  # The trails are the rows after row 3, so for 1 edge we should extract just row 2

        trails = frame.parallel_trails_of_region(
            region=region, rows=(0, 1)
        )

        assert trails == (2, 3, 0, 3)

        # The trails are the row after row 3, so for these 2 edges we extract rows 1->3

        trails = frame.parallel_trails_of_region(
            region=region, rows=(0, 2)
        )

        assert trails == (1, 3, 0, 3)

          # The trails are the row after row 3, so for these 2 edges we extract rows 0 & 2

        trails = frame.parallel_trails_of_region(
            region=region, rows=(1, 3)
        )

        assert trails == (0, 2, 0, 3)

    def test__top_right__extacts_same_rows_as_above(
        self
    ):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(0, 1))

        region = ac.Region(
            (3, 5, 0, 3)
        )  # The trails are the row after row 3, so for these 2 edges we extract rows 0 & 2

        trails = frame.parallel_trails_of_region(
            region=region, rows=(1, 3)
        )

        assert trails == (0, 2, 0, 3)

    def test__bottom_left__extracts_rows_above_region(self):

        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 0))

        region = ac.Region(
            region=(0, 3, 0, 3)
        )  # The trails are row 3 and above, so extract 3 -> 4

        trails = frame.parallel_trails_of_region(
            region=region, rows=(0, 1)
        )

        assert trails == (3, 4, 0, 3)

         # The trails are row 3 and above, so extract 3 -> 5

        trails = frame.parallel_trails_of_region(
            region=region, rows=(0, 2)
        )

        assert trails == (3, 5, 0, 3)

         # The trails are row 3 and above, so extract 4 -> 6

        trails = frame.parallel_trails_of_region(
            region=region, rows=(1, 3)
        )

        assert trails == (4, 6, 0, 3)

    def test__bottom_right__extracts_same_rows_as_above(
        self
    ):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 1))

        region = ac.Region(
            (0, 3, 0, 3)
        )  # The trails are row 3 and above, so extract 4 -> 6

        trails = frame.parallel_trails_of_region(
            region=region, rows=(1, 3)
        )

        assert trails == (4, 6, 0, 3)


class TestParallelRegionNearestReadOut:

    def test__top_left__columns_0_to_1__asymetric_image(self):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 0))
        region = ac.Region(region=(1, 3, 0, 5))

        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, image_shape=(2, 5), columns=(0, 1)
        )

        assert parallel_region == (0, 2, 0, 1)

    def test__top_right__columns_0_to_1__asymetric_image(self):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(0, 1))
        region = ac.Region(region=(0, 1, 0, 5))
        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, image_shape=(2, 5), columns=(0, 1)
        )

        assert parallel_region == (0, 2, 4, 5)

    def test__bottom_left__columns_0_to_1__region_is_left_hand_side(self):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 0))
        region = ac.Region(region=(1, 3, 0, 5))

        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, image_shape=(5, 5), columns=(0, 1)
        )

        assert parallel_region == (0, 5, 0, 1)

    def test__bottom_left__columns_1_to_3__region_is_left_hand_side(self):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 0))
        region = ac.Region(region=(1, 3, 0, 5))

        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, image_shape=(4, 4), columns=(1, 3)
        )

        assert parallel_region == (0, 4, 1, 3)

    def test__bottom_left__columns_1_to_3__different_region(self):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 0))
        region = ac.Region(region=(1, 3, 2, 5))

        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, image_shape=(4, 4), columns=(1, 3)
        )

        assert parallel_region == (0, 4, 3, 5)

    def test__bottom_left__columns_0_to_1__asymetric_image(self):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 0))
        region = ac.Region(region=(1, 3, 0, 5))

        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, image_shape=(2, 5), columns=(0, 1)
        )

        assert parallel_region == (0, 2, 0, 1)

    def test__bottom_right__columns_0_to_1__region_is_right_hand_side(self):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 1))
        region = ac.Region(region=(1, 3, 0, 5))

        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, image_shape=(5, 5), columns=(0, 1)
        )

        assert parallel_region == (0, 5, 4, 5)

    def test__bottom_right__columns_1_to_3__region_is_right_hand_side(self):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 1))
        region = ac.Region(region=(1, 3, 0, 4))

        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, image_shape=(4, 4), columns=(1, 3)
        )

        assert parallel_region == (0, 4, 1, 3)

    def test__bottom_right__columns_1_to_3__different_region(self):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 1))
        region = ac.Region(region=(1, 3, 2, 4))

        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, image_shape=(4, 4), columns=(1, 3)
        )

        assert parallel_region == (0, 4, 1, 3)

    def test__bottom_right__columns_0_to_1__asymetric_image(self):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 1))
        region = ac.Region(region=(0, 1, 0, 5))
        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, image_shape=(2, 5), columns=(0, 1)
        )

        assert parallel_region == (0, 2, 4, 5)


class TestSerialFrontEdgeRegion:

    def test__top_left__extract_two_columns__second_and_third__takes_coordinates_from_top_of_region(
        self
    ):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(0, 0))

        region = ac.Region(
            region=(0, 3, 0, 3)
        )  # Front edge is column 0, so for these 2 columns we extract 1 ->2

        front_edge = frame.serial_front_edge_region(
            region=region, columns=(1, 3)
        )

        assert front_edge == (0, 3, 1, 3)

    def test__top_right__extract_two_columns__second_and_third__takes_coordinates_from_right_of_region(
        self
    ):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 1))

        region = ac.Region(
            region=(0, 3, 0, 3)
        )  # Front edge is column 0, so for these 2 columns we extract 1 ->2

        front_edge = frame.serial_front_edge_region(
            region=region, columns=(1, 3)
        )

        assert front_edge == (0, 3, 0, 2)

    def test__bottom_left__extract_one_column__takes_coordinates_from_left_of_region(self):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 0))

        region = ac.Region(
            region=(0, 3, 0, 3)
        )  # Front edge is column 0, so for 1 column we extract 0 -> 1

        front_edge = frame.serial_front_edge_region(
            region=region, columns=(0, 1)
        )

        assert front_edge == (0, 3, 0, 1)

    def test__bottom_left__extract_two_columns__first_and_second__takes_coordinates_from_left_of_region(
        self
    ):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 0))

        region = ac.Region(
            region=(0, 3, 0, 3)
        )  # Front edge is column 0, so for 2 columns we extract 0 -> 2

        front_edge = frame.serial_front_edge_region(
            region=region, columns=(0, 2)
        )

        assert front_edge == (0, 3, 0, 2)

    def test__bottom_left__extract_two_columns__second_and_third__takes_coordinates_from_left_of_region(
        self
    ):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 0))

        region = ac.Region(
            region=(0, 3, 0, 3)
        )  # Front edge is column 0, so for these 2 columns we extract 1 ->2

        front_edge = frame.serial_front_edge_region(
            region=region, columns=(1, 3)
        )

        assert front_edge == (0, 3, 1, 3)

    def test__bottom_right__extract_one_column__takes_coordinates_from_right_of_region(self):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 1))

        region = ac.Region(
            region=(0, 3, 0, 3)
        )  # Front edge is column 0, so for 1 column we extract 0 -> 1

        front_edge = frame.serial_front_edge_region(
            region=region, columns=(0, 1)
        )

        assert front_edge == (0, 3, 2, 3)

    def test__bottom_right__extract_two_columns__first_and_second__takes_coordinates_from_right_of_region(
        self
    ):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 1))

        region = ac.Region(
            region=(0, 3, 0, 3)
        )  # Front edge is column 0, so for 2 columns we extract 0 -> 2

        front_edge = frame.serial_front_edge_region(
            region=region, columns=(0, 2)
        )

        assert front_edge == (0, 3, 1, 3)

    def test__bottom_right__extract_two_columns__second_and_third__takes_coordinates_from_right_of_region(
        self
    ):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 1))

        region = ac.Region(
            region=(0, 3, 0, 3)
        )  # Front edge is column 0, so for these 2 columns we extract 1 ->2

        front_edge = frame.serial_front_edge_region(
            region=region, columns=(1, 3)
        )

        assert front_edge == (0, 3, 0, 2)


class TestSerialTrailsRegion:

    def test__top_left__extract_two_columns__second_and_third__takes_coordinates_after_bottom_of_region(
        self
    ):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(0, 0))

        region = ac.Region(
            (0, 3, 0, 3)
        )  # The trails are column 3 and above, so extract 4 -> 6

        trails = frame.serial_trails_region(
            region=region, columns=(1, 3)
        )

        assert trails == (0, 3, 4, 6)

    def test__top_right__extract_two_columns__second_and_third__takes_coordinates_after_left_of_region(
        self
    ):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 1))

        region = ac.Region(
            (0, 3, 3, 6)
        )  # The trails are column 3 and above, so extract 4 -> 6

        trails = frame.serial_trails_region(
            region=region, columns=(1, 3)
        )

        assert trails == (0, 3, 0, 2)

    def test__bottom_left__extract_one_row__takes_coordinates_after_right_of_region(self):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 0))

        region = ac.Region(
            region=(0, 3, 0, 3)
        )  # The trails are column 3 and above, so extract 3 -> 4

        trails = frame.serial_trails_region(
            region=region, columns=(0, 1)
        )

        assert trails == (0, 3, 3, 4)

    def test__bottom_left__extract_two_columns__first_and_second__takes_coordinates_after_right_of_region(
        self
    ):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 0))

        region = ac.Region(
            (0, 3, 0, 3)
        )  # The trails are column 3 and above, so extract 3 -> 5

        trails = frame.serial_trails_region(
            region=region, columns=(0, 2)
        )

        assert trails == (0, 3, 3, 5)

    def test__bottom_left__extract_two_columns__second_and_third__takes_coordinates_after_right_of_region(
        self
    ):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 0))

        region = ac.Region(
            (0, 3, 0, 3)
        )  # The trails are column 3 and above, so extract 4 -> 6

        trails = frame.serial_trails_region(
            region=region, columns=(1, 3)
        )

        assert trails == (0, 3, 4, 6)

    def test__bottom_right__extract_one_row__takes_coordinates_after_left_of_region(self):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 1))

        region = ac.Region(
            region=(0, 3, 3, 6)
        )  # The trails are column 3 and above, so extract 3 -> 4

        trails = frame.serial_trails_region(
            region=region, columns=(0, 1)
        )

        assert trails == (0, 3, 2, 3)

    def test__bottom_right__extract_two_columns__first_and_second__takes_coordinates_after_left_of_region(
        self
    ):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 1))

        region = ac.Region(
            (0, 3, 3, 6)
        )  # The trails are column 3 and above, so extract 3 -> 5

        trails = frame.serial_trails_region(
            region=region, columns=(0, 2)
        )

        assert trails == (0, 3, 1, 3)

    def test__bottom_right__extract_two_columns__second_and_third__takes_coordinates_after_left_of_region(
        self
    ):
        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 1))

        region = ac.Region(
            (0, 3, 3, 6)
        )  # The trails are column 3 and above, so extract 4 -> 6

        trails = frame.serial_trails_region(
            region=region, columns=(1, 3)
        )

        assert trails == (0, 3, 0, 2)


class TestSerialChargeInjectionAndTrails:

    def test__top_left__different_region(self):

        frame = ac.frame.manual(array=np.ones((3,3)), corner=(0, 0))
        region = ac.Region(region=(3, 5, 5, 30))

        serial_region = frame.serial_prescan_region_and_trails(
            region=region, image_shape=(8, 55)
        )

        assert serial_region == (3, 5, 0, 55)

    def test__top_right__different_region(self):

        frame = ac.frame.manual(array=np.ones((3,3)), corner=(0, 1))
        region = ac.Region(region=(3, 5, 5, 30))

        serial_region = frame.serial_prescan_region_and_trails(
            region=region, image_shape=(8, 55)
        )

        assert serial_region == (3, 5, 0, 55)

    def test__bottom_left__column_0_of_front_edge__region_is_left_hand_side__no_overscan_beyond_region(
        self
    ):

        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 0))
        region = ac.Region(region=(1, 3, 0, 5))

        serial_region = frame.serial_prescan_region_and_trails(
            region=region, image_shape=(5, 5)
        )

        assert serial_region == (1, 3, 0, 5)

    def test__bottom_left__column_0_of_front_edge__region_is_left_hand_side__overscan_beyond_region(
        self
    ):

        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 0))
        region = ac.Region(region=(1, 3, 0, 5))

        serial_region = frame.serial_prescan_region_and_trails(
            region=region, image_shape=(5, 25)
        )

        assert serial_region == (1, 3, 0, 25)

    def test__bottom_right__front_edge__region_is_left_hand_side__no_overscan_beyond_region(
        self
    ):

        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 1))
        region = ac.Region(region=(1, 3, 0, 5))

        serial_region = frame.serial_prescan_region_and_trails(
            region=region, image_shape=(5, 5)
        )

        assert serial_region == (1, 3, 0, 5)

    def test__bottom_right__front_edge__region_is_left_hand_side__overscan_beyond_region(self):

        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 1))
        region = ac.Region(region=(1, 3, 20, 25))

        serial_region = frame.serial_prescan_region_and_trails(
            region=region, image_shape=(5, 25)
        )

        assert serial_region == (1, 3, 0, 25)

    def test__bottom_right__region_has_overscan_and_prescan_either_side__prescan_included(
        self
    ):

        frame = ac.frame.manual(array=np.ones((3,3)), corner=(1, 1))
        region = ac.Region(region=(1, 3, 10, 20))

        serial_region = frame.serial_prescan_region_and_trails(
            region=region, image_shape=(5, 25)
        )

        assert serial_region == (1, 3, 0, 25)


class TestFrameArrayRotations:
    def test__top_left__rotate_for_parallel_clocking_and_back_again__returns_to_original_orientation(
        self, frame_data
    ):
        # Quadrant 2 - top left panel of Euclid CCD - input pre_ctis should be rotateped upside-down for parallel cti

        frame = ac.frame.manual(array=frame_data, corner=(0, 0))

        frame_rotated = frame.rotated_for_parallel_cti
        assert (frame_rotated == np.flipud(frame_data)).all()

        frame_rotated_back = frame_rotated.rotated_for_parallel_cti
        assert (frame_rotated_back == frame_data).all()

    def test__top_left__rotate_for_serial_clocking_and_back_again__returns_to_original_orientation(
        self, frame_data
    ):
        # Quadrant 2 - top left panel of Euclid CCD - input pre_ctis should be rotated 90 degrees for serial cti.

        frame = ac.frame.manual(array=frame_data, corner=(0, 0))

        frame_rotated = frame.rotated_before_serial_clocking
        assert (frame_rotated == frame_data.T).all()

        frame_rotated_back = frame_rotated.rotated_after_serial_clocking
        assert (frame_rotated_back == frame_data).all()

    def test__top_right__rotate_for_parallel_clocking_and_back_again__returns_to_original_orientation(
        self, frame_data
    ):
        # Quadrant 3 - top right panel of Euclid CCD - input pre_ctis should be rotateped upside-down for parallel cti.

        frame = ac.frame.manual(array=frame_data, corner=(0, 1))

        frame_rotated = frame.rotated_for_parallel_cti
        assert (frame_rotated == np.flipud(frame_data)).all()

        frame_rotated_back = frame_rotated.rotated_for_parallel_cti
        assert (frame_rotated_back == frame_data).all()

    def test__top_right__rotate_for_serial_clocking_and_back_again__returns_to_original_orientation(
        self, frame_data
    ):
        # Quadrant 3 - top right panel of Euclid CCD - input pre_ctis should be rotated 270 degrees for serial cti.

        frame = ac.frame.manual(array=frame_data, corner=(0, 1))

        frame_rotated = frame.rotated_before_serial_clocking
        assert (frame_rotated == np.fliplr(frame_data).T).all()

        frame_rotated_back = frame_rotated.rotated_after_serial_clocking
        assert (frame_rotated_back == frame_data).all()

    def test__bottom_left__rotate_for_parallel_clocking_and_back_again__returns_to_original_orientation(
        self, frame_data
    ):
        # Quadrant 0 - Bottom left panel of Euclid CCD - input pre_ctis should not be rotated for parallel cti.

        frame = ac.frame.manual(array=frame_data, corner=(1, 0))
        frame_rotated = frame.rotated_for_parallel_cti

        assert (frame_rotated == frame_data).all()

        frame_rotated_back = frame_rotated.rotated_for_parallel_cti

        assert (frame_rotated_back == frame_data).all()

    def test__bottom__left__rotate_for_serial_clocking_and_back_again__returns_to_original_orientation(
        self, frame_data
    ):
        # Quadrant 0 - Bottom left panel of Euclid CCD - input pre_ctis should be rotated 90 degrees for
        # serial cti.

        frame = ac.frame.manual(array=frame_data, corner=(1, 0))
        frame_rotated = frame.rotated_before_serial_clocking

        assert (frame_rotated == frame_data.T).all()

        frame_rotated_back = frame_rotated.rotated_after_serial_clocking

        assert (frame_rotated_back == frame_data).all()

    def test__bottom_right__rotate_for_parallel_clocking_and_back_again__returns_to_original_orientation(
        self, frame_data
    ):
        # Quadrant 1 - Bottom right panel of Euclid CCD - input pre_ctis should not be rotateped for parallel cti.

        frame = ac.frame.manual(array=frame_data, corner=(1, 1))

        frame_rotated = frame.rotated_for_parallel_cti
        assert (frame_rotated == frame_data).all()

        frame_rotated_back = frame_rotated.rotated_for_parallel_cti

        assert (frame_rotated_back == frame_data).all()

    def test__bottom_right__rotate_for_serial_clocking_and_back_again__returns_to_original_orientation(
        self, frame_data
    ):
        frame = ac.frame.manual(array=frame_data, corner=(1, 1))

        frame_rotated = frame.rotated_before_serial_clocking
        assert (frame_rotated == np.fliplr(frame_data).T).all()

        frame_rotated_back = frame_rotated.rotated_after_serial_clocking
        assert (frame_rotated_back == frame_data).all()
