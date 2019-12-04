import os

import numpy as np
import pytest

from autocti import exc
import autocti as ac


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
            array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

            region = ac.Region(region=(0, 2, 0, 2))

            new_array = array[region.slice]

            assert (new_array == np.array([[1.0, 2.0], [4.0, 5.0]])).all()

        def test__extracts_2x3_region_of_3x3_array(self):
            array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

            region = ac.Region(region=(1, 3, 0, 3))

            new_array = array[region.slice]

            assert (new_array == np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])).all()

    class TestAddRegionToArrayFromImage:
        def test__array_is_all_zeros__image_goes_into_correct_region_of_array(self):
            array = np.zeros((2, 2))
            image = np.ones((2, 2))
            region = ac.Region(region=(0, 1, 0, 1))

            array[region.slice] += image[region.slice]

            assert (array == np.array([[1.0, 0.0], [0.0, 0.0]])).all()

        def test__array_is_all_1s__image_goes_into_correct_region_of_array_and_adds_to_it(
            self
        ):
            array = np.ones((2, 2))
            image = np.ones((2, 2))
            region = ac.Region(region=(0, 1, 0, 1))

            array[region.slice] += image[region.slice]

            assert (array == np.array([[2.0, 1.0], [1.0, 1.0]])).all()

        def test__different_region(self):
            array = np.ones((3, 3))
            image = np.ones((3, 3))
            region = ac.Region(region=(1, 3, 2, 3))

            array[region.slice] += image[region.slice]

            assert (
                array == np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 2.0], [1.0, 1.0, 2.0]])
            ).all()

    class TestSetRegionToZeros:
        def test__region_is_corner__sets_to_0(self):
            array = np.ones((2, 2))

            region = ac.Region(region=(0, 1, 0, 1))

            array[region.slice] = 0

            assert (array == np.array([[0.0, 1.0], [1.0, 1.0]])).all()

        def test__different_region___sets_to_0(self):
            array = np.ones((3, 3))

            region = ac.Region(region=(1, 3, 2, 3))

            array[region.slice] = 0

            assert (
                array == np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
            ).all()


class TestFrameArrayRotations:

        def test__bottom_left__rotate_for_parallel_clocking_and_back_again__returns_to_original_orientation(
            self, frame_data
        ):
            # Quadrant 0 - Bottom left panel of Euclid CCD - input ci_pre_ctis should not be rotated for parallel cti.

            frame = ac.FrameArray(array=frame_data, corner=(0, 0))
            frame_rotated = frame.rotated_for_parallel_cti

            assert (frame_rotated == frame_data).all()

            frame_rotated_back = frame_rotated.rotated_for_parallel_cti

            assert (frame_rotated_back == frame_data).all()

        def test__bottom__left__rotate_for_serial_clocking_and_back_again__returns_to_original_orientation(
            self, frame_data
        ):
            # Quadrant 0 - Bottom left panel of Euclid CCD - input ci_pre_ctis should be rotated 90 degrees for
            # serial cti.

            frame = ac.FrameArray(array=frame_data, corner=(0, 0))
            frame_rotated = frame.rotated_before_serial_clocking

            assert (frame_rotated == frame_data.T).all()

            frame_rotated_back = frame_rotated.rotated_after_serial_clocking

            assert (frame_rotated_back == frame_data).all()

        def test__bottom_right__rotate_for_parallel_clocking_and_back_again__returns_to_original_orientation(
            self, frame_data
        ):
            # Quadrant 1 - Bottom right panel of Euclid CCD - input ci_pre_ctis should not be rotateped for parallel cti.

            frame = ac.FrameArray(array=frame_data, corner=(0, 1))

            frame_rotated = frame.rotated_for_parallel_cti
            assert (frame_rotated == frame_data).all()

            frame_rotated_back = frame_rotated.rotated_for_parallel_cti

            assert (frame_rotated_back == frame_data).all()

        def test__bottom_right__rotate_for_serial_clocking_and_back_again__returns_to_original_orientation(
            self, frame_data
        ):
            frame = ac.FrameArray(array=frame_data, corner=(0, 1))

            frame_rotated = frame.rotated_before_serial_clocking
            assert (frame_rotated == np.fliplr(frame_data).T).all()

            frame_rotated_back = frame_rotated.rotated_after_serial_clocking
            assert (frame_rotated_back == frame_data).all()

        def test__top_left__rotate_for_parallel_clocking_and_back_again__returns_to_original_orientation(
            self, frame_data
        ):
            # Quadrant 2 - top left panel of Euclid CCD - input ci_pre_ctis should be rotateped upside-down for parallel cti

            frame = ac.FrameArray(array=frame_data, corner=(1, 0))

            frame_rotated = frame.rotated_for_parallel_cti
            assert (frame_rotated == np.flipud(frame_data)).all()

            frame_rotated_back = frame_rotated.rotated_for_parallel_cti
            assert (frame_rotated_back == frame_data).all()

        def test__top_left__rotate_for_serial_clocking_and_back_again__returns_to_original_orientation(
            self, frame_data
        ):
            # Quadrant 2 - top left panel of Euclid CCD - input ci_pre_ctis should be rotated 90 degrees for serial cti.

            frame = ac.FrameArray(array=frame_data, corner=(1, 0))

            frame_rotated = frame.rotated_before_serial_clocking
            assert (frame_rotated == frame_data.T).all()

            frame_rotated_back = frame_rotated.rotated_after_serial_clocking
            assert (frame_rotated_back == frame_data).all()

        def test__top_right__rotate_for_parallel_clocking_and_back_again__returns_to_original_orientation(
            self, frame_data
        ):
            # Quadrant 3 - top right panel of Euclid CCD - input ci_pre_ctis should be rotateped upside-down for parallel cti.

            frame = ac.FrameArray(array=frame_data, corner=(1, 1))

            frame_rotated = frame.rotated_for_parallel_cti
            assert (frame_rotated == np.flipud(frame_data)).all()

            frame_rotated_back = frame_rotated.rotated_for_parallel_cti
            assert (frame_rotated_back == frame_data).all()

        def test__top_right__rotate_for_serial_clocking_and_back_again__returns_to_original_orientation(
            self, frame_data
        ):
            # Quadrant 3 - top right panel of Euclid CCD - input ci_pre_ctis should be rotated 270 degrees for serial cti.

            frame = ac.FrameArray(array=frame_data, corner=(1, 1))

            frame_rotated = frame.rotated_before_serial_clocking
            assert (frame_rotated == np.fliplr(frame_data).T).all()

            frame_rotated_back = frame_rotated.rotated_after_serial_clocking
            assert (frame_rotated_back == frame_data).all()


class TestEuclidFrame:
    def test__euclid_arrays_for_four_quandrants__loads_data_and_dimensions(
        self, euclid_data
    ):

        euclid_frame = ac.EuclidArray.euclid_bottom_left(array=euclid_data)

        assert type(euclid_frame) == ac.EuclidArray
        assert euclid_frame.shape_2d == (2048, 2066)
        assert (euclid_frame.in_2d == np.zeros((2048, 2066))).all()
        assert euclid_frame.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_frame.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.serial_overscan == (0, 2086, 2099, 2119)

        euclid_frame = ac.EuclidArray.euclid_bottom_right(array=euclid_data)

        assert type(euclid_frame) == ac.EuclidArray
        assert euclid_frame.shape_2d == (2048, 2066)
        assert (euclid_frame.in_2d == np.zeros((2048, 2066))).all()
        assert euclid_frame.parallel_overscan == (2066, 2086, 20, 2068)
        assert euclid_frame.serial_prescan == (0, 2086, 2068, 2119)
        assert euclid_frame.serial_overscan == (0, 2086, 0, 20)

        euclid_frame = ac.EuclidArray.euclid_top_left(array=euclid_data)

        assert type(euclid_frame) == ac.EuclidArray
        assert euclid_frame.shape_2d == (2048, 2066)
        assert (euclid_frame.in_2d == np.zeros((2048, 2066))).all()
        assert euclid_frame.parallel_overscan == (0, 20, 51, 2099)
        assert euclid_frame.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.serial_overscan == (0, 2086, 2099, 2119)

        euclid_frame = ac.EuclidArray.euclid_top_right(array=euclid_data)

        assert type(euclid_frame) == ac.EuclidArray
        assert euclid_frame.shape_2d == (2048, 2066)
        assert (euclid_frame.in_2d == np.zeros((2048, 2066))).all()
        assert euclid_frame.parallel_overscan == (0, 20, 20, 2068)
        assert euclid_frame.serial_prescan == (0, 2086, 2068, 2119)
        assert euclid_frame.serial_overscan == (0, 2086, 0, 20)

    def test__left_side__chooses_correct_frame_given_input(
        self, frame_data
    ):
        frame = ac.EuclidArray.euclid_from_ccd_and_quadrant_id(
            array=frame_data, ccd_id="text1", quad_id="E"
        )

        assert frame.corner == (0, 0)

        frame = ac.EuclidArray.euclid_from_ccd_and_quadrant_id(
            array=frame_data, ccd_id="text2", quad_id="E"
        )

        assert frame.corner == (0, 0)

        frame = ac.EuclidArray.euclid_from_ccd_and_quadrant_id(
            array=frame_data, ccd_id="text3", quad_id="E"
        )

        assert frame.corner == (0, 0)

        frame = ac.EuclidArray.euclid_from_ccd_and_quadrant_id(
            array=frame_data, ccd_id="text1", quad_id="F"
        )

        assert frame.corner == (0, 1)

        frame = ac.EuclidArray.euclid_from_ccd_and_quadrant_id(
            array=frame_data, ccd_id="text2", quad_id="F"
        )

        assert frame.corner == (0, 1)

        frame = ac.EuclidArray.euclid_from_ccd_and_quadrant_id(
            array=frame_data, ccd_id="text3", quad_id="F"
        )

        assert frame.corner == (0, 1)

        frame = ac.EuclidArray.euclid_from_ccd_and_quadrant_id(
            array=frame_data, ccd_id="text1", quad_id="G"
        )

        assert frame.corner == (1, 1)

        frame = ac.EuclidArray.euclid_from_ccd_and_quadrant_id(
            array=frame_data, ccd_id="text2", quad_id="G"
        )

        assert frame.corner == (1, 1)

        frame = ac.EuclidArray.euclid_from_ccd_and_quadrant_id(
            array=frame_data, ccd_id="text3", quad_id="G"
        )

        assert frame.corner == (1, 1)

        frame = ac.EuclidArray.euclid_from_ccd_and_quadrant_id(
            array=frame_data, ccd_id="text1", quad_id="H"
        )

        assert frame.corner == (1, 0)

        frame = ac.EuclidArray.euclid_from_ccd_and_quadrant_id(
            array=frame_data, ccd_id="text2", quad_id="H"
        )

        assert frame.corner == (1, 0)

        frame = ac.EuclidArray.euclid_from_ccd_and_quadrant_id(
            array=frame_data, ccd_id="text3", quad_id="H"
        )

        assert frame.corner == (1, 0)

    def test__right_side__chooses_correct_frame_given_input(
        self, frame_data
    ):
        frame = ac.EuclidArray.euclid_from_ccd_and_quadrant_id(
            array=frame_data, ccd_id="text4", quad_id="E"
        )

        assert frame.corner == (1, 1)

        frame = ac.EuclidArray.euclid_from_ccd_and_quadrant_id(
            array=frame_data, ccd_id="text5", quad_id="E"
        )

        assert frame.corner == (1, 1)

        frame = ac.EuclidArray.euclid_from_ccd_and_quadrant_id(
            array=frame_data, ccd_id="text6", quad_id="E"
        )

        assert frame.corner == (1, 1)

        frame = ac.EuclidArray.euclid_from_ccd_and_quadrant_id(
            array=frame_data, ccd_id="text4", quad_id="F"
        )

        assert frame.corner == (1, 0)

        frame = ac.EuclidArray.euclid_from_ccd_and_quadrant_id(
            array=frame_data, ccd_id="text5", quad_id="F"
        )

        assert frame.corner == (1, 0)

        frame = ac.EuclidArray.euclid_from_ccd_and_quadrant_id(
            array=frame_data, ccd_id="text6", quad_id="F"
        )

        assert frame.corner == (1, 0)

        frame = ac.EuclidArray.euclid_from_ccd_and_quadrant_id(
            array=frame_data, ccd_id="text4", quad_id="G"
        )

        assert frame.corner == (0, 0)

        frame = ac.EuclidArray.euclid_from_ccd_and_quadrant_id(
            array=frame_data, ccd_id="text5", quad_id="G"
        )

        assert frame.corner == (0, 0)

        frame = ac.EuclidArray.euclid_from_ccd_and_quadrant_id(
            array=frame_data, ccd_id="text6", quad_id="G"
        )

        assert frame.corner == (0, 0)

        frame = ac.EuclidArray.euclid_from_ccd_and_quadrant_id(
            array=frame_data, ccd_id="text4", quad_id="H"
        )

        assert frame.corner == (0, 1)

        frame = ac.EuclidArray.euclid_from_ccd_and_quadrant_id(
            array=frame_data, ccd_id="text5", quad_id="H"
        )

        assert frame.corner == (0, 1)

        frame = ac.EuclidArray.euclid_from_ccd_and_quadrant_id(
            array=frame_data, ccd_id="text6", quad_id="H"
        )

        assert frame.corner == (0, 1)