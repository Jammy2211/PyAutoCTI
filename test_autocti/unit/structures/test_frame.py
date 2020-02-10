import os

import numpy as np
import pytest

from autocti import exc
import autocti as ac
from autocti.structures import frame


@pytest.fixture(scope="class")
def euclid_data():
    euclid_data = np.zeros((2086, 2119))
    return euclid_data


path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))


class TestFrameAPI:
    class TestConstructors:
        def test__manual__makes_frame_using_inputs__include_rotations(self):

            frame = ac.frame.manual(
                array=[[1.0, 2.0], [3.0, 4.0]],
                roe_corner=(1, 0),
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            )

            assert (frame == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (frame.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (frame.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
            assert frame.original_roe_corner == (1, 0)
            assert frame.parallel_overscan == (0, 1, 0, 1)
            assert frame.serial_prescan == (1, 2, 1, 2)
            assert frame.serial_overscan == (0, 2, 0, 2)
            assert (frame.mask == np.array([[False, False], [False, False]])).all()

            frame = ac.frame.manual(
                array=[[1.0, 2.0], [3.0, 4.0]],
                roe_corner=(0, 0),
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            )

            assert (frame == np.array([[3.0, 4.0], [1.0, 2.0]])).all()
            assert (frame.in_2d == np.array([[3.0, 4.0], [1.0, 2.0]])).all()
            assert (frame.in_1d == np.array([3.0, 4.0, 1.0, 2.0])).all()
            assert frame.original_roe_corner == (0, 0)
            assert frame.parallel_overscan == (1, 2, 0, 1)
            assert frame.serial_prescan == (0, 1, 1, 2)
            assert frame.serial_overscan == (0, 2, 0, 2)
            assert (frame.mask == np.array([[False, False], [False, False]])).all()

            frame = ac.frame.manual(
                array=[[1.0, 2.0], [3.0, 4.0]],
                roe_corner=(1, 1),
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            )

            assert (frame == np.array([[2.0, 1.0], [4.0, 3.0]])).all()
            assert (frame.in_2d == np.array([[2.0, 1.0], [4.0, 3.0]])).all()
            assert (frame.in_1d == np.array([2.0, 1.0, 4.0, 3.0])).all()
            assert frame.original_roe_corner == (1, 1)
            assert frame.parallel_overscan == (0, 1, 1, 2)
            assert frame.serial_prescan == (1, 2, 0, 1)
            assert frame.serial_overscan == (0, 2, 0, 2)
            assert (frame.mask == np.array([[False, False], [False, False]])).all()

            frame = ac.frame.manual(
                array=[[1.0, 2.0], [3.0, 4.0]],
                roe_corner=(0, 1),
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            )

            assert (frame == np.array([[4.0, 3.0], [2.0, 1.0]])).all()
            assert (frame.in_2d == np.array([[4.0, 3.0], [2.0, 1.0]])).all()
            assert (frame.in_1d == np.array([4.0, 3.0, 2.0, 1.0])).all()
            assert frame.original_roe_corner == (0, 1)
            assert frame.parallel_overscan == (1, 2, 1, 2)
            assert frame.serial_prescan == (0, 1, 0, 1)
            assert frame.serial_overscan == (0, 2, 0, 2)
            assert (frame.mask == np.array([[False, False], [False, False]])).all()

        def test__full_ones_zeros__makes_frame_using_inputs(self):

            frame = ac.frame.full(
                fill_value=8.0,
                shape_2d=(2, 2),
                roe_corner=(1, 0),
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            )

            assert (frame == np.array([[8.0, 8.0], [8.0, 8.0]])).all()
            assert (frame.in_2d == np.array([[8.0, 8.0], [8.0, 8.0]])).all()
            assert (frame.in_1d == np.array([8.0, 8.0, 8.0, 8.0])).all()
            assert frame.original_roe_corner == (1, 0)
            assert frame.parallel_overscan == (0, 1, 0, 1)
            assert frame.serial_prescan == (1, 2, 1, 2)
            assert frame.serial_overscan == (0, 2, 0, 2)
            assert (frame.mask == np.array([[False, False], [False, False]])).all()

            frame = ac.frame.ones(
                shape_2d=(2, 2),
                roe_corner=(1, 0),
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            )

            assert (frame == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert (frame.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert (frame.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert frame.original_roe_corner == (1, 0)
            assert frame.parallel_overscan == (0, 1, 0, 1)
            assert frame.serial_prescan == (1, 2, 1, 2)
            assert frame.serial_overscan == (0, 2, 0, 2)
            assert (frame.mask == np.array([[False, False], [False, False]])).all()

            frame = ac.frame.zeros(
                shape_2d=(2, 2),
                roe_corner=(1, 0),
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            )

            assert (frame == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert (frame.in_2d == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert (frame.in_1d == np.array([0.0, 0.0, 0.0, 0.0])).all()
            assert frame.original_roe_corner == (1, 0)
            assert frame.parallel_overscan == (0, 1, 0, 1)
            assert frame.serial_prescan == (1, 2, 1, 2)
            assert frame.serial_overscan == (0, 2, 0, 2)
            assert (frame.mask == np.array([[False, False], [False, False]])).all()

    class TestEuclid:
        def test__euclid_frame_for_four_quandrants__loads_data_and_dimensions(
            self, euclid_data
        ):

            euclid_frame = ac.euclid_frame.top_left(array=euclid_data)

            assert euclid_frame.original_roe_corner == (0, 0)
            assert euclid_frame.shape_2d == (2086, 2119)
            assert (euclid_frame.in_2d == np.zeros((2086, 2119))).all()
            assert euclid_frame.parallel_overscan == (2066, 2086, 51, 2099)
            assert euclid_frame.serial_prescan == (0, 2086, 0, 51)
            assert euclid_frame.serial_overscan == (0, 2086, 2099, 2119)

            euclid_frame = ac.euclid_frame.top_right(array=euclid_data)

            assert euclid_frame.original_roe_corner == (0, 1)
            assert euclid_frame.shape_2d == (2086, 2119)
            assert (euclid_frame.in_2d == np.zeros((2086, 2119))).all()
            assert euclid_frame.parallel_overscan == (2066, 2086, 51, 2099)
            assert euclid_frame.serial_prescan == (0, 2086, 0, 51)
            assert euclid_frame.serial_overscan == (0, 2086, 2099, 2119)

            euclid_frame = ac.euclid_frame.bottom_left(array=euclid_data)

            assert euclid_frame.original_roe_corner == (1, 0)
            assert euclid_frame.shape_2d == (2086, 2119)
            assert (euclid_frame.in_2d == np.zeros((2086, 2119))).all()
            assert euclid_frame.parallel_overscan == (2066, 2086, 51, 2099)
            assert euclid_frame.serial_prescan == (0, 2086, 0, 51)
            assert euclid_frame.serial_overscan == (0, 2086, 2099, 2119)

            euclid_frame = ac.euclid_frame.bottom_right(array=euclid_data)

            assert euclid_frame.original_roe_corner == (1, 1)
            assert euclid_frame.shape_2d == (2086, 2119)
            assert (euclid_frame.in_2d == np.zeros((2086, 2119))).all()
            assert euclid_frame.parallel_overscan == (2066, 2086, 51, 2099)
            assert euclid_frame.serial_prescan == (0, 2086, 0, 51)
            assert euclid_frame.serial_overscan == (0, 2086, 2099, 2119)

        def test__left_side__chooses_correct_frame_given_input(self, euclid_data):
            frame = ac.euclid_frame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text1", quad_id="E"
            )

            assert frame.original_roe_corner == (1, 0)

            frame = ac.euclid_frame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text2", quad_id="E"
            )

            assert frame.original_roe_corner == (1, 0)

            frame = ac.euclid_frame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text3", quad_id="E"
            )

            assert frame.original_roe_corner == (1, 0)

            frame = ac.euclid_frame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text1", quad_id="F"
            )

            assert frame.original_roe_corner == (1, 1)

            frame = ac.euclid_frame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text2", quad_id="F"
            )

            assert frame.original_roe_corner == (1, 1)

            frame = ac.euclid_frame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text3", quad_id="F"
            )

            assert frame.original_roe_corner == (1, 1)

            frame = ac.euclid_frame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text1", quad_id="G"
            )

            assert frame.original_roe_corner == (0, 1)

            frame = ac.euclid_frame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text2", quad_id="G"
            )

            assert frame.original_roe_corner == (0, 1)

            frame = ac.euclid_frame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text3", quad_id="G"
            )

            assert frame.original_roe_corner == (0, 1)

            frame = ac.euclid_frame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text1", quad_id="H"
            )

            assert frame.original_roe_corner == (0, 0)

            frame = ac.euclid_frame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text2", quad_id="H"
            )

            assert frame.original_roe_corner == (0, 0)

            frame = ac.euclid_frame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text3", quad_id="H"
            )

            assert frame.original_roe_corner == (0, 0)

        def test__right_side__chooses_correct_frame_given_input(self, euclid_data):
            frame = ac.euclid_frame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text4", quad_id="E"
            )

            assert frame.original_roe_corner == (0, 1)

            frame = ac.euclid_frame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text5", quad_id="E"
            )

            assert frame.original_roe_corner == (0, 1)

            frame = ac.euclid_frame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text6", quad_id="E"
            )

            assert frame.original_roe_corner == (0, 1)

            frame = ac.euclid_frame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text4", quad_id="F"
            )

            assert frame.original_roe_corner == (0, 0)

            frame = ac.euclid_frame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text5", quad_id="F"
            )

            assert frame.original_roe_corner == (0, 0)

            frame = ac.euclid_frame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text6", quad_id="F"
            )

            assert frame.original_roe_corner == (0, 0)

            frame = ac.euclid_frame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text4", quad_id="G"
            )

            assert frame.original_roe_corner == (1, 0)

            frame = ac.euclid_frame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text5", quad_id="G"
            )

            assert frame.original_roe_corner == (1, 0)

            frame = ac.euclid_frame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text6", quad_id="G"
            )

            assert frame.original_roe_corner == (1, 0)

            frame = ac.euclid_frame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text4", quad_id="H"
            )

            assert frame.original_roe_corner == (1, 1)

            frame = ac.euclid_frame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text5", quad_id="H"
            )

            assert frame.original_roe_corner == (1, 1)

            frame = ac.euclid_frame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text6", quad_id="H"
            )

            assert frame.original_roe_corner == (1, 1)


class TestMaskedFrameAPI:
    def test__manual__makes_frame_using_inputs__includes_rotation_which_includes_mask(
        self
    ):

        mask = ac.mask.manual(mask_2d=[[False, True], [False, False]])

        frame = ac.masked.frame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            mask=mask,
            roe_corner=(1, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (frame == np.array([[1.0, 0.0], [3.0, 4.0]])).all()
        assert (frame.in_2d == np.array([[1.0, 0.0], [3.0, 4.0]])).all()
        assert (frame.in_1d == np.array([1.0, 3.0, 4.0])).all()
        assert frame.original_roe_corner == (1, 0)
        assert frame.parallel_overscan == (0, 1, 0, 1)
        assert frame.serial_prescan == (1, 2, 1, 2)
        assert frame.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, True], [False, False]])).all()

        frame = ac.masked.frame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            mask=mask,
            roe_corner=(0, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (frame == np.array([[3.0, 4.0], [1.0, 0.0]])).all()
        assert (frame.in_2d == np.array([[3.0, 4.0], [1.0, 0.0]])).all()
        assert (frame.in_1d == np.array([3.0, 4.0, 1.0])).all()
        assert frame.original_roe_corner == (0, 0)
        assert frame.parallel_overscan == (1, 2, 0, 1)
        assert frame.serial_prescan == (0, 1, 1, 2)
        assert frame.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, False], [False, True]])).all()

        frame = ac.masked.frame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            mask=mask,
            roe_corner=(1, 1),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (frame == np.array([[0.0, 1.0], [4.0, 3.0]])).all()
        assert (frame.in_2d == np.array([[0.0, 1.0], [4.0, 3.0]])).all()
        assert (frame.in_1d == np.array([1.0, 4.0, 3.0])).all()
        assert frame.original_roe_corner == (1, 1)
        assert frame.parallel_overscan == (0, 1, 1, 2)
        assert frame.serial_prescan == (1, 2, 0, 1)
        assert frame.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[True, False], [False, False]])).all()

        frame = ac.masked.frame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            mask=mask,
            roe_corner=(0, 1),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (frame == np.array([[4.0, 3.0], [0.0, 1.0]])).all()
        assert (frame.in_2d == np.array([[4.0, 3.0], [0.0, 1.0]])).all()
        assert (frame.in_1d == np.array([4.0, 3.0, 1.0])).all()
        assert frame.original_roe_corner == (0, 1)
        assert frame.parallel_overscan == (1, 2, 1, 2)
        assert frame.serial_prescan == (0, 1, 0, 1)
        assert frame.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, False], [True, False]])).all()

    def test__from_frame__no_rotation_as_frame_is_correct_orientation(self):

        mask = ac.mask.manual(mask_2d=[[False, True], [False, False]])

        frame = ac.frame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            roe_corner=(1, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        frame = ac.masked.frame.from_frame(frame=frame, mask=mask)

        assert (frame == np.array([[1.0, 0.0], [3.0, 4.0]])).all()
        assert (frame.in_2d == np.array([[1.0, 0.0], [3.0, 4.0]])).all()
        assert (frame.in_1d == np.array([1.0, 3.0, 4.0])).all()
        assert frame.original_roe_corner == (1, 0)
        assert frame.parallel_overscan == (0, 1, 0, 1)
        assert frame.serial_prescan == (1, 2, 1, 2)
        assert frame.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, True], [False, False]])).all()

        mask = ac.mask.manual(mask_2d=[[False, True], [False, False]])

        frame = ac.frame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            roe_corner=(0, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        frame = ac.masked.frame.from_frame(frame=frame, mask=mask)

        assert (frame == np.array([[3.0, 0.0], [1.0, 2.0]])).all()
        assert (frame.in_2d == np.array([[3.0, 0.0], [1.0, 2.0]])).all()
        assert (frame.in_1d == np.array([3.0, 1.0, 2.0])).all()
        assert frame.original_roe_corner == (0, 0)
        assert frame.parallel_overscan == (1, 2, 0, 1)
        assert frame.serial_prescan == (0, 1, 1, 2)
        assert frame.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, True], [False, False]])).all()

    def test__ones_zeros_full__makes_frame_using_inputs(self):

        mask = ac.mask.manual(mask_2d=[[False, True], [False, False]])

        frame = ac.masked.frame.full(
            fill_value=8.0,
            mask=mask,
            roe_corner=(1, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (frame == np.array([[8.0, 0.0], [8.0, 8.0]])).all()
        assert (frame.in_2d == np.array([[8.0, 0.0], [8.0, 8.0]])).all()
        assert (frame.in_1d == np.array([8.0, 8.0, 8.0])).all()
        assert frame.original_roe_corner == (1, 0)
        assert frame.parallel_overscan == (0, 1, 0, 1)
        assert frame.serial_prescan == (1, 2, 1, 2)
        assert frame.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, True], [False, False]])).all()

        frame = ac.masked.frame.ones(
            mask=mask,
            roe_corner=(1, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (frame == np.array([[1.0, 0.0], [1.0, 1.0]])).all()
        assert (frame.in_2d == np.array([[1.0, 0.0], [1.0, 1.0]])).all()
        assert (frame.in_1d == np.array([1.0, 1.0, 1.0])).all()
        assert frame.original_roe_corner == (1, 0)
        assert frame.parallel_overscan == (0, 1, 0, 1)
        assert frame.serial_prescan == (1, 2, 1, 2)
        assert frame.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, True], [False, False]])).all()

        mask = ac.mask.manual(mask_2d=[[False, True], [False, False]])

        frame = ac.masked.frame.zeros(
            mask=mask,
            roe_corner=(1, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (frame == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
        assert (frame.in_2d == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
        assert (frame.in_1d == np.array([0.0, 0.0, 0.0])).all()
        assert frame.original_roe_corner == (1, 0)
        assert frame.parallel_overscan == (0, 1, 0, 1)
        assert frame.serial_prescan == (1, 2, 1, 2)
        assert frame.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, True], [False, False]])).all()

    def test__euclid_frame_for_four_quandrants__loads_data_and_dimensions(
        self, euclid_data
    ):

        mask = np.full(shape=(2086, 2119), fill_value=False)
        mask[0, 0] = True

        euclid_frame = ac.masked.euclid_frame.top_left(array=euclid_data, mask=mask)

        assert euclid_frame.original_roe_corner == (0, 0)
        assert euclid_frame.shape_2d == (2086, 2119)
        assert (euclid_frame.in_2d == np.zeros((2086, 2119))).all()
        assert euclid_frame.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_frame.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.serial_overscan == (0, 2086, 2099, 2119)
        assert euclid_frame.mask[2085, 0] == True
        assert euclid_frame.mask[2085, 1] == False

        euclid_frame = ac.masked.euclid_frame.top_right(array=euclid_data, mask=mask)

        assert euclid_frame.original_roe_corner == (0, 1)
        assert euclid_frame.shape_2d == (2086, 2119)
        assert (euclid_frame.in_2d == np.zeros((2086, 2119))).all()
        assert euclid_frame.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_frame.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.serial_overscan == (0, 2086, 2099, 2119)
        assert euclid_frame.mask[2085, 2118] == True
        assert euclid_frame.mask[2085, 2117] == False

        euclid_frame = ac.masked.euclid_frame.bottom_left(array=euclid_data, mask=mask)

        assert euclid_frame.original_roe_corner == (1, 0)
        assert euclid_frame.shape_2d == (2086, 2119)
        assert (euclid_frame.in_2d == np.zeros((2086, 2119))).all()
        assert euclid_frame.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_frame.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.serial_overscan == (0, 2086, 2099, 2119)
        assert euclid_frame.mask[0, 0] == True
        assert euclid_frame.mask[0, 1] == False

        euclid_frame = ac.masked.euclid_frame.bottom_right(array=euclid_data, mask=mask)

        assert euclid_frame.original_roe_corner == (1, 1)
        assert euclid_frame.shape_2d == (2086, 2119)
        assert (euclid_frame.in_2d == np.zeros((2086, 2119))).all()
        assert euclid_frame.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_frame.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.serial_overscan == (0, 2086, 2099, 2119)
        assert euclid_frame.mask[0, 2118] == True
        assert euclid_frame.mask[0, 2117] == False

    def test__left_side__chooses_correct_frame_given_input(self, euclid_data):

        mask = np.full(shape=(2086, 2119), fill_value=False)
        mask[0, 0] = True

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text1", quad_id="E"
        )

        assert frame.original_roe_corner == (1, 0)
        assert frame.mask[0, 0] == True
        assert frame.mask[0, 1] == False

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text2", quad_id="E"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text3", quad_id="E"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text1", quad_id="F"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text2", quad_id="F"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text3", quad_id="F"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text1", quad_id="G"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text2", quad_id="G"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text3", quad_id="G"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text1", quad_id="H"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text2", quad_id="H"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text3", quad_id="H"
        )

        assert frame.original_roe_corner == (0, 0)

    def test__right_side__chooses_correct_frame_given_input(self, euclid_data):

        mask = np.full(shape=(2086, 2119), fill_value=False)
        mask[0, 0] = True

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text4", quad_id="E"
        )

        assert frame.original_roe_corner == (0, 1)
        assert frame.mask[2085, 2118] == True
        assert frame.mask[2085, 2117] == False

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text5", quad_id="E"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text6", quad_id="E"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text4", quad_id="F"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text5", quad_id="F"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text6", quad_id="F"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text4", quad_id="G"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text5", quad_id="G"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text6", quad_id="G"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text4", quad_id="H"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text5", quad_id="H"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = ac.masked.euclid_frame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text6", quad_id="H"
        )

        assert frame.original_roe_corner == (1, 1)


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
        frame = ac.frame.manual(array=np.ones((3, 3)), roe_corner=(0, 0))

        region = ac.Region(
            (0, 3, 0, 3)
        )  # The front edge is closest to 3, so for 1 edge we extract row 3-> 4

        front_edge = frame.parallel_front_edge_of_region(region=region, rows=(0, 1))

        assert front_edge == (2, 3, 0, 3)

        # The front edge is closest to 3, so for these 2 rows we extract 2 & 3

        front_edge = frame.parallel_front_edge_of_region(region=region, rows=(0, 2))

        assert front_edge == (1, 3, 0, 3)

        # The front edge is closest to 3, so for these 2 rows we extract 1 & 2

        front_edge = frame.parallel_front_edge_of_region(region=region, rows=(1, 3))

        assert front_edge == (0, 2, 0, 3)

    def test__top_right__same_extraction_as_above(self):
        frame = ac.frame.manual(array=np.ones((3, 3)), roe_corner=(0, 1))

        region = ac.Region(
            (0, 3, 0, 3)
        )  # The front edge is closest to 3, so for these 2 rows we extract 1 & 2

        front_edge = frame.parallel_front_edge_of_region(region=region, rows=(1, 3))

        assert front_edge == (0, 2, 0, 3)

    def test__bottom_left__extracts_rows_from_bottom_of_region(self):

        frame = ac.frame.manual(array=np.ones((3, 3)), roe_corner=(1, 0))

        region = ac.Region(region=(0, 3, 0, 3))

        # Front edge is row 0, so for 1 row we extract 0 -> 1

        front_edge = frame.parallel_front_edge_of_region(region=region, rows=(0, 1))

        assert front_edge == (0, 1, 0, 3)

        # Front edge is row 0, so for 2 rows we extract 0 -> 2

        front_edge = frame.parallel_front_edge_of_region(region=region, rows=(0, 2))

        assert front_edge == (0, 2, 0, 3)

        # Front edge is row 0, so for these 2 rows we extract 1 ->2

        front_edge = frame.parallel_front_edge_of_region(region=region, rows=(1, 3))

        assert front_edge == (1, 3, 0, 3)

    def test__bottom_right__same_extraction_as_above(self):
        frame = ac.frame.manual(array=np.ones((3, 3)), roe_corner=(1, 1))

        region = ac.Region(
            region=(0, 3, 0, 3)
        )  # Front edge is row 0, so for these 2 rows we extract 1 ->2

        front_edge = frame.parallel_front_edge_of_region(region=region, rows=(1, 3))

        assert front_edge == (1, 3, 0, 3)


class TestParallelTrailsOfRegion:
    def test__top_left__extracts_rows_behind_region(self):
        frame = ac.frame.manual(array=np.ones((3, 3)), roe_corner=(0, 0))

        region = ac.Region(
            (3, 5, 0, 3)
        )  # The trails are the rows after row 3, so for 1 edge we should extract just row 2

        trails = frame.parallel_trails_of_region(region=region, rows=(0, 1))

        assert trails == (2, 3, 0, 3)

        # The trails are the row after row 3, so for these 2 edges we extract rows 1->3

        trails = frame.parallel_trails_of_region(region=region, rows=(0, 2))

        assert trails == (1, 3, 0, 3)

        # The trails are the row after row 3, so for these 2 edges we extract rows 0 & 2

        trails = frame.parallel_trails_of_region(region=region, rows=(1, 3))

        assert trails == (0, 2, 0, 3)

    def test__top_right__extacts_same_rows_as_above(self):
        frame = ac.frame.manual(array=np.ones((3, 3)), roe_corner=(0, 1))

        region = ac.Region(
            (3, 5, 0, 3)
        )  # The trails are the row after row 3, so for these 2 edges we extract rows 0 & 2

        trails = frame.parallel_trails_of_region(region=region, rows=(1, 3))

        assert trails == (0, 2, 0, 3)

    def test__bottom_left__extracts_rows_above_region(self):

        frame = ac.frame.manual(array=np.ones((3, 3)), roe_corner=(1, 0))

        region = ac.Region(
            region=(0, 3, 0, 3)
        )  # The trails are row 3 and above, so extract 3 -> 4

        trails = frame.parallel_trails_of_region(region=region, rows=(0, 1))

        assert trails == (3, 4, 0, 3)

        # The trails are row 3 and above, so extract 3 -> 5

        trails = frame.parallel_trails_of_region(region=region, rows=(0, 2))

        assert trails == (3, 5, 0, 3)

        # The trails are row 3 and above, so extract 4 -> 6

        trails = frame.parallel_trails_of_region(region=region, rows=(1, 3))

        assert trails == (4, 6, 0, 3)

    def test__bottom_right__extracts_same_rows_as_above(self):
        frame = ac.frame.manual(array=np.ones((3, 3)), roe_corner=(1, 1))

        region = ac.Region(
            (0, 3, 0, 3)
        )  # The trails are row 3 and above, so extract 4 -> 6

        trails = frame.parallel_trails_of_region(region=region, rows=(1, 3))

        assert trails == (4, 6, 0, 3)


class TestParallelRegionNearestReadOut:
    def test__top_left__extracts_region_to_left_of_region(self):
        frame = ac.frame.manual(array=np.ones((2, 5)), roe_corner=(1, 0))
        region = ac.Region(region=(1, 3, 0, 5))

        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, columns=(0, 1)
        )

        assert parallel_region == (0, 2, 0, 1)

    def test__top_right__extracts_region_to_right_of_region(self):

        frame = ac.frame.manual(array=np.ones((2, 5)), roe_corner=(0, 1))
        region = ac.Region(region=(0, 1, 0, 5))
        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, columns=(0, 1)
        )

        assert parallel_region == (0, 2, 4, 5)

    def test__bottom_left__extracts_regioon_to_left_of_region(self):
        frame = ac.frame.manual(array=np.ones((5, 5)), roe_corner=(1, 0))
        region = ac.Region(region=(1, 3, 0, 5))

        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, columns=(0, 1)
        )

        assert parallel_region == (0, 5, 0, 1)

        frame = ac.frame.manual(array=np.ones((4, 4)), roe_corner=(1, 0))
        region = ac.Region(region=(1, 3, 0, 5))

        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, columns=(1, 3)
        )

        assert parallel_region == (0, 4, 1, 3)

        region = ac.Region(region=(1, 3, 2, 5))

        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, columns=(1, 3)
        )

        assert parallel_region == (0, 4, 3, 5)

        frame = ac.frame.manual(array=np.ones((2, 5)), roe_corner=(1, 0))
        region = ac.Region(region=(1, 3, 0, 5))

        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, columns=(0, 1)
        )

        assert parallel_region == (0, 2, 0, 1)

    def test__bottom_right__extracts_region_to_right_of_region(self):
        frame = ac.frame.manual(array=np.ones((5, 5)), roe_corner=(1, 1))
        region = ac.Region(region=(1, 3, 0, 5))

        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, columns=(0, 1)
        )

        assert parallel_region == (0, 5, 4, 5)

        frame = ac.frame.manual(array=np.ones((4, 4)), roe_corner=(1, 1))
        region = ac.Region(region=(1, 3, 0, 4))

        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, columns=(1, 3)
        )

        assert parallel_region == (0, 4, 1, 3)

        region = ac.Region(region=(1, 3, 2, 4))

        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, columns=(1, 3)
        )

        assert parallel_region == (0, 4, 1, 3)

        frame = ac.frame.manual(array=np.ones((2, 5)), roe_corner=(1, 1))
        region = ac.Region(region=(0, 1, 0, 5))
        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, columns=(0, 1)
        )

        assert parallel_region == (0, 2, 4, 5)


class TestSerialFrontEdgeOfRegion:
    def test__top_left__extracts_coluns_to_left_of_region(self):
        frame = ac.frame.manual(array=np.ones((3, 3)), roe_corner=(0, 0))

        region = ac.Region(
            region=(0, 3, 0, 3)
        )  # Front edge is column 0, so for these 2 columns we extract 1 ->2

        front_edge = frame.serial_front_edge_of_region(region=region, columns=(1, 3))

        assert front_edge == (0, 3, 1, 3)

    def test__top_right__extracts_regions_to_right_of_region(self):
        frame = ac.frame.manual(array=np.ones((3, 3)), roe_corner=(1, 1))

        region = ac.Region(
            region=(0, 3, 0, 3)
        )  # Front edge is column 0, so for these 2 columns we extract 1 ->2

        front_edge = frame.serial_front_edge_of_region(region=region, columns=(1, 3))

        assert front_edge == (0, 3, 0, 2)

    def test__bottom_left__extracts_region_to_left_of_region(self):
        frame = ac.frame.manual(array=np.ones((3, 3)), roe_corner=(1, 0))

        region = ac.Region(
            region=(0, 3, 0, 3)
        )  # Front edge is column 0, so for 1 column we extract 0 -> 1

        front_edge = frame.serial_front_edge_of_region(region=region, columns=(0, 1))

        assert front_edge == (0, 3, 0, 1)

        # Front edge is column 0, so for 2 columns we extract 0 -> 2

        front_edge = frame.serial_front_edge_of_region(region=region, columns=(0, 2))

        assert front_edge == (0, 3, 0, 2)

        # Front edge is column 0, so for these 2 columns we extract 1 ->2

        front_edge = frame.serial_front_edge_of_region(region=region, columns=(1, 3))

        assert front_edge == (0, 3, 1, 3)

    def test__bottom_right__extracts_region_to_right_of_region(self):
        frame = ac.frame.manual(array=np.ones((3, 3)), roe_corner=(1, 1))

        region = ac.Region(
            region=(0, 3, 0, 3)
        )  # Front edge is column 0, so for 1 column we extract 0 -> 1

        front_edge = frame.serial_front_edge_of_region(region=region, columns=(0, 1))

        assert front_edge == (0, 3, 2, 3)

        # Front edge is column 0, so for 2 columns we extract 0 -> 2

        front_edge = frame.serial_front_edge_of_region(region=region, columns=(0, 2))

        assert front_edge == (0, 3, 1, 3)

        # Front edge is column 0, so for these 2 columns we extract 1 ->2

        front_edge = frame.serial_front_edge_of_region(region=region, columns=(1, 3))

        assert front_edge == (0, 3, 0, 2)


class TestSerialTrailsOfRegion:
    def test__top_left__extracts_region_to_left_of_region(self):
        frame = ac.frame.manual(array=np.ones((3, 3)), roe_corner=(0, 0))

        region = ac.Region(
            (0, 3, 0, 3)
        )  # The trails are column 3 and above, so extract 4 -> 6

        trails = frame.serial_trails_of_region(region=region, columns=(1, 3))

        assert trails == (0, 3, 4, 6)

    def test__top_right__extracts_region_to_right_of_region(self):
        frame = ac.frame.manual(array=np.ones((3, 3)), roe_corner=(1, 1))

        region = ac.Region(
            (0, 3, 3, 6)
        )  # The trails are column 3 and above, so extract 4 -> 6

        trails = frame.serial_trails_of_region(region=region, columns=(1, 3))

        assert trails == (0, 3, 0, 2)

    def test__bottom_left__extracts_region_to_left_of_region(self):
        frame = ac.frame.manual(array=np.ones((3, 3)), roe_corner=(1, 0))

        region = ac.Region(
            region=(0, 3, 0, 3)
        )  # The trails are column 3 and above, so extract 3 -> 4

        trails = frame.serial_trails_of_region(region=region, columns=(0, 1))

        assert trails == (0, 3, 3, 4)

        # The trails are column 3 and above, so extract 3 -> 5

        trails = frame.serial_trails_of_region(region=region, columns=(0, 2))

        assert trails == (0, 3, 3, 5)

        # The trails are column 3 and above, so extract 4 -> 6

        trails = frame.serial_trails_of_region(region=region, columns=(1, 3))

        assert trails == (0, 3, 4, 6)

    def test__bottom_right__extracts_region_to_right_of_region(self):
        frame = ac.frame.manual(array=np.ones((3, 3)), roe_corner=(1, 1))

        region = ac.Region(
            region=(0, 3, 3, 6)
        )  # The trails are column 3 and above, so extract 3 -> 4

        trails = frame.serial_trails_of_region(region=region, columns=(0, 1))

        assert trails == (0, 3, 2, 3)

        # The trails are column 3 and above, so extract 3 -> 5

        trails = frame.serial_trails_of_region(region=region, columns=(0, 2))

        assert trails == (0, 3, 1, 3)

        # The trails are column 3 and above, so extract 4 -> 6

        trails = frame.serial_trails_of_region(region=region, columns=(1, 3))

        assert trails == (0, 3, 0, 2)


class TestSerialEntireRowsOfRegion:
    def test__top_left__full_region_from_left_most_prescan_to_right_most_end_of_trails(
        self
    ):

        frame = ac.frame.manual(array=np.ones((8, 55)), roe_corner=(0, 0))
        region = ac.Region(region=(3, 5, 5, 30))

        serial_region = frame.serial_entire_rows_of_region(region=region)

        assert serial_region == (3, 5, 0, 55)

    def test__top_right__full_region_from_left_most_prescan_to_right_most_end_of_trails(
        self
    ):

        frame = ac.frame.manual(array=np.ones((8, 55)), roe_corner=(0, 1))
        region = ac.Region(region=(3, 5, 5, 30))

        serial_region = frame.serial_entire_rows_of_region(region=region)

        assert serial_region == (3, 5, 0, 55)

    def test__bottom_left__full_region_from_left_most_prescan_to_right_most_end_of_trails(
        self
    ):

        frame = ac.frame.manual(array=np.ones((5, 5)), roe_corner=(1, 0))
        region = ac.Region(region=(1, 3, 0, 5))

        serial_region = frame.serial_entire_rows_of_region(region=region)

        assert serial_region == (1, 3, 0, 5)

        frame = ac.frame.manual(array=np.ones((5, 25)), roe_corner=(1, 0))
        region = ac.Region(region=(1, 3, 0, 5))

        serial_region = frame.serial_entire_rows_of_region(region=region)

        assert serial_region == (1, 3, 0, 25)

    def test__bottom_right__full_region_from_left_most_prescan_to_right_most_end_of_trails(
        self
    ):

        frame = ac.frame.manual(array=np.ones((5, 5)), roe_corner=(1, 1))
        region = ac.Region(region=(1, 3, 0, 5))

        serial_region = frame.serial_entire_rows_of_region(region=region)

        assert serial_region == (1, 3, 0, 5)

        frame = ac.frame.manual(array=np.ones((5, 25)), roe_corner=(1, 1))
        region = ac.Region(region=(1, 3, 20, 25))

        serial_region = frame.serial_entire_rows_of_region(region=region)

        assert serial_region == (1, 3, 0, 25)

        region = ac.Region(region=(1, 3, 10, 20))

        serial_region = frame.serial_entire_rows_of_region(region=region)

        assert serial_region == (1, 3, 0, 25)
