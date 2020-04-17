import os

import numpy as np
import pytest

from autocti import structures as struct


@pytest.fixture(scope="class")
def euclid_data():
    euclid_data = np.zeros((2086, 2119))
    return euclid_data


path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))


class TestFrameAPI:
    class TestConstructors:
        def test__manual__makes_frame_using_inputs__include_rotations(self):

            frame = struct.Frame.manual(
                array=[[1.0, 2.0], [3.0, 4.0]],
                roe_corner=(1, 0),
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            )

            assert (frame == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert frame.original_roe_corner == (1, 0)
            assert frame.parallel_overscan == (0, 1, 0, 1)
            assert frame.serial_prescan == (1, 2, 1, 2)
            assert frame.serial_overscan == (0, 2, 0, 2)
            assert (frame.mask == np.array([[False, False], [False, False]])).all()

            frame = struct.Frame.manual(
                array=[[1.0, 2.0], [3.0, 4.0]],
                roe_corner=(0, 0),
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            )

            assert (frame == np.array([[3.0, 4.0], [1.0, 2.0]])).all()
            assert frame.original_roe_corner == (0, 0)
            assert frame.parallel_overscan == (1, 2, 0, 1)
            assert frame.serial_prescan == (0, 1, 1, 2)
            assert frame.serial_overscan == (0, 2, 0, 2)
            assert (frame.mask == np.array([[False, False], [False, False]])).all()

            frame = struct.Frame.manual(
                array=[[1.0, 2.0], [3.0, 4.0]],
                roe_corner=(1, 1),
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            )

            assert (frame == np.array([[2.0, 1.0], [4.0, 3.0]])).all()
            assert frame.original_roe_corner == (1, 1)
            assert frame.parallel_overscan == (0, 1, 1, 2)
            assert frame.serial_prescan == (1, 2, 0, 1)
            assert frame.serial_overscan == (0, 2, 0, 2)
            assert (frame.mask == np.array([[False, False], [False, False]])).all()

            frame = struct.Frame.manual(
                array=[[1.0, 2.0], [3.0, 4.0]],
                roe_corner=(0, 1),
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            )

            assert (frame == np.array([[4.0, 3.0], [2.0, 1.0]])).all()
            assert frame.original_roe_corner == (0, 1)
            assert frame.parallel_overscan == (1, 2, 1, 2)
            assert frame.serial_prescan == (0, 1, 0, 1)
            assert frame.serial_overscan == (0, 2, 0, 2)
            assert (frame.mask == np.array([[False, False], [False, False]])).all()

        def test__full_ones_zeros__makes_frame_using_inputs(self):

            frame = struct.Frame.full(
                fill_value=8.0,
                shape_2d=(2, 2),
                roe_corner=(1, 0),
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            )

            assert (frame == np.array([[8.0, 8.0], [8.0, 8.0]])).all()
            assert frame.original_roe_corner == (1, 0)
            assert frame.parallel_overscan == (0, 1, 0, 1)
            assert frame.serial_prescan == (1, 2, 1, 2)
            assert frame.serial_overscan == (0, 2, 0, 2)
            assert (frame.mask == np.array([[False, False], [False, False]])).all()

            frame = struct.Frame.ones(
                shape_2d=(2, 2),
                roe_corner=(1, 0),
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            )

            assert (frame == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert frame.original_roe_corner == (1, 0)
            assert frame.parallel_overscan == (0, 1, 0, 1)
            assert frame.serial_prescan == (1, 2, 1, 2)
            assert frame.serial_overscan == (0, 2, 0, 2)
            assert (frame.mask == np.array([[False, False], [False, False]])).all()

            frame = struct.Frame.zeros(
                shape_2d=(2, 2),
                roe_corner=(1, 0),
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            )

            assert (frame == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert frame.original_roe_corner == (1, 0)
            assert frame.parallel_overscan == (0, 1, 0, 1)
            assert frame.serial_prescan == (1, 2, 1, 2)
            assert frame.serial_overscan == (0, 2, 0, 2)
            assert (frame.mask == np.array([[False, False], [False, False]])).all()

        def test__extracted_frame_from_frame_and_extraction_region(self):

            frame = struct.Frame.manual(
                array=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                roe_corner=(1, 0),
                parallel_overscan=None,
                serial_prescan=(0, 2, 0, 2),
                serial_overscan=(1, 2, 1, 2),
            )

            frame = struct.Frame.extracted_frame_from_frame_and_extraction_region(
                frame=frame, extraction_region=struct.Region(region=(1, 3, 1, 3))
            )

            assert (frame == np.array([[5.0, 6.0], [8.0, 9.0]])).all()
            assert frame.original_roe_corner == (1, 0)
            assert frame.parallel_overscan == None
            assert frame.serial_prescan == (0, 1, 0, 1)
            assert frame.serial_overscan == (0, 1, 0, 1)
            assert (frame.mask == np.array([[False, False], [False, False]])).all()

    class TestEuclid:
        def test__euclid_frame_for_four_quandrants__loads_data_and_dimensions(
            self, euclid_data
        ):

            euclid_frame = struct.EuclidFrame.top_left(array=euclid_data)

            assert euclid_frame.original_roe_corner == (0, 0)
            assert euclid_frame.shape_2d == (2086, 2119)
            assert (euclid_frame == np.zeros((2086, 2119))).all()
            assert euclid_frame.parallel_overscan == (2066, 2086, 51, 2099)
            assert euclid_frame.serial_prescan == (0, 2086, 0, 51)
            assert euclid_frame.serial_overscan == (0, 2086, 2099, 2119)

            euclid_frame = struct.EuclidFrame.top_right(array=euclid_data)

            assert euclid_frame.original_roe_corner == (0, 1)
            assert euclid_frame.shape_2d == (2086, 2119)
            assert (euclid_frame == np.zeros((2086, 2119))).all()
            assert euclid_frame.parallel_overscan == (2066, 2086, 51, 2099)
            assert euclid_frame.serial_prescan == (0, 2086, 0, 51)
            assert euclid_frame.serial_overscan == (0, 2086, 2099, 2119)

            euclid_frame = struct.EuclidFrame.bottom_left(array=euclid_data)

            assert euclid_frame.original_roe_corner == (1, 0)
            assert euclid_frame.shape_2d == (2086, 2119)
            assert (euclid_frame == np.zeros((2086, 2119))).all()
            assert euclid_frame.parallel_overscan == (2066, 2086, 51, 2099)
            assert euclid_frame.serial_prescan == (0, 2086, 0, 51)
            assert euclid_frame.serial_overscan == (0, 2086, 2099, 2119)

            euclid_frame = struct.EuclidFrame.bottom_right(array=euclid_data)

            assert euclid_frame.original_roe_corner == (1, 1)
            assert euclid_frame.shape_2d == (2086, 2119)
            assert (euclid_frame == np.zeros((2086, 2119))).all()
            assert euclid_frame.parallel_overscan == (2066, 2086, 51, 2099)
            assert euclid_frame.serial_prescan == (0, 2086, 0, 51)
            assert euclid_frame.serial_overscan == (0, 2086, 2099, 2119)

        def test__left_side__chooses_correct_frame_given_input(self, euclid_data):
            frame = struct.EuclidFrame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text1", quadrant_id="E"
            )

            assert frame.original_roe_corner == (1, 0)

            frame = struct.EuclidFrame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text2", quadrant_id="E"
            )

            assert frame.original_roe_corner == (1, 0)

            frame = struct.EuclidFrame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text3", quadrant_id="E"
            )

            assert frame.original_roe_corner == (1, 0)

            frame = struct.EuclidFrame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text1", quadrant_id="F"
            )

            assert frame.original_roe_corner == (1, 1)

            frame = struct.EuclidFrame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text2", quadrant_id="F"
            )

            assert frame.original_roe_corner == (1, 1)

            frame = struct.EuclidFrame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text3", quadrant_id="F"
            )

            assert frame.original_roe_corner == (1, 1)

            frame = struct.EuclidFrame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text1", quadrant_id="G"
            )

            assert frame.original_roe_corner == (0, 1)

            frame = struct.EuclidFrame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text2", quadrant_id="G"
            )

            assert frame.original_roe_corner == (0, 1)

            frame = struct.EuclidFrame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text3", quadrant_id="G"
            )

            assert frame.original_roe_corner == (0, 1)

            frame = struct.EuclidFrame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text1", quadrant_id="H"
            )

            assert frame.original_roe_corner == (0, 0)

            frame = struct.EuclidFrame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text2", quadrant_id="H"
            )

            assert frame.original_roe_corner == (0, 0)

            frame = struct.EuclidFrame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text3", quadrant_id="H"
            )

            assert frame.original_roe_corner == (0, 0)

        def test__right_side__chooses_correct_frame_given_input(self, euclid_data):
            frame = struct.EuclidFrame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text4", quadrant_id="E"
            )

            assert frame.original_roe_corner == (0, 1)

            frame = struct.EuclidFrame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text5", quadrant_id="E"
            )

            assert frame.original_roe_corner == (0, 1)

            frame = struct.EuclidFrame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text6", quadrant_id="E"
            )

            assert frame.original_roe_corner == (0, 1)

            frame = struct.EuclidFrame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text4", quadrant_id="F"
            )

            assert frame.original_roe_corner == (0, 0)

            frame = struct.EuclidFrame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text5", quadrant_id="F"
            )

            assert frame.original_roe_corner == (0, 0)

            frame = struct.EuclidFrame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text6", quadrant_id="F"
            )

            assert frame.original_roe_corner == (0, 0)

            frame = struct.EuclidFrame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text4", quadrant_id="G"
            )

            assert frame.original_roe_corner == (1, 0)

            frame = struct.EuclidFrame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text5", quadrant_id="G"
            )

            assert frame.original_roe_corner == (1, 0)

            frame = struct.EuclidFrame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text6", quadrant_id="G"
            )

            assert frame.original_roe_corner == (1, 0)

            frame = struct.EuclidFrame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text4", quadrant_id="H"
            )

            assert frame.original_roe_corner == (1, 1)

            frame = struct.EuclidFrame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text5", quadrant_id="H"
            )

            assert frame.original_roe_corner == (1, 1)

            frame = struct.EuclidFrame.ccd_and_quadrant_id(
                array=euclid_data, ccd_id="text6", quadrant_id="H"
            )

            assert frame.original_roe_corner == (1, 1)


class TestMaskedFrameAPI:
    def test__manual__makes_frame_using_inputs__includes_rotation_which_includes_mask(
        self
    ):

        mask = struct.Mask.manual(mask_2d=[[False, True], [False, False]])

        frame = struct.MaskedFrame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            mask=mask,
            roe_corner=(1, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (frame == np.array([[1.0, 0.0], [3.0, 4.0]])).all()
        assert frame.original_roe_corner == (1, 0)
        assert frame.parallel_overscan == (0, 1, 0, 1)
        assert frame.serial_prescan == (1, 2, 1, 2)
        assert frame.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, True], [False, False]])).all()

        frame = struct.MaskedFrame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            mask=mask,
            roe_corner=(0, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (frame == np.array([[3.0, 4.0], [1.0, 0.0]])).all()
        assert frame.original_roe_corner == (0, 0)
        assert frame.parallel_overscan == (1, 2, 0, 1)
        assert frame.serial_prescan == (0, 1, 1, 2)
        assert frame.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, False], [False, True]])).all()

        frame = struct.MaskedFrame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            mask=mask,
            roe_corner=(1, 1),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (frame == np.array([[0.0, 1.0], [4.0, 3.0]])).all()
        assert frame.original_roe_corner == (1, 1)
        assert frame.parallel_overscan == (0, 1, 1, 2)
        assert frame.serial_prescan == (1, 2, 0, 1)
        assert frame.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[True, False], [False, False]])).all()

        frame = struct.MaskedFrame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            mask=mask,
            roe_corner=(0, 1),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (frame == np.array([[4.0, 3.0], [0.0, 1.0]])).all()
        assert frame.original_roe_corner == (0, 1)
        assert frame.parallel_overscan == (1, 2, 1, 2)
        assert frame.serial_prescan == (0, 1, 0, 1)
        assert frame.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, False], [True, False]])).all()

    def test__from_frame__no_rotation_as_frame_is_correct_orientation(self):

        mask = struct.Mask.manual(mask_2d=[[False, True], [False, False]])

        frame = struct.Frame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            roe_corner=(1, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        frame = struct.MaskedFrame.from_frame(frame=frame, mask=mask)

        assert (frame == np.array([[1.0, 0.0], [3.0, 4.0]])).all()
        assert frame.original_roe_corner == (1, 0)
        assert frame.parallel_overscan == (0, 1, 0, 1)
        assert frame.serial_prescan == (1, 2, 1, 2)
        assert frame.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, True], [False, False]])).all()

        mask = struct.Mask.manual(mask_2d=[[False, True], [False, False]])

        frame = struct.Frame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            roe_corner=(0, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        frame = struct.MaskedFrame.from_frame(frame=frame, mask=mask)

        assert (frame == np.array([[3.0, 0.0], [1.0, 2.0]])).all()
        assert frame.original_roe_corner == (0, 0)
        assert frame.parallel_overscan == (1, 2, 0, 1)
        assert frame.serial_prescan == (0, 1, 1, 2)
        assert frame.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, True], [False, False]])).all()

    def test__ones_zeros_full__makes_frame_using_inputs(self):

        mask = struct.Mask.manual(mask_2d=[[False, True], [False, False]])

        frame = struct.MaskedFrame.full(
            fill_value=8.0,
            mask=mask,
            roe_corner=(1, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (frame == np.array([[8.0, 0.0], [8.0, 8.0]])).all()
        assert frame.original_roe_corner == (1, 0)
        assert frame.parallel_overscan == (0, 1, 0, 1)
        assert frame.serial_prescan == (1, 2, 1, 2)
        assert frame.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, True], [False, False]])).all()

        frame = struct.MaskedFrame.ones(
            mask=mask,
            roe_corner=(1, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (frame == np.array([[1.0, 0.0], [1.0, 1.0]])).all()
        assert frame.original_roe_corner == (1, 0)
        assert frame.parallel_overscan == (0, 1, 0, 1)
        assert frame.serial_prescan == (1, 2, 1, 2)
        assert frame.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, True], [False, False]])).all()

        mask = struct.Mask.manual(mask_2d=[[False, True], [False, False]])

        frame = struct.MaskedFrame.zeros(
            mask=mask,
            roe_corner=(1, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (frame == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
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

        euclid_frame = struct.MaskedEuclidFrame.top_left(array=euclid_data, mask=mask)

        assert euclid_frame.original_roe_corner == (0, 0)
        assert euclid_frame.shape_2d == (2086, 2119)
        assert (euclid_frame == np.zeros((2086, 2119))).all()
        assert euclid_frame.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_frame.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.serial_overscan == (0, 2086, 2099, 2119)
        assert euclid_frame.mask[2085, 0] == True
        assert euclid_frame.mask[2085, 1] == False

        euclid_frame = struct.MaskedEuclidFrame.top_right(array=euclid_data, mask=mask)

        assert euclid_frame.original_roe_corner == (0, 1)
        assert euclid_frame.shape_2d == (2086, 2119)
        assert (euclid_frame == np.zeros((2086, 2119))).all()
        assert euclid_frame.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_frame.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.serial_overscan == (0, 2086, 2099, 2119)
        assert euclid_frame.mask[2085, 2118] == True
        assert euclid_frame.mask[2085, 2117] == False

        euclid_frame = struct.MaskedEuclidFrame.bottom_left(
            array=euclid_data, mask=mask
        )

        assert euclid_frame.original_roe_corner == (1, 0)
        assert euclid_frame.shape_2d == (2086, 2119)
        assert (euclid_frame == np.zeros((2086, 2119))).all()
        assert euclid_frame.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_frame.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.serial_overscan == (0, 2086, 2099, 2119)
        assert euclid_frame.mask[0, 0] == True
        assert euclid_frame.mask[0, 1] == False

        euclid_frame = struct.MaskedEuclidFrame.bottom_right(
            array=euclid_data, mask=mask
        )

        assert euclid_frame.original_roe_corner == (1, 1)
        assert euclid_frame.shape_2d == (2086, 2119)
        assert (euclid_frame == np.zeros((2086, 2119))).all()
        assert euclid_frame.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_frame.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.serial_overscan == (0, 2086, 2099, 2119)
        assert euclid_frame.mask[0, 2118] == True
        assert euclid_frame.mask[0, 2117] == False

    def test__left_side__chooses_correct_frame_given_input(self, euclid_data):

        mask = np.full(shape=(2086, 2119), fill_value=False)
        mask[0, 0] = True

        frame = struct.MaskedEuclidFrame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text1", quadrant_id="E"
        )

        assert frame.original_roe_corner == (1, 0)
        assert frame.mask[0, 0] == True
        assert frame.mask[0, 1] == False

        frame = struct.MaskedEuclidFrame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text2", quadrant_id="E"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = struct.MaskedEuclidFrame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text3", quadrant_id="E"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = struct.MaskedEuclidFrame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text1", quadrant_id="F"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = struct.MaskedEuclidFrame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text2", quadrant_id="F"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = struct.MaskedEuclidFrame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text3", quadrant_id="F"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = struct.MaskedEuclidFrame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text1", quadrant_id="G"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = struct.MaskedEuclidFrame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text2", quadrant_id="G"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = struct.MaskedEuclidFrame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text3", quadrant_id="G"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = struct.MaskedEuclidFrame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text1", quadrant_id="H"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = struct.MaskedEuclidFrame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text2", quadrant_id="H"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = struct.MaskedEuclidFrame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text3", quadrant_id="H"
        )

        assert frame.original_roe_corner == (0, 0)

    def test__right_side__chooses_correct_frame_given_input(self, euclid_data):

        mask = np.full(shape=(2086, 2119), fill_value=False)
        mask[0, 0] = True

        frame = struct.MaskedEuclidFrame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text4", quadrant_id="E"
        )

        assert frame.original_roe_corner == (0, 1)
        assert frame.mask[2085, 2118] == True
        assert frame.mask[2085, 2117] == False

        frame = struct.MaskedEuclidFrame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text5", quadrant_id="E"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = struct.MaskedEuclidFrame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text6", quadrant_id="E"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = struct.MaskedEuclidFrame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text4", quadrant_id="F"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = struct.MaskedEuclidFrame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text5", quadrant_id="F"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = struct.MaskedEuclidFrame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text6", quadrant_id="F"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = struct.MaskedEuclidFrame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text4", quadrant_id="G"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = struct.MaskedEuclidFrame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text5", quadrant_id="G"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = struct.MaskedEuclidFrame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text6", quadrant_id="G"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = struct.MaskedEuclidFrame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text4", quadrant_id="H"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = struct.MaskedEuclidFrame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text5", quadrant_id="H"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = struct.MaskedEuclidFrame.ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text6", quadrant_id="H"
        )

        assert frame.original_roe_corner == (1, 1)


class TestBinnedAcross:
    def test__parallel__different_arrays__gives_frame_binned(self):

        frame = struct.Frame.manual(array=np.ones((3, 3)))

        assert (frame.binned_across_parallel == np.array([1.0, 1.0, 1.0])).all()

        frame = struct.Frame.manual(array=np.ones((4, 3)))

        assert (frame.binned_across_parallel == np.array([1.0, 1.0, 1.0])).all()

        frame = struct.Frame.manual(array=np.ones((3, 4)))

        assert (frame.binned_across_parallel == np.array([1.0, 1.0, 1.0, 1.0])).all()

        frame = struct.Frame.manual(
            array=np.array([[1.0, 6.0, 9.0], [2.0, 6.0, 9.0], [3.0, 6.0, 9.0]])
        )

        assert (frame.binned_across_parallel == np.array([2.0, 6.0, 9.0])).all()

    def test__parallel__same_as_above_but_with_mask(self):

        mask = np.ma.array(
            [[False, False, False], [False, False, False], [True, False, False]]
        )

        frame = struct.MaskedFrame.manual(
            array=np.array([[1.0, 6.0, 9.0], [2.0, 6.0, 9.0], [3.0, 6.0, 9.0]]),
            mask=mask,
        )

        assert (frame.binned_across_parallel == np.array([1.5, 6.0, 9.0])).all()

    def test__serial__different_arrays__gives_frame_binned(self):

        frame = struct.Frame.manual(array=np.ones((3, 3)))

        assert (frame.binned_across_serial == np.array([1.0, 1.0, 1.0])).all()

        frame = struct.Frame.manual(array=np.ones((4, 3)))

        assert (frame.binned_across_serial == np.array([1.0, 1.0, 1.0, 1.0])).all()

        frame = struct.Frame.manual(array=np.ones((3, 4)))

        assert (frame.binned_across_serial == np.array([1.0, 1.0, 1.0])).all()

        frame = struct.Frame.manual(
            array=np.array([[1.0, 2.0, 3.0], [6.0, 6.0, 6.0], [9.0, 9.0, 9.0]])
        )

        assert (frame.binned_across_serial == np.array([2.0, 6.0, 9.0])).all()

    def test__serial__same_as_above_but_with_mask(self):

        mask = np.ma.array(
            [[False, False, True], [False, False, False], [False, False, False]]
        )

        frame = struct.MaskedFrame.manual(
            array=np.array([[1.0, 2.0, 3.0], [6.0, 6.0, 6.0], [9.0, 9.0, 9.0]]),
            mask=mask,
        )

        assert (frame.binned_across_serial == np.array([1.5, 6.0, 9.0])).all()


class TestFrameRegions:
    def test__parallel_overscan_frame(self):

        arr = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        )

        frame = struct.Frame.manual(
            array=arr, roe_corner=(1, 0), parallel_overscan=(0, 1, 0, 1)
        )

        assert (frame.parallel_overscan_frame == np.array([[0.0]])).all()

        frame = struct.Frame.manual(
            array=arr, roe_corner=(1, 0), parallel_overscan=(0, 3, 0, 2)
        )

        assert (
            frame.parallel_overscan_frame
            == np.array([[0.0, 1.0], [3.0, 4.0], [6.0, 7.0]])
        ).all()

        frame = struct.Frame.manual(
            array=arr, roe_corner=(1, 0), parallel_overscan=(0, 4, 2, 3)
        )

        assert (
            frame.parallel_overscan_frame == np.array([[2.0], [5.0], [8.0], [11.0]])
        ).all()

    def test__parallel_overscan_binned_line(self):

        arr = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        )

        frame = struct.Frame.manual(
            array=arr, roe_corner=(1, 0), parallel_overscan=(0, 1, 0, 1)
        )

        assert (frame.parallel_overscan_binned_line == np.array([0.0])).all()

        frame = struct.Frame.manual(
            array=arr, roe_corner=(1, 0), parallel_overscan=(0, 3, 0, 2)
        )

        assert (frame.parallel_overscan_binned_line == np.array([0.5, 3.5, 6.5])).all()

        frame = struct.Frame.manual(
            array=arr, roe_corner=(1, 0), parallel_overscan=(0, 4, 2, 3)
        )

        assert (
            frame.parallel_overscan_binned_line == np.array([2.0, 5.0, 8.0, 11.0])
        ).all()

    def test__parallel_front_edge_of_region__extracts_rows_within_bottom_of_region(
        self
    ):

        frame = struct.Frame.ones(shape_2d=(3, 3), roe_corner=(1, 0))

        region = struct.Region(region=(0, 3, 0, 3))

        # Front edge is row 0, so for 1 row we extract 0 -> 1

        front_edge = frame.parallel_front_edge_of_region(region=region, rows=(0, 1))

        assert front_edge == (0, 1, 0, 3)

        # Front edge is row 0, so for 2 rows we extract 0 -> 2

        front_edge = frame.parallel_front_edge_of_region(region=region, rows=(0, 2))

        assert front_edge == (0, 2, 0, 3)

        # Front edge is row 0, so for these 2 rows we extract 1 ->2

        front_edge = frame.parallel_front_edge_of_region(region=region, rows=(1, 3))

        assert front_edge == (1, 3, 0, 3)

    def test__parallel_trails_of_region__extracts_rows_above_region(self):

        frame = struct.Frame.ones(shape_2d=(3, 3), roe_corner=(1, 0))

        region = struct.Region(
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

    def test__parallel_side_nearest_read_out_region(self):
        frame = struct.Frame.manual(array=np.ones((5, 5)), roe_corner=(1, 0))
        region = struct.Region(region=(1, 3, 0, 5))

        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, columns=(0, 1)
        )

        assert parallel_region == (0, 5, 0, 1)

        frame = struct.Frame.manual(array=np.ones((4, 4)), roe_corner=(1, 0))
        region = struct.Region(region=(1, 3, 0, 5))

        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, columns=(1, 3)
        )

        assert parallel_region == (0, 4, 1, 3)

        region = struct.Region(region=(1, 3, 2, 5))

        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, columns=(1, 3)
        )

        assert parallel_region == (0, 4, 3, 5)

        frame = struct.Frame.manual(array=np.ones((2, 5)), roe_corner=(1, 0))
        region = struct.Region(region=(1, 3, 0, 5))

        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, columns=(0, 1)
        )

        assert parallel_region == (0, 2, 0, 1)

    def test__serial_overscan_frame(self):

        arr = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        )

        frame = struct.Frame.manual(
            array=arr, roe_corner=(1, 0), serial_overscan=(0, 1, 0, 1)
        )

        assert (frame.serial_overscan_frame == np.array([[0.0]])).all()

        frame = struct.Frame.manual(
            array=arr, roe_corner=(1, 0), serial_overscan=(0, 3, 0, 2)
        )

        assert (
            frame.serial_overscan_frame
            == np.array([[0.0, 1.0], [3.0, 4.0], [6.0, 7.0]])
        ).all()

        frame = struct.Frame.manual(
            array=arr, roe_corner=(1, 0), serial_overscan=(0, 4, 2, 3)
        )

        assert (
            frame.serial_overscan_frame == np.array([[2.0], [5.0], [8.0], [11.0]])
        ).all()

    def test__serial_overscan_binned_line(self):

        arr = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        )

        frame = struct.Frame.manual(
            array=arr, roe_corner=(1, 0), serial_overscan=(0, 1, 0, 1)
        )

        assert (frame.serial_overscan_binned_line == np.array([0.0])).all()

        frame = struct.Frame.manual(
            array=arr, roe_corner=(1, 0), serial_overscan=(0, 3, 0, 2)
        )

        assert (frame.serial_overscan_binned_line == np.array([3.0, 4.0])).all()

        frame = struct.Frame.manual(
            array=arr, roe_corner=(1, 0), serial_overscan=(0, 4, 2, 3)
        )

        assert (frame.serial_overscan_binned_line == np.array([6.5])).all()

    def test__serial_front_edge_of_region__extracts_region_within_left_of_region(self):
        frame = struct.Frame.ones(shape_2d=(3, 3), roe_corner=(1, 0))

        region = struct.Region(
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

    def test__serial_trails_of_regions__extracts_region_to_right_of_region(self):
        frame = struct.Frame.ones(shape_2d=(3, 3), roe_corner=(1, 0))

        region = struct.Region(
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

    def test__serial_entie_rows_of_regioons__full_region_from_left_most_prescan_to_right_most_end_of_trails(
        self
    ):

        frame = struct.Frame.manual(array=np.ones((5, 5)), roe_corner=(1, 0))
        region = struct.Region(region=(1, 3, 0, 5))

        serial_region = frame.serial_entire_rows_of_region(region=region)

        assert serial_region == (1, 3, 0, 5)

        frame = struct.Frame.manual(array=np.ones((5, 25)), roe_corner=(1, 0))
        region = struct.Region(region=(1, 3, 0, 5))

        serial_region = frame.serial_entire_rows_of_region(region=region)

        assert serial_region == (1, 3, 0, 25)

        frame = struct.Frame.manual(array=np.ones((8, 55)), roe_corner=(1, 0))
        region = struct.Region(region=(3, 5, 5, 30))

        serial_region = frame.serial_entire_rows_of_region(region=region)

        assert serial_region == (3, 5, 0, 55)
