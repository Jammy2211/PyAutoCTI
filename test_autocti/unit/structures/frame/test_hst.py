import os

import numpy as np
import autocti as ac


path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))


class TestFrameAPI:
    def test__hst_frame_for_left_and_right_quandrants__loads_data_and_dimensions(
        self, hst_quadrant
    ):

        hst_frame = ac.HSTFrame.left(
            array=hst_quadrant,
            parallel_size=2068,
            serial_size=2072,
            serial_prescan_size=24,
            parallel_overscan_size=20,
        )

        assert hst_frame.original_roe_corner == (1, 0)
        assert hst_frame.shape_2d == (2068, 2072)
        assert (hst_frame == np.zeros((2068, 2072))).all()
        assert hst_frame.parallel_overscan == (2048, 2068, 24, 2072)
        assert hst_frame.serial_prescan == (0, 2068, 0, 24)

        hst_frame = ac.HSTFrame.left(
            array=hst_quadrant,
            parallel_size=2070,
            serial_size=2072,
            serial_prescan_size=28,
            parallel_overscan_size=10,
        )

        assert hst_frame.original_roe_corner == (1, 0)
        assert hst_frame.shape_2d == (2068, 2072)
        assert (hst_frame == np.zeros((2068, 2072))).all()
        assert hst_frame.parallel_overscan == (2060, 2070, 28, 2072)
        assert hst_frame.serial_prescan == (0, 2070, 0, 28)

        hst_frame = ac.HSTFrame.right(
            array=hst_quadrant,
            parallel_size=2068,
            serial_size=2072,
            serial_prescan_size=24,
            parallel_overscan_size=20,
        )

        assert hst_frame.original_roe_corner == (1, 1)
        assert hst_frame.shape_2d == (2068, 2072)
        assert (hst_frame == np.zeros((2068, 2072))).all()
        assert hst_frame.parallel_overscan == (2048, 2068, 24, 2072)
        assert hst_frame.serial_prescan == (0, 2068, 0, 24)

        hst_frame = ac.HSTFrame.right(
            array=hst_quadrant,
            parallel_size=2070,
            serial_size=2072,
            serial_prescan_size=28,
            parallel_overscan_size=10,
        )

        assert hst_frame.original_roe_corner == (1, 1)
        assert hst_frame.shape_2d == (2068, 2072)
        assert (hst_frame == np.zeros((2068, 2072))).all()
        assert hst_frame.parallel_overscan == (2060, 2070, 28, 2072)
        assert hst_frame.serial_prescan == (0, 2070, 0, 28)

    def test__from_ccd__chooses_correct_frame_given_quadrant_letter(self, hst_ccd):

        frame = ac.HSTFrame.from_ccd(array=hst_ccd, quadrant_letter="B")

        assert frame.original_roe_corner == (1, 0)
        assert frame.shape_2d == (2068, 2072)

        frame = ac.HSTFrame.from_ccd(array=hst_ccd, quadrant_letter="C")

        assert frame.original_roe_corner == (1, 0)
        assert frame.shape_2d == (2068, 2072)

        frame = ac.HSTFrame.from_ccd(array=hst_ccd, quadrant_letter="A")

        assert frame.original_roe_corner == (1, 1)
        assert frame.shape_2d == (2068, 2072)

        frame = ac.HSTFrame.from_ccd(array=hst_ccd, quadrant_letter="D")

        assert frame.original_roe_corner == (1, 1)
        assert frame.shape_2d == (2068, 2072)


class TestMaskedFrameAPI:
    def test__hst_frame_for_four_quandrants__loads_data_and_dimensions(self, hst_data):

        mask = np.full(shape=(2086, 2119), fill_value=False)
        mask[0, 0] = True

        hst_frame = ac.MaskedHSTFrame.top_left(array=hst_data, mask=mask)

        assert hst_frame.original_roe_corner == (0, 0)
        assert hst_frame.shape_2d == (2086, 2119)
        assert (hst_frame == np.zeros((2086, 2119))).all()
        assert hst_frame.parallel_overscan == (2066, 2086, 51, 2099)
        assert hst_frame.serial_prescan == (0, 2086, 0, 51)
        assert hst_frame.serial_overscan == (0, 2086, 2099, 2119)
        assert hst_frame.mask[2085, 0] == True
        assert hst_frame.mask[2085, 1] == False

        hst_frame = ac.MaskedHSTFrame.top_right(array=hst_data, mask=mask)

        assert hst_frame.original_roe_corner == (0, 1)
        assert hst_frame.shape_2d == (2086, 2119)
        assert (hst_frame == np.zeros((2086, 2119))).all()
        assert hst_frame.parallel_overscan == (2066, 2086, 51, 2099)
        assert hst_frame.serial_prescan == (0, 2086, 0, 51)
        assert hst_frame.serial_overscan == (0, 2086, 2099, 2119)
        assert hst_frame.mask[2085, 2118] == True
        assert hst_frame.mask[2085, 2117] == False

        hst_frame = ac.MaskedHSTFrame.bottom_left(array=hst_data, mask=mask)

        assert hst_frame.original_roe_corner == (1, 0)
        assert hst_frame.shape_2d == (2086, 2119)
        assert (hst_frame == np.zeros((2086, 2119))).all()
        assert hst_frame.parallel_overscan == (2066, 2086, 51, 2099)
        assert hst_frame.serial_prescan == (0, 2086, 0, 51)
        assert hst_frame.serial_overscan == (0, 2086, 2099, 2119)
        assert hst_frame.mask[0, 0] == True
        assert hst_frame.mask[0, 1] == False

        hst_frame = ac.MaskedHSTFrame.bottom_right(array=hst_data, mask=mask)

        assert hst_frame.original_roe_corner == (1, 1)
        assert hst_frame.shape_2d == (2086, 2119)
        assert (hst_frame == np.zeros((2086, 2119))).all()
        assert hst_frame.parallel_overscan == (2066, 2086, 51, 2099)
        assert hst_frame.serial_prescan == (0, 2086, 0, 51)
        assert hst_frame.serial_overscan == (0, 2086, 2099, 2119)
        assert hst_frame.mask[0, 2118] == True
        assert hst_frame.mask[0, 2117] == False

    def test__left_side__chooses_correct_frame_given_input(self, hst_data):

        mask = np.full(shape=(2086, 2119), fill_value=False)
        mask[0, 0] = True

        frame = ac.MaskedHSTFrame.from_ccd_and_quadrant_id(
            array=hst_data, mask=mask, ccd_id="text1", quadrant_id="E"
        )

        assert frame.original_roe_corner == (1, 0)
        assert frame.mask[0, 0] == True
        assert frame.mask[0, 1] == False

        frame = ac.MaskedHSTFrame.from_ccd_and_quadrant_id(
            array=hst_data, mask=mask, ccd_id="text2", quadrant_id="E"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = ac.MaskedHSTFrame.from_ccd_and_quadrant_id(
            array=hst_data, mask=mask, ccd_id="text3", quadrant_id="E"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = ac.MaskedHSTFrame.from_ccd_and_quadrant_id(
            array=hst_data, mask=mask, ccd_id="text1", quadrant_id="F"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = ac.MaskedHSTFrame.from_ccd_and_quadrant_id(
            array=hst_data, mask=mask, ccd_id="text2", quadrant_id="F"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = ac.MaskedHSTFrame.from_ccd_and_quadrant_id(
            array=hst_data, mask=mask, ccd_id="text3", quadrant_id="F"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = ac.MaskedHSTFrame.from_ccd_and_quadrant_id(
            array=hst_data, mask=mask, ccd_id="text1", quadrant_id="G"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = ac.MaskedHSTFrame.from_ccd_and_quadrant_id(
            array=hst_data, mask=mask, ccd_id="text2", quadrant_id="G"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = ac.MaskedHSTFrame.from_ccd_and_quadrant_id(
            array=hst_data, mask=mask, ccd_id="text3", quadrant_id="G"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = ac.MaskedHSTFrame.from_ccd_and_quadrant_id(
            array=hst_data, mask=mask, ccd_id="text1", quadrant_id="H"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = ac.MaskedHSTFrame.from_ccd_and_quadrant_id(
            array=hst_data, mask=mask, ccd_id="text2", quadrant_id="H"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = ac.MaskedHSTFrame.from_ccd_and_quadrant_id(
            array=hst_data, mask=mask, ccd_id="text3", quadrant_id="H"
        )

        assert frame.original_roe_corner == (0, 0)

    def test__right_side__chooses_correct_frame_given_input(self, hst_data):

        mask = np.full(shape=(2086, 2119), fill_value=False)
        mask[0, 0] = True

        frame = ac.MaskedHSTFrame.from_ccd_and_quadrant_id(
            array=hst_data, mask=mask, ccd_id="text4", quadrant_id="E"
        )

        assert frame.original_roe_corner == (0, 1)
        assert frame.mask[2085, 2118] == True
        assert frame.mask[2085, 2117] == False

        frame = ac.MaskedHSTFrame.from_ccd_and_quadrant_id(
            array=hst_data, mask=mask, ccd_id="text5", quadrant_id="E"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = ac.MaskedHSTFrame.from_ccd_and_quadrant_id(
            array=hst_data, mask=mask, ccd_id="text6", quadrant_id="E"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = ac.MaskedHSTFrame.from_ccd_and_quadrant_id(
            array=hst_data, mask=mask, ccd_id="text4", quadrant_id="F"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = ac.MaskedHSTFrame.from_ccd_and_quadrant_id(
            array=hst_data, mask=mask, ccd_id="text5", quadrant_id="F"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = ac.MaskedHSTFrame.from_ccd_and_quadrant_id(
            array=hst_data, mask=mask, ccd_id="text6", quadrant_id="F"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = ac.MaskedHSTFrame.from_ccd_and_quadrant_id(
            array=hst_data, mask=mask, ccd_id="text4", quadrant_id="G"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = ac.MaskedHSTFrame.from_ccd_and_quadrant_id(
            array=hst_data, mask=mask, ccd_id="text5", quadrant_id="G"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = ac.MaskedHSTFrame.from_ccd_and_quadrant_id(
            array=hst_data, mask=mask, ccd_id="text6", quadrant_id="G"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = ac.MaskedHSTFrame.from_ccd_and_quadrant_id(
            array=hst_data, mask=mask, ccd_id="text4", quadrant_id="H"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = ac.MaskedHSTFrame.from_ccd_and_quadrant_id(
            array=hst_data, mask=mask, ccd_id="text5", quadrant_id="H"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = ac.MaskedHSTFrame.from_ccd_and_quadrant_id(
            array=hst_data, mask=mask, ccd_id="text6", quadrant_id="H"
        )

        assert frame.original_roe_corner == (1, 1)
