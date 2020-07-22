import numpy as np
import autocti as ac

from astropy.io import fits
import copy
import shutil
import os

path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))


def create_hst_fits(fits_path, hst_ccd, hst_ccd_0, hst_ccd_1, units):

    if os.path.exists(fits_path):
        shutil.rmtree(fits_path)

    os.makedirs(fits_path)

    new_hdul = fits.HDUList()

    new_hdul.append(fits.ImageHDU(hst_ccd))
    new_hdul.append(fits.ImageHDU(hst_ccd_0))
    new_hdul.append(fits.ImageHDU(hst_ccd))
    new_hdul.append(fits.ImageHDU(hst_ccd))
    new_hdul.append(fits.ImageHDU(hst_ccd_1))
    new_hdul.append(fits.ImageHDU(hst_ccd))

    if units in "COUNTS":
        new_hdul[1].header.set("BUNIT", "COUNTS", "brightness units")
        new_hdul[4].header.set("BUNIT", "COUNTS", "brightness units")
    elif units in "CPS":
        new_hdul[1].header.set("BUNIT", "CPS", "brightness units")
        new_hdul[4].header.set("BUNIT", "CPS", "brightness units")

    new_hdul[1].header.set(
        "BSCALE", 2.0, "scale factor for array value to physical value"
    )
    new_hdul[1].header.set("BZERO", 10.0, "physical value for an array value of zero")
    new_hdul[1].header.set("EXPTIME", 1000.0, "exposure duration (seconds)--calculated")
    new_hdul[4].header.set(
        "BSCALE", 2.0, "scale factor for array value to physical value"
    )
    new_hdul[4].header.set("BZERO", 10.0, "physical value for an array value of zero")
    new_hdul[4].header.set("EXPTIME", 2000.0, "exposure duration (seconds)--calculated")

    new_hdul.writeto(f"{fits_path}/hst_ccd.fits")


class TestFrameAPI:
    def test__hst_frame_for_left_and_right_quandrants__loads_data_and_dimensions(
        self, hst_quadrant
    ):

        hst_frame = ac.HSTFrame.left(
            array_electrons=hst_quadrant,
            parallel_size=2068,
            serial_size=2072,
            serial_prescan_size=24,
            parallel_overscan_size=20,
        )

        assert hst_frame.original_roe_corner == (1, 0)
        assert hst_frame.shape_2d == (2068, 2072)
        assert (hst_frame == np.zeros((2068, 2072))).all()
        assert hst_frame.scans.parallel_overscan == (2048, 2068, 24, 2072)
        assert hst_frame.scans.serial_prescan == (0, 2068, 0, 24)

        hst_frame = ac.HSTFrame.left(
            array_electrons=hst_quadrant,
            parallel_size=2070,
            serial_size=2072,
            serial_prescan_size=28,
            parallel_overscan_size=10,
        )

        assert hst_frame.original_roe_corner == (1, 0)
        assert hst_frame.shape_2d == (2068, 2072)
        assert (hst_frame == np.zeros((2068, 2072))).all()
        assert hst_frame.scans.parallel_overscan == (2060, 2070, 28, 2072)
        assert hst_frame.scans.serial_prescan == (0, 2070, 0, 28)

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
        assert hst_frame.scans.parallel_overscan == (2048, 2068, 24, 2072)
        assert hst_frame.scans.serial_prescan == (0, 2068, 0, 24)

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
        assert hst_frame.scans.parallel_overscan == (2060, 2070, 28, 2072)
        assert hst_frame.scans.serial_prescan == (0, 2070, 0, 28)

    def test__from_ccd__chooses_correct_frame_given_quadrant_letter(self, hst_ccd):

        frame = ac.HSTFrame.from_ccd(array_electrons=hst_ccd, quadrant_letter="B")

        assert frame.original_roe_corner == (1, 0)
        assert frame.shape_2d == (2068, 2072)

        frame = ac.HSTFrame.from_ccd(array_electrons=hst_ccd, quadrant_letter="C")

        assert frame.original_roe_corner == (1, 0)
        assert frame.shape_2d == (2068, 2072)

        frame = ac.HSTFrame.from_ccd(array_electrons=hst_ccd, quadrant_letter="A")

        assert frame.original_roe_corner == (1, 1)
        assert frame.shape_2d == (2068, 2072)

        frame = ac.HSTFrame.from_ccd(array_electrons=hst_ccd, quadrant_letter="D")

        assert frame.original_roe_corner == (1, 1)
        assert frame.shape_2d == (2068, 2072)

    def test__from_fits__in_counts__uses_fits_header_correctly_converts_and_picks_correct_quadrant(
        self, hst_ccd
    ):

        fits_path = "{}/files/hst/".format(os.path.dirname(os.path.realpath(__file__)))

        hst_ccd_0 = copy.copy(hst_ccd)
        hst_ccd_0[0, 0] = 10.0
        hst_ccd_0[0, 4143] = 20.0

        hst_ccd_1 = copy.copy(hst_ccd)
        hst_ccd_1[0, 0] = 30.0
        hst_ccd_1[0, 4143] = 40.0

        create_hst_fits(
            fits_path=fits_path,
            hst_ccd=hst_ccd,
            hst_ccd_0=hst_ccd_0,
            hst_ccd_1=hst_ccd_1,
            units="COUNTS",
        )

        frame = ac.HSTFrame.from_fits(
            file_path=f"{fits_path}/hst_ccd.fits", quadrant_letter="B"
        )

        assert frame[0, 0] == (10.0 * 2.0) + 10.0
        assert frame.in_counts[0, 0] == 10.0
        assert frame.original_roe_corner == (1, 0)
        assert frame.shape_2d == (2068, 2072)

        frame = ac.HSTFrame.from_fits(
            file_path=f"{fits_path}/hst_ccd.fits", quadrant_letter="A"
        )

        assert frame[0, 0] == (20.0 * 2.0) + 10.0
        assert frame.original_roe_corner == (1, 1)
        assert frame.shape_2d == (2068, 2072)

        frame = ac.HSTFrame.from_fits(
            file_path=f"{fits_path}/hst_ccd.fits", quadrant_letter="C"
        )

        assert frame[0, 0] == (30.0 * 2.0) + 10.0
        assert frame.original_roe_corner == (1, 0)
        assert frame.shape_2d == (2068, 2072)

        frame = ac.HSTFrame.from_fits(
            file_path=f"{fits_path}/hst_ccd.fits", quadrant_letter="D"
        )

        assert frame[0, 0] == (40.0 * 2.0) + 10.0
        assert frame.original_roe_corner == (1, 1)
        assert frame.shape_2d == (2068, 2072)

    def test__from_fits__in_counts_per_second__uses_fits_header_correctly_converts_and_picks_correct_quadrant(
        self, hst_ccd
    ):

        fits_path = "{}/files/hst/".format(os.path.dirname(os.path.realpath(__file__)))

        hst_ccd_0 = copy.copy(hst_ccd)
        hst_ccd_0[0, 0] = 10.0
        hst_ccd_0[0, 4143] = 20.0

        hst_ccd_1 = copy.copy(hst_ccd)
        hst_ccd_1[0, 0] = 30.0
        hst_ccd_1[0, 4143] = 40.0

        create_hst_fits(
            fits_path=fits_path,
            hst_ccd=hst_ccd,
            hst_ccd_0=hst_ccd_0,
            hst_ccd_1=hst_ccd_1,
            units="CPS",
        )

        frame = ac.HSTFrame.from_fits(
            file_path=f"{fits_path}/hst_ccd.fits", quadrant_letter="B"
        )

        assert frame[0, 0] == (10.0 * 1000.0 * 2.0) + 10.0
        assert frame.original_roe_corner == (1, 0)
        assert frame.shape_2d == (2068, 2072)

        frame = ac.HSTFrame.from_fits(
            file_path=f"{fits_path}/hst_ccd.fits", quadrant_letter="A"
        )

        assert frame[0, 0] == (20.0 * 1000.0 * 2.0) + 10.0
        assert frame.original_roe_corner == (1, 1)
        assert frame.shape_2d == (2068, 2072)

        frame = ac.HSTFrame.from_fits(
            file_path=f"{fits_path}/hst_ccd.fits", quadrant_letter="C"
        )

        assert frame[0, 0] == (30.0 * 2000.0 * 2.0) + 10.0
        assert frame.original_roe_corner == (1, 0)
        assert frame.shape_2d == (2068, 2072)

        frame = ac.HSTFrame.from_fits(
            file_path=f"{fits_path}/hst_ccd.fits", quadrant_letter="D"
        )

        assert frame[0, 0] == (40.0 * 2000.0 * 2.0) + 10.0
        assert frame.original_roe_corner == (1, 1)
        assert frame.shape_2d == (2068, 2072)


class TestMaskedFrameAPI:
    def test__hst_frame_for_left_and_right_quandrants__loads_data_and_dimensions(
        self, hst_quadrant
    ):

        mask = np.full(shape=(2068, 2072), fill_value=False)
        mask[0, 0] = True

        hst_frame = ac.MaskedHSTFrame.left(
            array=hst_quadrant,
            mask=mask,
            parallel_size=2068,
            serial_size=2072,
            serial_prescan_size=24,
            parallel_overscan_size=20,
        )

        assert hst_frame.original_roe_corner == (1, 0)
        assert hst_frame.shape_2d == (2068, 2072)
        assert (hst_frame == np.zeros((2068, 2072))).all()
        assert hst_frame.scans.parallel_overscan == (2048, 2068, 24, 2072)
        assert hst_frame.scans.serial_prescan == (0, 2068, 0, 24)
        assert hst_frame.mask[0, 0] == True
        assert hst_frame.mask[0, 1] == False

        hst_frame = ac.MaskedHSTFrame.left(
            array=hst_quadrant,
            mask=mask,
            parallel_size=2070,
            serial_size=2072,
            serial_prescan_size=28,
            parallel_overscan_size=10,
        )

        assert hst_frame.original_roe_corner == (1, 0)
        assert hst_frame.shape_2d == (2068, 2072)
        assert (hst_frame == np.zeros((2068, 2072))).all()
        assert hst_frame.scans.parallel_overscan == (2060, 2070, 28, 2072)
        assert hst_frame.scans.serial_prescan == (0, 2070, 0, 28)
        assert hst_frame.mask[0, 0] == True
        assert hst_frame.mask[0, 1] == False

        hst_frame = ac.MaskedHSTFrame.right(
            array=hst_quadrant,
            mask=mask,
            parallel_size=2068,
            serial_size=2072,
            serial_prescan_size=24,
            parallel_overscan_size=20,
        )

        assert hst_frame.original_roe_corner == (1, 1)
        assert hst_frame.shape_2d == (2068, 2072)
        assert (hst_frame == np.zeros((2068, 2072))).all()
        assert hst_frame.scans.parallel_overscan == (2048, 2068, 24, 2072)
        assert hst_frame.scans.serial_prescan == (0, 2068, 0, 24)
        assert hst_frame.mask[0, -1] == True
        assert hst_frame.mask[0, 1] == False

        hst_frame = ac.MaskedHSTFrame.right(
            array=hst_quadrant,
            mask=mask,
            parallel_size=2070,
            serial_size=2072,
            serial_prescan_size=28,
            parallel_overscan_size=10,
        )

        assert hst_frame.original_roe_corner == (1, 1)
        assert hst_frame.shape_2d == (2068, 2072)
        assert (hst_frame == np.zeros((2068, 2072))).all()
        assert hst_frame.scans.parallel_overscan == (2060, 2070, 28, 2072)
        assert hst_frame.scans.serial_prescan == (0, 2070, 0, 28)
        assert hst_frame.mask[0, -1] == True
        assert hst_frame.mask[0, 1] == False

    def test__from_ccd__chooses_correct_frame_given_quadrant_letter(self, hst_ccd):

        mask = np.full(shape=(2068, 4144), fill_value=False)
        mask[0, 0] = True
        mask[0, 2072] = True

        frame = ac.MaskedHSTFrame.from_ccd(
            array=hst_ccd, quadrant_letter="B", mask=mask
        )

        assert frame.original_roe_corner == (1, 0)
        assert frame.shape_2d == (2068, 2072)
        assert frame.mask[0, 0] == True

        frame = ac.MaskedHSTFrame.from_ccd(
            array=hst_ccd, quadrant_letter="C", mask=mask
        )

        assert frame.original_roe_corner == (1, 0)
        assert frame.shape_2d == (2068, 2072)
        assert frame.mask[0, 0] == True

        frame = ac.MaskedHSTFrame.from_ccd(
            array=hst_ccd, quadrant_letter="A", mask=mask
        )

        assert frame.original_roe_corner == (1, 1)
        assert frame.shape_2d == (2068, 2072)
        assert frame.mask[0, -1] == True
        assert frame.mask[0, 0] == False

        frame = ac.MaskedHSTFrame.from_ccd(
            array=hst_ccd, quadrant_letter="D", mask=mask
        )

        assert frame.original_roe_corner == (1, 1)
        assert frame.shape_2d == (2068, 2072)
        assert frame.mask[0, -1] == True

    def test__from_fits__in_counts__uses_fits_header_correctly_converts_and_picks_correct_quadrant(
        self, hst_ccd
    ):

        fits_path = "{}/files/hst/".format(os.path.dirname(os.path.realpath(__file__)))

        hst_ccd_0 = copy.copy(hst_ccd)
        hst_ccd_0[0, 0] = 10.0
        hst_ccd_0[0, 4143] = 20.0

        hst_ccd_1 = copy.copy(hst_ccd)
        hst_ccd_1[0, 0] = 30.0
        hst_ccd_1[0, 4143] = 40.0

        create_hst_fits(
            fits_path=fits_path,
            hst_ccd=hst_ccd,
            hst_ccd_0=hst_ccd_0,
            hst_ccd_1=hst_ccd_1,
            units="COUNTS",
        )

        mask = np.full(shape=(2068, 4144), fill_value=False)
        mask[0, 1] = True
        mask[0, 4142] = True

        frame = ac.MaskedHSTFrame.from_fits(
            file_path=f"{fits_path}/hst_ccd.fits", quadrant_letter="B", mask=mask
        )

        assert frame[0, 0] == (10.0 * 2.0) + 10.0
        assert frame.original_roe_corner == (1, 0)
        assert frame.shape_2d == (2068, 2072)
        assert frame.mask[0, 1] == True

        frame = ac.MaskedHSTFrame.from_fits(
            file_path=f"{fits_path}/hst_ccd.fits", quadrant_letter="A", mask=mask
        )

        assert frame[0, 0] == (20.0 * 2.0) + 10.0
        assert frame.original_roe_corner == (1, 1)
        assert frame.shape_2d == (2068, 2072)
        assert frame.mask[0, 1] == True

        frame = ac.MaskedHSTFrame.from_fits(
            file_path=f"{fits_path}/hst_ccd.fits", quadrant_letter="C", mask=mask
        )

        assert frame[0, 0] == (30.0 * 2.0) + 10.0
        assert frame.original_roe_corner == (1, 0)
        assert frame.shape_2d == (2068, 2072)
        assert frame.mask[0, 1] == True

        frame = ac.MaskedHSTFrame.from_fits(
            file_path=f"{fits_path}/hst_ccd.fits", quadrant_letter="D", mask=mask
        )

        assert frame[0, 0] == (40.0 * 2.0) + 10.0
        assert frame.original_roe_corner == (1, 1)
        assert frame.shape_2d == (2068, 2072)
        assert frame.mask[0, 1] == True
