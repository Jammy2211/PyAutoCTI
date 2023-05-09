import numpy as np
from astropy.io import fits
import copy
import shutil
import os
from os import path
import pytest

import autocti as ac


def create_acs_fits(
    fits_path, acs_ccd, acs_ccd_0, acs_ccd_1, units, bias_file_path=None
):
    if path.exists(fits_path):
        shutil.rmtree(fits_path)

    os.makedirs(fits_path)

    hdu_list = fits.HDUList()

    hdu_list.append(fits.ImageHDU(acs_ccd))
    hdu_list.append(fits.ImageHDU(acs_ccd_0))
    hdu_list.append(fits.ImageHDU(acs_ccd))
    hdu_list.append(fits.ImageHDU(acs_ccd))
    hdu_list.append(fits.ImageHDU(acs_ccd_1))
    hdu_list.append(fits.ImageHDU(acs_ccd))

    hdu_list[0].header.set("CCDGAIN", 1.0, "Instrument GAIN")
    hdu_list[0].header.set("TELESCOP", "HST", "Telescope Name")
    hdu_list[0].header.set("INSTRUME", "ACS", "Instrument Name")
    hdu_list[0].header.set("EXPTIME", 1000.0, "exposure duration (seconds)--calculated")
    hdu_list[0].header.set(
        "DATE-OBS", "2000-01-01", "UT date of start of observation (yyyy-mm-dd)"
    )
    hdu_list[0].header.set(
        "TIME-OBS", "00:00:00", "UT time of start of observation (hh:mm:ss)"
    )
    hdu_list[0].header.set("BIASFILE", f"jref${bias_file_path}", "Bias file name")

    if units in "COUNTS":
        hdu_list[1].header.set("BUNIT", "COUNTS", "brightness units")
        hdu_list[4].header.set("BUNIT", "COUNTS", "brightness units")
    elif units in "CPS":
        hdu_list[1].header.set("BUNIT", "CPS", "brightness units")
        hdu_list[4].header.set("BUNIT", "CPS", "brightness units")

    hdu_list[1].header.set(
        "BSCALE", 2.0, "scale factor for array value to physical value"
    )
    hdu_list[1].header.set("BZERO", 10.0, "physical value for an array value of zero")

    hdu_list[4].header.set(
        "BSCALE", 2.0, "scale factor for array value to physical value"
    )
    hdu_list[4].header.set("BZERO", 10.0, "physical value for an array value of zero")

    hdu_list.writeto(path.join(fits_path, "acs_ccd.fits"))


def create_acs_bias_fits(fits_path, bias_ccd, bias_ccd_0, bias_ccd_1):
    hdu_list = fits.HDUList()

    hdu_list.append(fits.ImageHDU(bias_ccd))
    hdu_list.append(fits.ImageHDU(bias_ccd_0))
    hdu_list.append(fits.ImageHDU(bias_ccd))
    hdu_list.append(fits.ImageHDU(bias_ccd))
    hdu_list.append(fits.ImageHDU(bias_ccd_1))
    hdu_list.append(fits.ImageHDU(bias_ccd))

    hdu_list[0].header.set("CCDGAIN", 1.0, "Instrument GAIN")
    hdu_list[1].header.set("BUNIT", "COUNTS", "brightness units")
    hdu_list[4].header.set("BUNIT", "COUNTS", "brightness units")

    hdu_list.writeto(path.join(fits_path, "acs_bias_ccd.fits"))


def test__from_fits__reads_header_from_header_correctly(acs_ccd):
    fits_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "acs"
    )

    create_acs_fits(
        fits_path=fits_path,
        acs_ccd=acs_ccd,
        acs_ccd_0=acs_ccd,
        acs_ccd_1=acs_ccd,
        units="COUNTS",
    )

    file_path = path.join(fits_path, "acs_ccd.fits")

    array = ac.acs.ImageACS.from_fits(file_path=file_path, quadrant_letter="B")

    assert array.header.exposure_time == 1000.0
    assert array.header.date_of_observation == "2000-01-01"
    assert array.header.time_of_observation == "00:00:00"
    assert array.header.modified_julian_date == 51544.0

    array = ac.acs.ImageACS.from_fits(file_path=file_path, quadrant_letter="C")

    assert array.header.exposure_time == 1000.0
    assert array.header.date_of_observation == "2000-01-01"
    assert array.header.time_of_observation == "00:00:00"
    assert array.header.modified_julian_date == 51544.0


def test__from_fits__in_counts__uses_fits_header_correctly_converts_and_picks_correct_quadrant(
    acs_ccd,
):
    fits_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "acs"
    )

    file_path = path.join(fits_path, "acs_ccd.fits")

    acs_ccd_0 = copy.copy(acs_ccd)
    acs_ccd_0[0, 0] = 10.0
    acs_ccd_0[0, -1] = 20.0

    acs_ccd_1 = copy.copy(acs_ccd)
    acs_ccd_1[-1, 0] = 30.0
    acs_ccd_1[-1, -1] = 40.0

    create_acs_fits(
        fits_path=fits_path,
        acs_ccd=acs_ccd,
        acs_ccd_0=acs_ccd_0,
        acs_ccd_1=acs_ccd_1,
        units="COUNTS",
    )

    array = ac.acs.ImageACS.from_fits(
        file_path=file_path, quadrant_letter="A", use_calibrated_gain=False
    )

    assert array.native[0, 0] == (30.0 * 2.0) + 10.0
    assert array.in_counts.native[0, 0] == 30.0
    assert array.shape_native == (2068, 2072)

    array_original = array.header.array_electrons_to_original(
        array=array, use_calibrated_gain=False
    )

    assert array_original.native[0, 0] == 30.0

    array = ac.acs.ImageACS.from_fits(
        file_path=file_path, quadrant_letter="B", use_calibrated_gain=False
    )

    assert array.native[0, 0] == (40.0 * 2.0) + 10.0
    assert array.shape_native == (2068, 2072)

    array_original = array.header.array_electrons_to_original(
        array=array, use_calibrated_gain=False
    )

    assert array_original.native[0, 0] == 40.0

    array = ac.acs.ImageACS.from_fits(
        file_path=file_path, quadrant_letter="C", use_calibrated_gain=False
    )

    assert array.native[0, 0] == (10.0 * 2.0) + 10.0
    assert array.shape_native == (2068, 2072)

    array_original = array.header.array_electrons_to_original(
        array=array, use_calibrated_gain=False
    )

    assert array_original.native[0, 0] == 10.0

    array = ac.acs.ImageACS.from_fits(
        file_path=file_path, quadrant_letter="D", use_calibrated_gain=False
    )

    assert array.native[0, 0] == (20.0 * 2.0) + 10.0
    assert array.shape_native == (2068, 2072)

    array_original = array.header.array_electrons_to_original(
        array=array, use_calibrated_gain=False
    )

    assert array_original.native[0, 0] == 20.0


def test__from_fits__in_counts_per_second__uses_fits_header_correctly_converts_and_picks_correct_quadrant(
    acs_ccd,
):
    fits_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "acs"
    )

    file_path = path.join(fits_path, "acs_ccd.fits")

    acs_ccd_0 = copy.copy(acs_ccd)
    acs_ccd_0[0, 0] = 10.0
    acs_ccd_0[0, -1] = 20.0

    acs_ccd_1 = copy.copy(acs_ccd)
    acs_ccd_1[-1, 0] = 30.0
    acs_ccd_1[-1, -1] = 40.0

    create_acs_fits(
        fits_path=fits_path,
        acs_ccd=acs_ccd,
        acs_ccd_0=acs_ccd_0,
        acs_ccd_1=acs_ccd_1,
        units="CPS",
    )

    array = ac.acs.ImageACS.from_fits(
        file_path=file_path, quadrant_letter="A", use_calibrated_gain=False
    )

    assert array.native[0, 0] == (30.0 * 1000.0 * 2.0) + 10.0
    assert array.shape_native == (2068, 2072)

    array_original = array.header.array_electrons_to_original(
        array=array, use_calibrated_gain=False
    )

    assert array_original.native[0, 0] == 30.0

    array = ac.acs.ImageACS.from_fits(
        file_path=file_path, quadrant_letter="B", use_calibrated_gain=False
    )

    assert array.native[0, 0] == (40.0 * 1000.0 * 2.0) + 10.0
    assert array.shape_native == (2068, 2072)

    array_original = array.header.array_electrons_to_original(
        array=array, use_calibrated_gain=False
    )

    assert array_original.native[0, 0] == 40.0

    array = ac.acs.ImageACS.from_fits(
        file_path=file_path, quadrant_letter="C", use_calibrated_gain=False
    )

    assert array.native[0, 0] == (10.0 * 1000.0 * 2.0) + 10.0
    assert array.shape_native == (2068, 2072)

    array_original = array.header.array_electrons_to_original(
        array=array, use_calibrated_gain=False
    )

    assert array_original.native[0, 0] == 10.0

    array = ac.acs.ImageACS.from_fits(
        file_path=file_path, quadrant_letter="D", use_calibrated_gain=False
    )

    assert array.native[0, 0] == (20.0 * 1000.0 * 2.0) + 10.0
    assert array.shape_native == (2068, 2072)

    array_original = array.header.array_electrons_to_original(
        array=array, use_calibrated_gain=False
    )

    assert array_original.native[0, 0] == 20.0


def test__from_fits__in_counts__uses_bias_prescan_correctly(acs_ccd):
    fits_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "acs"
    )

    file_path = path.join(fits_path, "acs_ccd.fits")

    acs_ccd_0 = copy.copy(acs_ccd)
    acs_ccd_0[0, 0] = 10.0
    acs_ccd_0[0, -1] = 20.0

    acs_ccd_1 = copy.copy(acs_ccd)
    acs_ccd_1[-1, 0] = 30.0
    acs_ccd_1[-1, -1] = 40.0

    create_acs_fits(
        fits_path=fits_path,
        acs_ccd=acs_ccd,
        acs_ccd_0=acs_ccd_0,
        acs_ccd_1=acs_ccd_1,
        units="COUNTS",
    )

    array = ac.acs.ImageACS.from_fits(
        file_path=file_path,
        quadrant_letter="A",
        bias_subtract_via_prescan=True,
        use_calibrated_gain=False,
    )

    assert array.native[0, 0] == pytest.approx(10.0, (30.0 * 2.0) + 10.0 - 10.0, 1.0e-4)
    assert array.header.bias_serial_prescan_column[0][0] == pytest.approx(10.0, 1.0e-4)

    array = ac.acs.ImageACS.from_fits(
        file_path=file_path,
        quadrant_letter="B",
        bias_subtract_via_prescan=True,
        use_calibrated_gain=False,
    )

    assert array.native[0, 0] == pytest.approx(10.0, (40.0 * 2.0) + 10.0 - 10.0, 1.0e-4)
    assert array.header.bias_serial_prescan_column[0][0] == pytest.approx(10.0, 1.0e-4)

    array = ac.acs.ImageACS.from_fits(
        file_path=file_path,
        quadrant_letter="C",
        bias_subtract_via_prescan=True,
        use_calibrated_gain=False,
    )

    assert array.native[0, 0] == pytest.approx(10.0, (10.0 * 2.0) + 10.0 - 10.0, 1.0e-4)
    assert array.header.bias_serial_prescan_column[0][0] == pytest.approx(10.0, 1.0e-4)

    array = ac.acs.ImageACS.from_fits(
        file_path=file_path,
        quadrant_letter="D",
        bias_subtract_via_prescan=True,
        use_calibrated_gain=False,
    )

    assert array.native[0, 0] == pytest.approx(10.0, (20.0 * 2.0) + 10.0 - 10.0, 1.0e-4)
    assert array.header.bias_serial_prescan_column[0][0] == pytest.approx(10.0, 1.0e-4)


def test__from_fits__in_counts__uses_bias_file_subtraction_correctly(acs_ccd):
    fits_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "acs"
    )

    acs_ccd_0 = copy.copy(acs_ccd)
    acs_ccd_0[0, 0] = 10.0
    acs_ccd_0[0, -1] = 20.0

    acs_ccd_1 = copy.copy(acs_ccd)
    acs_ccd_1[-1, 0] = 30.0
    acs_ccd_1[-1, -1] = 40.0

    create_acs_fits(
        fits_path=fits_path,
        acs_ccd=acs_ccd,
        acs_ccd_0=acs_ccd_0,
        acs_ccd_1=acs_ccd_1,
        units="COUNTS",
        bias_file_path=path.join(fits_path, "acs_bias_ccd.fits"),
    )

    create_acs_bias_fits(
        fits_path=fits_path,
        bias_ccd=np.zeros((2068, 4144)),
        bias_ccd_0=np.ones((2068, 4144)),
        bias_ccd_1=2.0 * np.ones((2068, 4144)),
    )

    file_path = path.join(fits_path, "acs_ccd.fits")

    array = ac.acs.ImageACS.from_fits(
        file_path=file_path,
        quadrant_letter="A",
        bias_subtract_via_bias_file=True,
        use_calibrated_gain=False,
    )

    assert array.native[0, 0] == pytest.approx(10.0, (30.0 * 2.0) + 10.0 - 2.0, 1.0e-4)
    assert array.header.bias[0] == pytest.approx(2.0, 1.0e-1)

    array = ac.acs.ImageACS.from_fits(
        file_path=file_path,
        quadrant_letter="B",
        bias_subtract_via_bias_file=True,
        use_calibrated_gain=False,
    )

    assert array.native[0, 0] == pytest.approx(10.0, (40.0 * 2.0) + 10.0 - 2.0, 1.0e-4)
    assert array.header.bias[0] == pytest.approx(2.0, 1.0e-1)

    array = ac.acs.ImageACS.from_fits(
        file_path=file_path,
        quadrant_letter="C",
        bias_subtract_via_bias_file=True,
        use_calibrated_gain=False,
    )

    assert array.native[0, 0] == pytest.approx(10.0, (10.0 * 2.0) + 10.0 - 1.0, 1.0e-4)
    assert array.header.bias[0] == pytest.approx(1.0, 1.0e-1)

    array = ac.acs.ImageACS.from_fits(
        file_path=file_path,
        quadrant_letter="D",
        bias_subtract_via_bias_file=True,
        use_calibrated_gain=False,
        bias_file_path=path.join(fits_path, "acs_bias_ccd.fits"),
    )

    assert array.native[0, 0] == pytest.approx(10.0, (20.0 * 2.0) + 10.0 - 1.0, 1.0e-4)
    assert array.header.bias[0] == pytest.approx(1.0, 1.0e-1)


def test__output_quadrants_to_fits(acs_ccd):
    fits_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "acs"
    )

    file_path = path.join(fits_path, "acs_ccd.fits")

    acs_ccd_0 = copy.copy(acs_ccd)
    acs_ccd_0[0, 0] = 10.0
    acs_ccd_0[0, -1] = 20.0

    acs_ccd_1 = copy.copy(acs_ccd)
    acs_ccd_1[-1, 0] = 30.0
    acs_ccd_1[-1, -1] = 40.0

    create_acs_fits(
        fits_path=fits_path,
        acs_ccd=acs_ccd,
        acs_ccd_0=acs_ccd_0,
        acs_ccd_1=acs_ccd_1,
        units="COUNTS",
    )

    quadrant_a = ac.acs.ImageACS.from_fits(file_path=file_path, quadrant_letter="A")
    quadrant_b = ac.acs.ImageACS.from_fits(file_path=file_path, quadrant_letter="B")
    quadrant_c = ac.acs.ImageACS.from_fits(file_path=file_path, quadrant_letter="C")
    quadrant_d = ac.acs.ImageACS.from_fits(file_path=file_path, quadrant_letter="D")

    file_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "output.fits"
    )

    ac.acs.acs_util.output_quadrants_to_fits(
        quadrant_a=quadrant_a,
        quadrant_b=quadrant_b,
        quadrant_c=quadrant_c,
        quadrant_d=quadrant_d,
        file_path=file_path,
        overwrite=True,
    )

    acs_ccd_output = ac.util.array_2d.numpy_array_2d_via_fits_from(
        file_path=file_path, hdu=1, do_not_scale_image_data=True
    )

    assert acs_ccd_output[0, 0] == 10.0
    assert acs_ccd_output[0, -1] == 20.0

    acs_ccd_output = ac.util.array_2d.numpy_array_2d_via_fits_from(
        file_path=file_path, hdu=4, do_not_scale_image_data=True
    )

    assert acs_ccd_output[-1, 0] == 30.0
    assert acs_ccd_output[-1, -1] == 40.0
