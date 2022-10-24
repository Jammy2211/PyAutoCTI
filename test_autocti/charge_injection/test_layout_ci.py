import numpy as np
from astropy.io import fits
import shutil
import os
from os import path
import pytest

import autocti as ac


def test__pre_cti_data_uniform_from():

    layout = ac.Layout2DCI(shape_2d=(4, 3), region_list=[(0, 1, 0, 2), (2, 3, 0, 2)])

    pre_cti_data = layout.pre_cti_data_uniform_from(norm=30.0, pixel_scales=1.0)

    assert (
        pre_cti_data.native
        == np.array(
            [[30.0, 30.0, 0.0], [0.0, 0.0, 0.0], [30.0, 30.0, 0.0], [0.0, 0.0, 0.0]]
        )
    ).all()


def test__pre_cti_data_non_uniform_from():

    layout = ac.Layout2DCI(shape_2d=(5, 5), region_list=[(0, 3, 0, 3)])

    image = layout.pre_cti_data_non_uniform_from(
        injection_norm_list=[100.0, 90.0, 80.0], pixel_scales=1.0, row_slope=0.0
    )

    assert (
        image.native
        == np.array(
            [
                [100.0, 90.0, 80.0, 0.0, 0.0],
                [100.0, 90.0, 80.0, 0.0, 0.0],
                [100.0, 90.0, 80.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    ).all()

    layout = ac.Layout2DCI(shape_2d=(5, 5), region_list=[(0, 2, 1, 3), (3, 5, 1, 3)])

    image = layout.pre_cti_data_non_uniform_from(
        injection_norm_list=[10.0, 20.0], pixel_scales=1.0, row_slope=0.0
    )

    assert (
        image.native
        == np.array(
            [
                [0.0, 10.0, 20.0, 0.0, 0.0],
                [0.0, 10.0, 20.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 10.0, 20.0, 0.0, 0.0],
                [0.0, 10.0, 20.0, 0.0, 0.0],
            ]
        )
    ).all()

    image = layout.pre_cti_data_non_uniform_from(
        injection_norm_list=[10.0, 20.0], pixel_scales=1.0, row_slope=0.01
    )

    assert image.native == pytest.approx(
        np.array(
            [
                [0.0, 10.0, 20.0, 0.0, 0.0],
                [0.0, 10.0695, 20.13911, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 10.0, 20.0, 0.0, 0.0],
                [0.0, 10.0695, 20.13911, 0.0, 0.0],
            ]
        ),
        1.0e-2,
    )


def test__pre_cti_data_from__compare_uniform_to_non_uniform():

    layout = ac.Layout2DCI(shape_2d=(5, 5), region_list=[(2, 4, 0, 2)])

    pre_cti_data_0 = layout.pre_cti_data_uniform_from(norm=30.0, pixel_scales=1.0)

    pre_cti_data_1 = layout.pre_cti_data_non_uniform_from(
        injection_norm_list=[30.0, 30.0], pixel_scales=1.0, row_slope=0.0
    )

    assert (pre_cti_data_0 == pre_cti_data_1).all()


def test__pre_cti_data_non_uniform_via_lists_from():

    layout = ac.Layout2DCI(shape_2d=(10, 3), region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

    pre_cti_data = layout.pre_cti_data_non_uniform_via_lists_from(
        injection_norm_lists=[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
        pixel_scales=1.0,
        row_slope=0.0,
    )

    assert (
        pre_cti_data.native
        == np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0],
                [0.0, 0.0, 0.0],
                [5.0, 6.0, 7.0],
                [5.0, 6.0, 7.0],
                [5.0, 6.0, 7.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
    ).all()

    pre_cti_data = layout.pre_cti_data_non_uniform_via_lists_from(
        injection_norm_lists=[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
        pixel_scales=1.0,
        row_slope=0.01,
    )

    assert pre_cti_data.native == pytest.approx(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 3.0, 4.0],
                [2.014, 3.021, 4.028],
                [2.022, 3.033, 4.044],
                [0.0, 0.0, 0.0],
                [5.0, 6.0, 7.0],
                [5.035, 6.042, 7.049],
                [5.55, 6.066, 7.077],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        1.0e-1,
    )


def create_tvac_fits(
    fits_path,
    filename,
    tvac_arr,
):

    if path.exists(fits_path):
        shutil.rmtree(fits_path)

    os.makedirs(fits_path)

    hdu_list = fits.HDUList()

    hdu_list.append(fits.ImageHDU())
    hdu_list.append(fits.ImageHDU(tvac_arr))

    hdu_list[1].header.set(
        "CCDID", "1-1", "e.g. Detector ID, e.g. '0-0', '1-1' ... '6-6'"
    )
    hdu_list[1].header.set(
        "QUADID", "E", "e.g. Quadrant ID, e.g. 'E', 'F', 'G' or 'H' "
    )
    hdu_list[1].header.set("OVRSCANX", 29, "")
    hdu_list[1].header.set("PRESCANX", 51, "")
    hdu_list[1].header.set("NAXIS1", 2128, "")
    hdu_list[1].header.set("NAXIS2", 2086, "")
    hdu_list[1].header.set(
        "CI_IJON", 420, "If charge injection on: number of lines injecte"
    )
    hdu_list[1].header.set(
        "CI_IJOFF", 100, "If charge injection on: number of lines without"
    )
    hdu_list[1].header.set(
        "CI_VSTAR", 16, "If charge injection on: image line where the pa"
    )
    hdu_list[1].header.set(
        "CI_VEND", 2066, "If charge injection on: image line where the pa"
    )

    hdu_list.writeto(path.join(fits_path, filename))


def test__tvac_example():

    fits_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "acs"
    )

    filename = "tvac.fits"

    file_path = path.join(fits_path, filename)

    tvac_arr = np.zeros(shape=(2086, 2128))

    tvac_arr[16:436, 51:2099] = 10000.0
    tvac_arr[536:956, 51:2099] = 10000.0
    tvac_arr[1056:1476, 51:2099] = 10000.0
    tvac_arr[1576:1996, 51:2099] = 10000.0

    create_tvac_fits(fits_path=fits_path, filename=filename, tvac_arr=tvac_arr)

    data_hdulist = fits.open(file_path)

    sci_header = data_hdulist[0].header
    data_header = data_hdulist[1].header

    data = data_hdulist[1].data

    image_ci = ac.euclid.Array2DEuclid.from_fits_header(
        array=data.astype("float"), ext_header=data_header
    )

    layout_2d = ac.Layout2DCI.from_euclid_fits_header(
        ext_header=data_header,
    )

    assert layout_2d.region_list[0] == (16, 436, 51, 2099)
    assert layout_2d.region_list[1] == (536, 956, 51, 2099)
    assert layout_2d.region_list[2] == (1056, 1476, 51, 2099)
    assert layout_2d.region_list[3] == (1576, 1996, 51, 2099)

    for region in layout_2d.region_list:

        tvac_region = ac.Region2D(region=region)

        assert (image_ci.native[tvac_region.slice] == 10000).all()
        assert (
            image_ci.native[tvac_region.y0 : tvac_region.y1, tvac_region.x0 - 1] == 0
        ).all()
        assert (
            image_ci.native[tvac_region.y0 : tvac_region.y1, tvac_region.x1] == 0
        ).all()
        assert (
            image_ci.native[tvac_region.y0 - 1, tvac_region.x0 : tvac_region.x1] == 0
        ).all()
        assert (
            image_ci.native[tvac_region.y1, tvac_region.x0 : tvac_region.x1] == 0
        ).all()
