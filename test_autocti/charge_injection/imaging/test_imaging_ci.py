import os
from os import path
import shutil

import numpy as np
import pytest
import autocti as ac

test_data_path = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "arrays"
)


def test__normalization_columns_list():

    image = ac.Array2D.full(
        fill_value=1.0, shape_native=(7, 7), pixel_scales=(1.0, 1.0)
    )

    layout = ac.ci.Layout2DCI(shape_2d=image.shape_native, region_list=[(1, 5, 1, 5)])

    imaging = ac.ci.ImagingCI(
        image=image, noise_map=image, pre_cti_data=image, layout=layout
    )

    assert imaging.normalization_columns_list == [1.0, 1.0, 1.0, 1.0]


def test__from_fits__load_all_data_components__has_correct_attributes(layout_ci_7x7):

    imaging = ac.ci.ImagingCI.from_fits(
        pixel_scales=1.0,
        layout=layout_ci_7x7,
        image_path=path.join(test_data_path, "3x3_ones.fits"),
        image_hdu=0,
        noise_map_path=path.join(test_data_path, "3x3_twos.fits"),
        noise_map_hdu=0,
        pre_cti_data_path=path.join(test_data_path, "3x3_threes.fits"),
        pre_cti_data_hdu=0,
        cosmic_ray_map_path=path.join(test_data_path, "3x3_fours.fits"),
        cosmic_ray_map_hdu=0,
    )

    assert (imaging.image.native == np.ones((3, 3))).all()
    assert (imaging.noise_map.native == 2.0 * np.ones((3, 3))).all()
    assert (imaging.pre_cti_data.native == 3.0 * np.ones((3, 3))).all()
    assert (imaging.cosmic_ray_map.native == 4.0 * np.ones((3, 3))).all()

    assert imaging.layout == layout_ci_7x7


def test__from_fits__load_all_image_components__load_from_multi_hdu_fits(layout_ci_7x7):

    imaging = ac.ci.ImagingCI.from_fits(
        pixel_scales=1.0,
        layout=layout_ci_7x7,
        image_path=path.join(test_data_path, "3x3_multiple_hdu.fits"),
        image_hdu=0,
        noise_map_path=path.join(test_data_path, "3x3_multiple_hdu.fits"),
        noise_map_hdu=1,
        pre_cti_data_path=path.join(test_data_path, "3x3_multiple_hdu.fits"),
        pre_cti_data_hdu=2,
        cosmic_ray_map_path=path.join(test_data_path, "3x3_multiple_hdu.fits"),
        cosmic_ray_map_hdu=3,
    )

    assert (imaging.image.native == np.ones((3, 3))).all()
    assert (imaging.noise_map.native == 2.0 * np.ones((3, 3))).all()
    assert (imaging.pre_cti_data.native == 3.0 * np.ones((3, 3))).all()
    assert (imaging.cosmic_ray_map.native == 4.0 * np.ones((3, 3))).all()

    assert imaging.layout == layout_ci_7x7


def test__from_fits__noise_map_from_single_value(layout_ci_7x7):

    imaging = ac.ci.ImagingCI.from_fits(
        pixel_scales=1.0,
        layout=layout_ci_7x7,
        image_path=path.join(test_data_path, "3x3_ones.fits"),
        image_hdu=0,
        noise_map_from_single_value=10.0,
        pre_cti_data_path=path.join(test_data_path, "3x3_threes.fits"),
        pre_cti_data_hdu=0,
    )

    assert (imaging.image.native == np.ones((3, 3))).all()
    assert (imaging.noise_map.native == 10.0 * np.ones((3, 3))).all()
    assert (imaging.pre_cti_data.native == 3.0 * np.ones((3, 3))).all()
    assert imaging.cosmic_ray_map == None

    assert imaging.layout == layout_ci_7x7


def test__output_to_fits___all_arrays(layout_ci_7x7):

    imaging = ac.ci.ImagingCI.from_fits(
        pixel_scales=1.0,
        layout=layout_ci_7x7,
        image_path=path.join(test_data_path, "3x3_ones.fits"),
        image_hdu=0,
        noise_map_path=path.join(test_data_path, "3x3_twos.fits"),
        noise_map_hdu=0,
        pre_cti_data_path=path.join(test_data_path, "3x3_threes.fits"),
        pre_cti_data_hdu=0,
        cosmic_ray_map_path=path.join(test_data_path, "3x3_fours.fits"),
        cosmic_ray_map_hdu=0,
    )

    output_data_dir = path.join(test_data_path, "output_test")

    if path.exists(output_data_dir):
        shutil.rmtree(output_data_dir)

    os.makedirs(output_data_dir)

    imaging.output_to_fits(
        image_path=path.join(output_data_dir, "image.fits"),
        noise_map_path=path.join(output_data_dir, "noise_map.fits"),
        pre_cti_data_path=path.join(output_data_dir, "pre_cti_data.fits"),
        cosmic_ray_map_path=path.join(output_data_dir, "cosmic_ray_map.fits"),
    )

    imaging = ac.ci.ImagingCI.from_fits(
        pixel_scales=1.0,
        layout=layout_ci_7x7,
        image_path=path.join(output_data_dir, "image.fits"),
        image_hdu=0,
        noise_map_path=path.join(output_data_dir, "noise_map.fits"),
        noise_map_hdu=0,
        pre_cti_data_path=path.join(output_data_dir, "pre_cti_data.fits"),
        pre_cti_data_hdu=0,
        cosmic_ray_map_path=path.join(output_data_dir, "cosmic_ray_map.fits"),
        cosmic_ray_map_hdu=0,
    )

    assert (imaging.image.native == np.ones((3, 3))).all()
    assert (imaging.noise_map.native == 2.0 * np.ones((3, 3))).all()
    assert (imaging.pre_cti_data.native == 3.0 * np.ones((3, 3))).all()
    assert (imaging.cosmic_ray_map.native == 4.0 * np.ones((3, 3))).all()


def test__apply_mask__masks_arrays_correctly(imaging_ci_7x7):

    mask = ac.Mask2D.unmasked(
        shape_native=imaging_ci_7x7.shape_native, pixel_scales=1.0
    )

    mask[0, 0] = True

    masked_imaging_ci = imaging_ci_7x7.apply_mask(mask=mask)

    assert (masked_imaging_ci.mask == mask).all()

    masked_image = imaging_ci_7x7.image
    masked_image[0, 0] = 0.0

    assert (masked_imaging_ci.image == masked_image).all()

    masked_noise_map = imaging_ci_7x7.noise_map
    masked_noise_map[0, 0] = 0.0

    assert (masked_imaging_ci.noise_map == masked_noise_map).all()

    assert (masked_imaging_ci.pre_cti_data == imaging_ci_7x7.pre_cti_data).all()

    masked_cosmic_ray_map = imaging_ci_7x7.cosmic_ray_map
    masked_cosmic_ray_map[0, 0] = 0.0

    assert (masked_imaging_ci.cosmic_ray_map == masked_cosmic_ray_map).all()


def test__apply_settings__include_parallel_columns_extraction(
    imaging_ci_7x7, mask_2d_7x7_unmasked, ci_noise_scaling_map_list_7x7
):

    mask = ac.Mask2D.unmasked(
        shape_native=imaging_ci_7x7.shape_native, pixel_scales=1.0
    )
    mask[0, 2] = True

    masked_imaging_ci = imaging_ci_7x7.apply_mask(mask=mask)
    masked_imaging_ci = masked_imaging_ci.apply_settings(
        settings=ac.ci.SettingsImagingCI(parallel_pixels=(1, 3))
    )

    mask = ac.Mask2D.unmasked(shape_native=(7, 2), pixel_scales=1.0)
    mask[0, 0] = True

    assert (masked_imaging_ci.mask == mask).all()

    image = np.ones((7, 2))
    image[0, 0] = 0.0

    assert masked_imaging_ci.image == pytest.approx(image, 1.0e-4)

    noise_map = 2.0 * np.ones((7, 2))
    noise_map[0, 0] = 0.0

    assert masked_imaging_ci.noise_map == pytest.approx(noise_map, 1.0e-4)

    pre_cti_data = 10.0 * np.ones((7, 2))

    assert masked_imaging_ci.pre_cti_data == pytest.approx(pre_cti_data, 1.0e-4)

    assert masked_imaging_ci.cosmic_ray_map.shape == (7, 2)

    noise_scaling_map_0 = np.ones((7, 2))
    noise_scaling_map_0[0, 0] = 0.0

    assert masked_imaging_ci.noise_scaling_map_list[0] == pytest.approx(
        noise_scaling_map_0, 1.0e-4
    )

    noise_scaling_map_1 = 2.0 * np.ones((7, 2))
    noise_scaling_map_1[0, 0] = 0.0

    assert masked_imaging_ci.noise_scaling_map_list[1] == pytest.approx(
        noise_scaling_map_1, 1.0e-4
    )


def test__apply_settings__serial_masked_imaging_ci(
    imaging_ci_7x7, mask_2d_7x7_unmasked, ci_noise_scaling_map_list_7x7
):

    mask = ac.Mask2D.unmasked(
        shape_native=imaging_ci_7x7.shape_native, pixel_scales=1.0
    )
    mask[1, 0] = True

    masked_imaging_ci = imaging_ci_7x7.apply_mask(mask=mask)
    masked_imaging_ci = masked_imaging_ci.apply_settings(
        settings=ac.ci.SettingsImagingCI(serial_pixels=(0, 1))
    )

    mask = ac.Mask2D.unmasked(shape_native=(1, 7), pixel_scales=1.0)
    mask[0, 0] = True

    assert (masked_imaging_ci.mask == mask).all()

    image = np.ones((1, 7))
    image[0, 0] = 0.0

    assert masked_imaging_ci.image == pytest.approx(image, 1.0e-4)

    noise_map = 2.0 * np.ones((1, 7))
    noise_map[0, 0] = 0.0

    assert masked_imaging_ci.noise_map == pytest.approx(noise_map, 1.0e-4)

    pre_cti_data = 10.0 * np.ones((1, 7))

    assert masked_imaging_ci.pre_cti_data == pytest.approx(pre_cti_data, 1.0e-4)

    assert masked_imaging_ci.cosmic_ray_map.shape == (1, 7)

    noise_scaling_map_0 = np.ones((1, 7))
    noise_scaling_map_0[0, 0] = 0.0

    assert masked_imaging_ci.noise_scaling_map_list[0] == pytest.approx(
        noise_scaling_map_0, 1.0e-4
    )

    noise_scaling_map_1 = 2.0 * np.ones((1, 7))
    noise_scaling_map_1[0, 0] = 0.0

    assert masked_imaging_ci.noise_scaling_map_list[1] == pytest.approx(
        noise_scaling_map_1, 1.0e-4
    )
