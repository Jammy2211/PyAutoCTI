import os
from os import path
import shutil

import numpy as np
import pytest
import autocti as ac

test_data_path = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "arrays"
)


def test__norm_columns_list():
    data = ac.Array2D.full(fill_value=1.0, shape_native=(5, 5), pixel_scales=(1.0, 1.0))
    noise_map = ac.Array2D.ones(
        shape_native=data.shape_native, pixel_scales=data.pixel_scales
    )

    layout = ac.Layout2DCI(shape_2d=data.shape_native, region_list=[(1, 4, 1, 4)])

    dataset = ac.ImagingCI(
        data=data, noise_map=noise_map, pre_cti_data=data, layout=layout
    )

    assert dataset.norm_columns_list == [1.0, 1.0, 1.0]

    data = ac.Array2D.no_mask(
        values=np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 4.0, 7.0, 0.0],
                [0.0, 2.0, 5.0, 8.0, 0.0],
                [0.0, 3.0, 6.0, 9.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        pixel_scales=(1.0, 1.0),
    )

    layout = ac.Layout2DCI(shape_2d=data.shape_native, region_list=[(1, 4, 1, 4)])

    dataset = ac.ImagingCI(
        data=data, noise_map=noise_map, pre_cti_data=data, layout=layout
    )

    assert dataset.norm_columns_list == [2.0, 5.0, 8.0]

    layout = ac.Layout2DCI(shape_2d=data.shape_native, region_list=[(2, 4, 1, 4)])

    dataset = ac.ImagingCI(
        data=data, noise_map=noise_map, pre_cti_data=data, layout=layout
    )

    assert dataset.norm_columns_list == [2.5, 5.5, 8.5]

    mask = ac.Mask2D(
        mask=np.array(
            [
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, False, True, False, True],
                [True, True, False, False, True],
                [True, True, True, True, True],
            ]
        ),
        pixel_scales=data.pixel_scales,
    )

    data = data.apply_mask(mask=mask)

    dataset = ac.ImagingCI(
        data=data, noise_map=noise_map, pre_cti_data=data, layout=layout
    )

    assert dataset.norm_columns_list == [2.0, 6.0, 8.5]


def test__pre_cti_data_residual_map():
    data = ac.Array2D.full(fill_value=1.0, shape_native=(5, 5), pixel_scales=(1.0, 1.0))
    noise_map = ac.Array2D.ones(
        shape_native=data.shape_native, pixel_scales=data.pixel_scales
    )

    pre_cti_data = ac.Array2D.full(
        fill_value=0.75, shape_native=(5, 5), pixel_scales=(1.0, 1.0)
    )

    layout = ac.Layout2DCI(shape_2d=data.shape_native, region_list=[(1, 4, 1, 4)])

    dataset = ac.ImagingCI(
        data=data, noise_map=noise_map, pre_cti_data=pre_cti_data, layout=layout
    )

    pre_cti_data_residual_map = ac.Array2D.full(
        fill_value=0.25, shape_native=(5, 5), pixel_scales=(1.0, 1.0)
    )

    assert dataset.pre_cti_data_residual_map.native == pytest.approx(
        pre_cti_data_residual_map.native, 1.0e-4
    )


def test__from_fits__load_all_data_components__has_correct_attributes(layout_ci_7x7):
    dataset = ac.ImagingCI.from_fits(
        pixel_scales=1.0,
        layout=layout_ci_7x7,
        data_path=path.join(test_data_path, "3x3_ones.fits"),
        data_hdu=0,
        noise_map_path=path.join(test_data_path, "3x3_twos.fits"),
        noise_map_hdu=0,
        pre_cti_data_path=path.join(test_data_path, "3x3_threes.fits"),
        pre_cti_data_hdu=0,
        cosmic_ray_map_path=path.join(test_data_path, "3x3_fours.fits"),
        cosmic_ray_map_hdu=0,
    )

    assert (dataset.data.native == np.ones((3, 3))).all()
    assert (dataset.noise_map.native == 2.0 * np.ones((3, 3))).all()
    assert (dataset.pre_cti_data.native == 3.0 * np.ones((3, 3))).all()
    assert (dataset.cosmic_ray_map.native == 4.0 * np.ones((3, 3))).all()

    assert dataset.layout == layout_ci_7x7


def test__from_fits__load_all_image_components__load_from_multi_hdu_fits(layout_ci_7x7):
    dataset = ac.ImagingCI.from_fits(
        pixel_scales=1.0,
        layout=layout_ci_7x7,
        data_path=path.join(test_data_path, "3x3_multiple_hdu.fits"),
        data_hdu=0,
        noise_map_path=path.join(test_data_path, "3x3_multiple_hdu.fits"),
        noise_map_hdu=1,
        pre_cti_data_path=path.join(test_data_path, "3x3_multiple_hdu.fits"),
        pre_cti_data_hdu=2,
        cosmic_ray_map_path=path.join(test_data_path, "3x3_multiple_hdu.fits"),
        cosmic_ray_map_hdu=3,
    )

    assert (dataset.data.native == np.ones((3, 3))).all()
    assert (dataset.noise_map.native == 2.0 * np.ones((3, 3))).all()
    assert (dataset.pre_cti_data.native == 3.0 * np.ones((3, 3))).all()
    assert (dataset.cosmic_ray_map.native == 4.0 * np.ones((3, 3))).all()

    assert dataset.layout == layout_ci_7x7


def test__from_fits__noise_map_from_single_value(layout_ci_7x7):
    dataset = ac.ImagingCI.from_fits(
        pixel_scales=1.0,
        layout=layout_ci_7x7,
        data_path=path.join(test_data_path, "3x3_ones.fits"),
        data_hdu=0,
        noise_map_from_single_value=10.0,
        pre_cti_data_path=path.join(test_data_path, "3x3_threes.fits"),
        pre_cti_data_hdu=0,
    )

    assert (dataset.data.native == np.ones((3, 3))).all()
    assert (dataset.noise_map.native == 10.0 * np.ones((3, 3))).all()
    assert (dataset.pre_cti_data.native == 3.0 * np.ones((3, 3))).all()
    assert dataset.cosmic_ray_map == None

    assert dataset.layout == layout_ci_7x7


def test__output_to_fits___all_arrays(layout_ci_7x7):
    dataset = ac.ImagingCI.from_fits(
        pixel_scales=1.0,
        layout=layout_ci_7x7,
        data_path=path.join(test_data_path, "3x3_ones.fits"),
        data_hdu=0,
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

    dataset.output_to_fits(
        data_path=path.join(output_data_dir, "data.fits"),
        noise_map_path=path.join(output_data_dir, "noise_map.fits"),
        pre_cti_data_path=path.join(output_data_dir, "pre_cti_data.fits"),
        cosmic_ray_map_path=path.join(output_data_dir, "cosmic_ray_map.fits"),

    )

    dataset = ac.ImagingCI.from_fits(
        pixel_scales=1.0,
        layout=layout_ci_7x7,
        data_path=path.join(output_data_dir, "data.fits"),
        data_hdu=0,
        noise_map_path=path.join(output_data_dir, "noise_map.fits"),
        noise_map_hdu=0,
        pre_cti_data_path=path.join(output_data_dir, "pre_cti_data.fits"),
        pre_cti_data_hdu=0,
        cosmic_ray_map_path=path.join(output_data_dir, "cosmic_ray_map.fits"),
        cosmic_ray_map_hdu=0,
    )

    assert (dataset.data.native == np.ones((3, 3))).all()
    assert (dataset.noise_map.native == 2.0 * np.ones((3, 3))).all()
    assert (dataset.pre_cti_data.native == 3.0 * np.ones((3, 3))).all()
    assert (dataset.cosmic_ray_map.native == 4.0 * np.ones((3, 3))).all()


def test__apply_mask__masks_arrays_correctly(imaging_ci_7x7):
    mask = ac.Mask2D.all_false(
        shape_native=imaging_ci_7x7.shape_native, pixel_scales=1.0
    )

    mask[0, 0] = True

    masked_dataset = imaging_ci_7x7.apply_mask(mask=mask)

    assert (masked_dataset.mask == mask).all()

    masked_image = imaging_ci_7x7.data
    masked_image[0, 0] = 0.0

    assert (masked_dataset.data == masked_image).all()

    masked_noise_map = imaging_ci_7x7.noise_map
    masked_noise_map[0, 0] = 0.0

    assert (masked_dataset.noise_map == masked_noise_map).all()

    assert (masked_dataset.pre_cti_data == imaging_ci_7x7.pre_cti_data).all()

    masked_cosmic_ray_map = imaging_ci_7x7.cosmic_ray_map
    masked_cosmic_ray_map[0, 0] = 0.0

    assert (masked_dataset.cosmic_ray_map == masked_cosmic_ray_map).all()


def test__apply_settings__include_parallel_columns_extraction(
    imaging_ci_7x7, mask_2d_7x7_unmasked, ci_noise_scaling_map_dict_7x7
):
    mask = ac.Mask2D.all_false(
        shape_native=imaging_ci_7x7.shape_native, pixel_scales=1.0
    )
    mask[0, 2] = True

    masked_dataset = imaging_ci_7x7.apply_mask(mask=mask)
    masked_dataset = masked_dataset.apply_settings(
        settings=ac.SettingsImagingCI(parallel_pixels=(1, 3))
    )

    mask = ac.Mask2D.all_false(shape_native=(7, 2), pixel_scales=1.0)
    mask[0, 0] = True

    assert (masked_dataset.mask == mask).all()

    data = np.ones((7, 2))
    data[0, 0] = 0.0

    assert masked_dataset.data == pytest.approx(data, 1.0e-4)

    noise_map = 2.0 * np.ones((7, 2))
    noise_map[0, 0] = 0.0

    assert masked_dataset.noise_map == pytest.approx(noise_map, 1.0e-4)

    pre_cti_data = 10.0 * np.ones((7, 2))

    assert masked_dataset.pre_cti_data == pytest.approx(pre_cti_data, 1.0e-4)

    assert masked_dataset.cosmic_ray_map.shape == (7, 2)

    noise_scaling_map_0 = np.ones((7, 2))
    noise_scaling_map_0[0, 0] = 0.0

    assert masked_dataset.noise_scaling_map_dict["parallel_eper"] == pytest.approx(
        noise_scaling_map_0, 1.0e-4
    )

    noise_scaling_map_1 = 2.0 * np.ones((7, 2))
    noise_scaling_map_1[0, 0] = 0.0

    assert masked_dataset.noise_scaling_map_dict["serial_eper"] == pytest.approx(
        noise_scaling_map_1, 1.0e-4
    )


def test__apply_settings__serial_masked_imaging_ci(
    imaging_ci_7x7, mask_2d_7x7_unmasked, ci_noise_scaling_map_dict_7x7
):
    mask = ac.Mask2D.all_false(
        shape_native=imaging_ci_7x7.shape_native, pixel_scales=1.0
    )
    mask[1, 0] = True

    masked_dataset = imaging_ci_7x7.apply_mask(mask=mask)
    masked_dataset = masked_dataset.apply_settings(
        settings=ac.SettingsImagingCI(serial_pixels=(0, 1))
    )

    mask = ac.Mask2D.all_false(shape_native=(1, 7), pixel_scales=1.0)
    mask[0, 0] = True

    assert (masked_dataset.mask == mask).all()

    data = np.ones((1, 7))
    data[0, 0] = 0.0

    assert masked_dataset.data == pytest.approx(data, 1.0e-4)

    noise_map = 2.0 * np.ones((1, 7))
    noise_map[0, 0] = 0.0

    assert masked_dataset.noise_map == pytest.approx(noise_map, 1.0e-4)

    pre_cti_data = 10.0 * np.ones((1, 7))

    assert masked_dataset.pre_cti_data == pytest.approx(pre_cti_data, 1.0e-4)

    assert masked_dataset.cosmic_ray_map.shape == (1, 7)

    noise_scaling_map_0 = np.ones((1, 7))
    noise_scaling_map_0[0, 0] = 0.0

    assert masked_dataset.noise_scaling_map_dict["parallel_eper"] == pytest.approx(
        noise_scaling_map_0, 1.0e-4
    )

    noise_scaling_map_1 = 2.0 * np.ones((1, 7))
    noise_scaling_map_1[0, 0] = 0.0

    assert masked_dataset.noise_scaling_map_dict["serial_eper"] == pytest.approx(
        noise_scaling_map_1, 1.0e-4
    )


def test__fpr_value():
    data = ac.Array2D.full(fill_value=1.0, shape_native=(5, 5), pixel_scales=(1.0, 1.0))
    noise_map = ac.Array2D.ones(
        shape_native=data.shape_native, pixel_scales=data.pixel_scales
    )

    layout = ac.Layout2DCI(shape_2d=data.shape_native, region_list=[(1, 4, 1, 4)])

    dataset = ac.ImagingCI(
        data=data, noise_map=noise_map, pre_cti_data=data, layout=layout
    )

    assert dataset.fpr_value == pytest.approx(1.0, 1.0e-4)


def test__set_noise_scaling_map_dict(imaging_ci_7x7, ci_noise_scaling_map_dict_7x7):
    imaging_ci_7x7.noise_scaling_map_dict = None

    imaging_ci_7x7.set_noise_scaling_map_dict(
        noise_scaling_map_dict=ci_noise_scaling_map_dict_7x7
    )

    assert (
        imaging_ci_7x7.noise_scaling_map_dict["parallel_eper"]
        == ci_noise_scaling_map_dict_7x7["parallel_eper"].native
    ).all()
