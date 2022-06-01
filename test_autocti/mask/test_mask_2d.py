import os
from os import path
import shutil

import numpy as np
import pytest
import autocti as ac
from autocti import exc

test_data_path = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "array"
)


def test__manual():

    mask = ac.Mask2D.manual(mask=[[False, False], [True, True]], pixel_scales=1.0)

    assert type(mask) == ac.Mask2D
    assert (mask == np.array([[False, False], [True, True]])).all()
    assert mask.pixel_scales == (1.0, 1.0)
    assert mask.origin == (0.0, 0.0)

    mask = ac.Mask2D.manual(
        mask=[[False, False, True], [True, True, False]],
        pixel_scales=(2.0, 3.0),
        origin=(0.0, 1.0),
    )

    assert type(mask) == ac.Mask2D
    assert (mask == np.array([[False, False, True], [True, True, False]])).all()
    assert mask.pixel_scales == (2.0, 3.0)
    assert mask.origin == (0.0, 1.0)

    mask = ac.Mask2D.manual(
        mask=[[False, False, True], [True, True, False]], pixel_scales=1.0, invert=True
    )

    assert type(mask) == ac.Mask2D
    assert (mask == np.array([[True, True, False], [False, False, True]])).all()


def test__mask__input_is_1d_mask__no_shape_native__raises_exception():

    with pytest.raises(exc.MaskException):

        ac.Mask2D.manual(mask=[False, False, True], pixel_scales=1.0)

    with pytest.raises(exc.MaskException):

        ac.Mask2D.manual(mask=[False, False, True], pixel_scales=False)

    with pytest.raises(exc.MaskException):

        ac.Mask2D.manual(mask=[False, False, True], pixel_scales=1.0)

    with pytest.raises(exc.MaskException):

        ac.Mask2D.manual(mask=[False, False, True], pixel_scales=False)


def test__unmasked():

    mask = ac.Mask2D.unmasked(shape_native=(5, 5), pixel_scales=1.0, invert=False)

    assert mask.shape == (5, 5)
    assert (
        mask
        == np.array(
            [
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ]
        )
    ).all()

    mask = ac.Mask2D.unmasked(
        shape_native=(3, 3), pixel_scales=(1.5, 1.5), invert=False
    )

    assert mask.shape == (3, 3)
    assert (
        mask
        == np.array(
            [[False, False, False], [False, False, False], [False, False, False]]
        )
    ).all()

    assert mask.pixel_scales == (1.5, 1.5)
    assert mask.origin == (0.0, 0.0)

    mask = ac.Mask2D.unmasked(
        shape_native=(3, 3), pixel_scales=(2.0, 2.5), invert=True, origin=(1.0, 2.0)
    )

    assert mask.shape == (3, 3)
    assert (
        mask == np.array([[True, True, True], [True, True, True], [True, True, True]])
    ).all()

    assert mask.pixel_scales == (2.0, 2.5)
    assert mask.origin == (1.0, 2.0)


def test__from_masked_regions():

    mask = ac.Mask2D.from_masked_regions(
        shape_native=(3, 3), masked_regions=[(0, 3, 2, 3)], pixel_scales=1.0
    )

    assert (
        mask
        == np.array([[False, False, True], [False, False, True], [False, False, True]])
    ).all()

    mask = ac.Mask2D.from_masked_regions(
        shape_native=(3, 3),
        masked_regions=[(0, 3, 2, 3), (0, 2, 0, 2)],
        pixel_scales=1.0,
    )

    assert (
        mask == np.array([[True, True, True], [True, True, True], [False, False, True]])
    ).all()


def test__cosmic_ray_mask_included_in_total_mask():

    cosmic_ray_map = ac.Array2D.manual(
        array=np.array(
            [[False, False, False], [False, True, False], [False, False, False]]
        ),
        pixel_scales=1.0,
    )

    mask = ac.Mask2D.from_cosmic_ray_map_buffed(
        cosmic_ray_map=cosmic_ray_map,
        settings=ac.SettingsMask2D(
            cosmic_ray_parallel_buffer=0,
            cosmic_ray_serial_buffer=0,
            cosmic_ray_diagonal_buffer=0,
        ),
    )

    assert (
        mask
        == np.array(
            [[False, False, False], [False, True, False], [False, False, False]]
        )
    ).all()

    cosmic_ray_map = ac.Array2D.manual(
        array=[[False, True, False], [False, False, False], [False, False, False]],
        pixel_scales=1.0,
    )

    mask = ac.Mask2D.from_cosmic_ray_map_buffed(
        cosmic_ray_map=cosmic_ray_map,
        settings=ac.SettingsMask2D(
            cosmic_ray_parallel_buffer=2,
            cosmic_ray_serial_buffer=0,
            cosmic_ray_diagonal_buffer=0,
        ),
    )

    assert (
        mask
        == np.array([[False, True, False], [False, True, False], [False, True, False]])
    ).all()

    cosmic_ray_map = ac.Array2D.manual(
        array=[[False, False, False], [True, False, False], [False, False, False]],
        pixel_scales=1.0,
    )

    mask = ac.Mask2D.from_cosmic_ray_map_buffed(
        cosmic_ray_map=cosmic_ray_map,
        settings=ac.SettingsMask2D(
            cosmic_ray_parallel_buffer=0,
            cosmic_ray_serial_buffer=2,
            cosmic_ray_diagonal_buffer=0,
        ),
    )

    assert (
        mask
        == np.array([[False, False, False], [True, True, True], [False, False, False]])
    ).all()

    cosmic_ray_map = ac.Array2D.manual(
        array=[
            [False, False, False, False],
            [False, True, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ],
        pixel_scales=1.0,
    )

    mask = ac.Mask2D.from_cosmic_ray_map_buffed(
        cosmic_ray_map=cosmic_ray_map,
        settings=ac.SettingsMask2D(
            cosmic_ray_parallel_buffer=0,
            cosmic_ray_serial_buffer=0,
            cosmic_ray_diagonal_buffer=2,
        ),
    )

    assert (
        mask
        == np.array(
            [
                [False, False, False, False],
                [False, True, True, True],
                [False, True, True, True],
                [False, True, True, True],
            ]
        )
    ).all()


def test__load_and_output_mask_to_fits():

    mask = ac.Mask2D.from_fits(
        file_path=path.join(test_data_path, "3x3_ones.fits"),
        hdu=0,
        pixel_scales=(1.0, 1.0),
    )

    output_data_dir = path.join(test_data_path, "output_test")

    if path.exists(output_data_dir):
        shutil.rmtree(output_data_dir)

    os.makedirs(output_data_dir)

    mask.output_to_fits(file_path=path.join(output_data_dir, "mask.fits"))

    mask = ac.Mask2D.from_fits(
        file_path=path.join(output_data_dir, "mask.fits"),
        hdu=0,
        pixel_scales=(1.0, 1.0),
        origin=(2.0, 2.0),
    )

    assert (mask == np.ones((3, 3))).all()
    assert mask.pixel_scales == (1.0, 1.0)
    assert mask.origin == (2.0, 2.0)


def test__masked_parallel_fpr_from():

    layout = ac.Layout2DCI(shape_2d=(10, 3), region_list=[(1, 4, 0, 3)])

    mask = ac.Mask2D.masked_parallel_fpr_from(
        layout=layout,
        settings=ac.SettingsMask2D(parallel_fpr_pixels=(0, 2)),
        pixel_scales=0.1,
    )

    assert type(mask) == ac.Mask2D

    assert (
        mask
        == np.array(
            [
                [False, False, False],
                [True, True, True],
                [True, True, True],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
            ]
        )
    ).all()

    layout = ac.Layout2DCI(shape_2d=(10, 3), region_list=[(1, 4, 0, 3)])

    mask = ac.Mask2D.masked_parallel_fpr_from(
        layout=layout,
        settings=ac.SettingsMask2D(parallel_fpr_pixels=(0, 2)),
        pixel_scales=0.1,
        invert=True,
    )

    assert type(mask) == ac.Mask2D

    assert (
        mask
        == np.array(
            [
                [True, True, True],
                [False, False, False],
                [False, False, False],
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, True],
            ]
        )
    ).all()

    layout = ac.Layout2DCI(shape_2d=(10, 3), region_list=[(1, 4, 0, 1), (1, 4, 2, 3)])

    mask = ac.Mask2D.masked_parallel_fpr_from(
        layout=layout,
        settings=ac.SettingsMask2D(parallel_fpr_pixels=(0, 2)),
        pixel_scales=0.1,
    )

    assert type(mask) == ac.Mask2D

    assert (
        mask
        == np.array(
            [
                [False, False, False],
                [True, False, True],
                [True, False, True],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
            ]
        )
    ).all()


def test__masked_parallel_epers_from():

    layout = ac.Layout2DCI(shape_2d=(10, 3), region_list=[(1, 4, 0, 3)])

    mask = ac.Mask2D.masked_parallel_epers_from(
        layout=layout,
        settings=ac.SettingsMask2D(parallel_epers_pixels=(0, 4)),
        pixel_scales=0.1,
    )

    assert type(mask) == ac.Mask2D

    assert (
        mask
        == np.array(
            [
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [False, False, False],
                [False, False, False],
            ]
        )
    ).all()

    layout = ac.Layout2DCI(shape_2d=(10, 3), region_list=[(1, 4, 0, 3)])

    mask = ac.Mask2D.masked_parallel_epers_from(
        layout=layout,
        settings=ac.SettingsMask2D(parallel_epers_pixels=(0, 4)),
        pixel_scales=0.1,
        invert=True,
    )

    assert type(mask) == ac.Mask2D

    assert (
        mask
        == np.array(
            [
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [True, True, True],
                [True, True, True],
            ]
        )
    ).all()

    layout = ac.Layout2DCI(shape_2d=(10, 3), region_list=[(1, 4, 0, 1), (1, 4, 2, 3)])

    mask = ac.Mask2D.masked_parallel_epers_from(
        layout=layout,
        settings=ac.SettingsMask2D(parallel_epers_pixels=(0, 4)),
        pixel_scales=0.1,
    )

    assert type(mask) == ac.Mask2D

    assert (
        mask
        == np.array(
            [
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [True, False, True],
                [True, False, True],
                [True, False, True],
                [True, False, True],
                [False, False, False],
                [False, False, False],
            ]
        )
    ).all()


def test__masked_serial_fpr_from():

    layout = ac.Layout2DCI(shape_2d=(3, 10), region_list=[(0, 3, 1, 4)])

    mask = ac.Mask2D.masked_serial_fpr_from(
        layout=layout,
        settings=ac.SettingsMask2D(serial_fpr_pixels=(0, 2)),
        pixel_scales=0.1,
    )

    assert type(mask) == ac.Mask2D

    assert (
        mask
        == np.array(
            [
                [False, True, True, False, False, False, False, False, False, False],
                [False, True, True, False, False, False, False, False, False, False],
                [False, True, True, False, False, False, False, False, False, False],
            ]
        )
    ).all()

    layout = ac.Layout2DCI(shape_2d=(3, 10), region_list=[(0, 3, 1, 4)])

    mask = ac.Mask2D.masked_serial_fpr_from(
        layout=layout,
        settings=ac.SettingsMask2D(serial_fpr_pixels=(0, 2)),
        pixel_scales=0.1,
        invert=True,
    )

    assert type(mask) == ac.Mask2D

    assert (
        mask
        == np.array(
            [
                [True, False, False, True, True, True, True, True, True, True],
                [True, False, False, True, True, True, True, True, True, True],
                [True, False, False, True, True, True, True, True, True, True],
            ]
        )
    ).all()

    layout = ac.Layout2DCI(shape_2d=(3, 10), region_list=[(0, 1, 1, 4), (2, 3, 1, 4)])

    mask = ac.Mask2D.masked_serial_fpr_from(
        layout=layout,
        settings=ac.SettingsMask2D(serial_fpr_pixels=(0, 3)),
        pixel_scales=0.1,
    )

    assert type(mask) == ac.Mask2D

    assert (
        mask
        == np.array(
            [
                [False, True, True, True, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False, False, False],
                [False, True, True, True, False, False, False, False, False, False],
            ]
        )
    ).all()


def test__masked_serial_epers_from():

    layout = ac.Layout2DCI(shape_2d=(3, 10), region_list=[(0, 3, 1, 4)])

    mask = ac.Mask2D.masked_serial_epers_from(
        layout=layout,
        settings=ac.SettingsMask2D(serial_eper_pixels=(0, 6)),
        pixel_scales=0.1,
    )

    assert type(mask) == ac.Mask2D

    assert (
        mask
        == np.array(
            [
                [False, False, False, False, True, True, True, True, True, True],
                [False, False, False, False, True, True, True, True, True, True],
                [False, False, False, False, True, True, True, True, True, True],
            ]
        )
    ).all()

    mask = ac.Mask2D.masked_serial_epers_from(
        layout=layout,
        settings=ac.SettingsMask2D(serial_eper_pixels=(0, 6)),
        pixel_scales=0.1,
        invert=True,
    )

    assert type(mask) == ac.Mask2D

    assert (
        mask
        == np.array(
            [
                [True, True, True, True, False, False, False, False, False, False],
                [True, True, True, True, False, False, False, False, False, False],
                [True, True, True, True, False, False, False, False, False, False],
            ]
        )
    ).all()

    layout = ac.Layout2DCI(shape_2d=(3, 10), region_list=[(0, 1, 1, 4), (2, 3, 1, 4)])

    mask = ac.Mask2D.masked_serial_epers_from(
        layout=layout,
        settings=ac.SettingsMask2D(serial_eper_pixels=(0, 6)),
        pixel_scales=0.1,
    )

    assert type(mask) == ac.Mask2D

    assert (
        mask
        == np.array(
            [
                [False, False, False, False, True, True, True, True, True, True],
                [False, False, False, False, False, False, False, False, False, False],
                [False, False, False, False, True, True, True, True, True, True],
            ]
        )
    ).all()


def test__masked_fprs_and_epers_from(imaging_ci_7x7):

    unmasked = ac.Mask2D.unmasked(
        shape_native=imaging_ci_7x7.shape_native, pixel_scales=1.0
    )

    layout = ac.Layout2DCI(shape_2d=(7, 7), region_list=[(1, 5, 1, 5)])

    mask = ac.Mask2D.masked_fprs_and_epers_from(
        layout=layout,
        mask=unmasked,
        settings=ac.SettingsMask2D(parallel_fpr_pixels=(0, 1)),
        pixel_scales=0.1,
    )

    assert (
        mask
        == np.array(
            [
                [False, False, False, False, False, False, False],
                [False, True, True, True, True, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
            ]
        )
    ).all()

    mask = ac.Mask2D.masked_fprs_and_epers_from(
        layout=layout,
        mask=unmasked,
        settings=ac.SettingsMask2D(parallel_epers_pixels=(0, 1)),
        pixel_scales=0.1,
    )

    assert (
        mask
        == np.array(
            [
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, True, True, True, True, False, False],
                [False, False, False, False, False, False, False],
            ]
        )
    ).all()

    mask = ac.Mask2D.masked_fprs_and_epers_from(
        layout=layout,
        mask=unmasked,
        settings=ac.SettingsMask2D(serial_fpr_pixels=(0, 1)),
        pixel_scales=0.1,
    )

    assert (
        mask
        == np.array(
            [
                [False, False, False, False, False, False, False],
                [False, True, False, False, False, False, False],
                [False, True, False, False, False, False, False],
                [False, True, False, False, False, False, False],
                [False, True, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
            ]
        )
    ).all()

    mask = ac.Mask2D.masked_fprs_and_epers_from(
        layout=layout,
        mask=unmasked,
        settings=ac.SettingsMask2D(serial_eper_pixels=(0, 1)),
        pixel_scales=0.1,
    )

    assert (
        mask
        == np.array(
            [
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, True, False],
                [False, False, False, False, False, True, False],
                [False, False, False, False, False, True, False],
                [False, False, False, False, False, True, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
            ]
        )
    ).all()
