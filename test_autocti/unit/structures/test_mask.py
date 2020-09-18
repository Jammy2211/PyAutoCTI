import os
import shutil

import numpy as np
import pytest
import autocti as ac
from autocti import exc

test_data_path = "{}/files/array/".format(os.path.dirname(os.path.realpath(__file__)))


class TestSettingsMask:
    def test__cosmic_ray_buffer_tag(self):

        settings = ac.SettingsMask(
            cosmic_ray_parallel_buffer=None,
            cosmic_ray_serial_buffer=None,
            cosmic_ray_diagonal_buffer=None,
        )
        assert settings.cosmic_ray_buffer_tag == ""

        settings = ac.SettingsMask(
            cosmic_ray_parallel_buffer=1,
            cosmic_ray_serial_buffer=None,
            cosmic_ray_diagonal_buffer=3,
        )
        assert settings.cosmic_ray_buffer_tag == "__cr_p1d3"

        settings = ac.SettingsMask(
            cosmic_ray_parallel_buffer=10,
            cosmic_ray_serial_buffer=20,
            cosmic_ray_diagonal_buffer=None,
        )
        assert settings.cosmic_ray_buffer_tag == "__cr_p10s20"

        settings = ac.SettingsMask(
            cosmic_ray_parallel_buffer=1,
            cosmic_ray_serial_buffer=2,
            cosmic_ray_diagonal_buffer=3,
        )
        assert settings.cosmic_ray_buffer_tag == "__cr_p1s2d3"

        settings = ac.SettingsMask(
            cosmic_ray_parallel_buffer=10,
            cosmic_ray_serial_buffer=5,
            cosmic_ray_diagonal_buffer=1,
        )
        assert settings.cosmic_ray_buffer_tag == "__cr_p10s5d1"

    def test__tag(self):

        settings = ac.SettingsMask(
            cosmic_ray_parallel_buffer=10,
            cosmic_ray_serial_buffer=5,
            cosmic_ray_diagonal_buffer=1,
        )
        assert settings.tag == "mask[__cr_p10s5d1]"


class TestMask:
    def test__mask__makes_mask_without_other_inputs(self):

        mask = ac.Mask.manual(mask=[[False, False], [False, False]])

        assert type(mask) == ac.Mask
        assert (mask == np.array([[False, False], [False, False]])).all()

        mask = ac.Mask.manual(mask=[[False, False, True], [True, True, False]])

        assert type(mask) == ac.Mask
        assert (mask == np.array([[False, False, True], [True, True, False]])).all()

    def test__mask__makes_mask_with_pixel_scale(self):

        mask = ac.Mask.manual(mask=[[False, False], [True, True]], pixel_scales=1.0)

        assert type(mask) == ac.Mask
        assert (mask == np.array([[False, False], [True, True]])).all()
        assert mask.pixel_scales == (1.0, 1.0)
        assert mask.origin == (0.0, 0.0)

        mask = ac.Mask.manual(
            mask=[[False, False, True], [True, True, False]],
            pixel_scales=(2.0, 3.0),
            origin=(0.0, 1.0),
        )

        assert type(mask) == ac.Mask
        assert (mask == np.array([[False, False, True], [True, True, False]])).all()
        assert mask.pixel_scales == (2.0, 3.0)
        assert mask.origin == (0.0, 1.0)

    def test__mask__invert_is_true_inverts_the_mask(self):

        mask = ac.Mask.manual(
            mask=[[False, False, True], [True, True, False]], invert=True
        )

        assert type(mask) == ac.Mask
        assert (mask == np.array([[True, True, False], [False, False, True]])).all()

    def test__mask__input_is_1d_mask__no_shape_2d__raises_exception(self):

        with pytest.raises(exc.MaskException):

            ac.Mask.manual(mask=[False, False, True])

        with pytest.raises(exc.MaskException):

            ac.Mask.manual(mask=[False, False, True], pixel_scales=False)

        with pytest.raises(exc.MaskException):

            ac.Mask.manual(mask=[False, False, True])

        with pytest.raises(exc.MaskException):

            ac.Mask.manual(mask=[False, False, True], pixel_scales=False)

    def test__is_all_false(self):

        mask = ac.Mask.manual(mask=[[False, False], [False, False]])

        assert mask.is_all_false == True

        mask = ac.Mask.manual(mask=[[False, False]])

        assert mask.is_all_false == True

        mask = ac.Mask.manual(mask=[[False, True], [False, False]])

        assert mask.is_all_false == False

        mask = ac.Mask.manual(mask=[[True, True], [False, False]])

        assert mask.is_all_false == False


class TestUnmasked:
    def test__mask_all_unmasked__5x5__input__all_are_false(self):

        mask = ac.Mask.unmasked(shape_2d=(5, 5), invert=False)

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

        mask = ac.Mask.unmasked(shape_2d=(3, 3), pixel_scales=(1.5, 1.5), invert=False)

        assert mask.shape == (3, 3)
        assert (
            mask
            == np.array(
                [[False, False, False], [False, False, False], [False, False, False]]
            )
        ).all()

        assert mask.pixel_scales == (1.5, 1.5)
        assert mask.origin == (0.0, 0.0)

        mask = ac.Mask.unmasked(
            shape_2d=(3, 3), pixel_scales=(2.0, 2.5), invert=True, origin=(1.0, 2.0)
        )

        assert mask.shape == (3, 3)
        assert (
            mask
            == np.array([[True, True, True], [True, True, True], [True, True, True]])
        ).all()

        assert mask.pixel_scales == (2.0, 2.5)
        assert mask.origin == (1.0, 2.0)


class MockPattern(object):
    def __init__(self):
        pass


class TestMaskRemoveRegions:
    def test__remove_one_region(self):

        mask = ac.Mask.from_masked_regions(
            shape_2d=(3, 3), masked_regions=[(0, 3, 2, 3)]
        )

        assert (
            mask
            == np.array(
                [[False, False, True], [False, False, True], [False, False, True]]
            )
        ).all()

    def test__remove_two_regions(self):

        mask = ac.Mask.from_masked_regions(
            shape_2d=(3, 3), masked_regions=[(0, 3, 2, 3), (0, 2, 0, 2)]
        )

        assert (
            mask
            == np.array([[True, True, True], [True, True, True], [False, False, True]])
        ).all()


class TestCosmicRayMask:
    def test__cosmic_ray_mask_included_in_total_mask(self):

        cosmic_ray_map = ac.Frame.manual(
            array=np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            ),
            roe_corner=(1, 0),
        )

        mask = ac.Mask.from_cosmic_ray_map_buffed(
            cosmic_ray_map=cosmic_ray_map,
            settings=ac.SettingsMask(
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


class TestMaskCosmics:
    def test__mask_cosmic_ray_in_different_directions(self):

        cosmic_ray_map = ac.Frame.manual(
            array=np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            )
        )

        mask = ac.Mask.from_cosmic_ray_map_buffed(
            cosmic_ray_map=cosmic_ray_map,
            settings=ac.SettingsMask(
                cosmic_ray_parallel_buffer=1,
                cosmic_ray_serial_buffer=0,
                cosmic_ray_diagonal_buffer=0,
            ),
        )

        assert (
            mask
            == np.array(
                [[False, True, False], [False, True, False], [False, False, False]]
            )
        ).all()

        cosmic_ray_map = ac.Frame.manual(
            array=np.array(
                [[False, False, False], [False, False, False], [False, True, False]]
            )
        )

        mask = ac.Mask.from_cosmic_ray_map_buffed(
            cosmic_ray_map=cosmic_ray_map,
            settings=ac.SettingsMask(
                cosmic_ray_parallel_buffer=2,
                cosmic_ray_serial_buffer=0,
                cosmic_ray_diagonal_buffer=0,
            ),
        )

        assert (
            mask
            == np.array(
                [[False, True, False], [False, True, False], [False, True, False]]
            )
        ).all()

        cosmic_ray_map = ac.Frame.manual(
            array=np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            )
        )

        mask = ac.Mask.from_cosmic_ray_map_buffed(
            cosmic_ray_map=cosmic_ray_map,
            settings=ac.SettingsMask(
                cosmic_ray_serial_buffer=1,
                cosmic_ray_parallel_buffer=0,
                cosmic_ray_diagonal_buffer=0,
            ),
        )

        assert (
            mask
            == np.array(
                [[False, False, False], [False, True, True], [False, False, False]]
            )
        ).all()

        cosmic_ray_map = ac.Frame.manual(
            array=np.array(
                [[False, False, False], [True, False, False], [False, False, False]]
            )
        )

        mask = ac.Mask.from_cosmic_ray_map_buffed(
            cosmic_ray_map=cosmic_ray_map,
            settings=ac.SettingsMask(
                cosmic_ray_serial_buffer=2,
                cosmic_ray_parallel_buffer=0,
                cosmic_ray_diagonal_buffer=0,
            ),
        )

        assert (
            mask
            == np.array(
                [[False, False, False], [True, True, True], [False, False, False]]
            )
        ).all()

        cosmic_ray_map = ac.Frame.manual(
            array=np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            )
        )

        mask = ac.Mask.from_cosmic_ray_map_buffed(
            cosmic_ray_map=cosmic_ray_map,
            settings=ac.SettingsMask(
                cosmic_ray_diagonal_buffer=1,
                cosmic_ray_parallel_buffer=0,
                cosmic_ray_serial_buffer=0,
            ),
        )

        assert (
            mask
            == np.array(
                [[False, True, True], [False, True, True], [False, False, False]]
            )
        ).all()

        cosmic_ray_map = ac.Frame.manual(
            array=np.array(
                [
                    [False, False, False, False],
                    [False, False, False, False],
                    [False, False, False, False],
                    [False, True, False, False],
                ]
            )
        )

        mask = ac.Mask.from_cosmic_ray_map_buffed(
            cosmic_ray_map=cosmic_ray_map,
            settings=ac.SettingsMask(
                cosmic_ray_diagonal_buffer=2,
                cosmic_ray_parallel_buffer=0,
                cosmic_ray_serial_buffer=0,
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


class TestFromAndToFits:
    def test__load_and_output_mask_to_fits(self):

        mask = ac.Mask.from_fits(
            file_path=test_data_path + "3x3_ones.fits", hdu=0, pixel_scales=(1.0, 1.0)
        )

        output_data_dir = "{}/files/array/output_test/".format(
            os.path.dirname(os.path.realpath(__file__))
        )

        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        mask.output_to_fits(file_path=output_data_dir + "mask.fits")

        mask = ac.Mask.from_fits(
            file_path=output_data_dir + "mask.fits",
            hdu=0,
            pixel_scales=(1.0, 1.0),
            origin=(2.0, 2.0),
        )

        assert (mask == np.ones((3, 3))).all()
        assert mask.pixel_scales == (1.0, 1.0)
        assert mask.origin == (2.0, 2.0)
