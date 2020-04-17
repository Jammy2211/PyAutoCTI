import os

import numpy as np
import pytest
import shutil

from autocti.util import exc
from autocti import structures as struct
from autocti import dataset as ds

test_data_path = "{}/files/array/".format(os.path.dirname(os.path.realpath(__file__)))


class TestImaging:
    def test__new_imaging__signal_to_noise_limit_above_max_signal_to_noise__signal_to_noise_map_unchanged(
        self
    ):
        image = struct.Array.full(fill_value=20.0, shape_2d=(2, 2))
        image[1, 1] = 5.0

        noise_map_array = struct.Array.full(fill_value=5.0, shape_2d=(2, 2))
        noise_map_array[1, 1] = 2.0

        imaging = ds.Imaging(image=image, noise_map=noise_map_array)

        imaging = imaging.signal_to_noise_limited_from_signal_to_noise_limit(
            signal_to_noise_limit=100.0
        )

        assert (imaging.image == np.array([[20.0, 20.0], [20.0, 5.0]])).all()

        assert (imaging.noise_map == np.array([[5.0, 5.0], [5.0, 2.0]])).all()

        assert (imaging.signal_to_noise_map == np.array([[4.0, 4.0], [4.0, 2.5]])).all()

    def test__new_imaging__signal_to_noise_limit_below_max_signal_to_noise__signal_to_noise_map_capped_to_limit(
        self
    ):
        image = struct.Array.full(fill_value=20.0, shape_2d=(2, 2))
        image[1, 1] = 5.0

        noise_map_array = struct.Array.full(fill_value=5.0, shape_2d=(2, 2))
        noise_map_array[1, 1] = 2.0

        imaging = ds.Imaging(image=image, noise_map=noise_map_array)

        imaging_capped = imaging.signal_to_noise_limited_from_signal_to_noise_limit(
            signal_to_noise_limit=2.0
        )

        assert (imaging_capped.image == np.array([[20.0, 20.0], [20.0, 5.0]])).all()

        assert (imaging_capped.noise_map == np.array([[10.0, 10.0], [10.0, 2.5]])).all()

        assert (
            imaging_capped.signal_to_noise_map == np.array([[2.0, 2.0], [2.0, 2.0]])
        ).all()

        imaging_capped = imaging.signal_to_noise_limited_from_signal_to_noise_limit(
            signal_to_noise_limit=3.0
        )

        assert (imaging_capped.image == np.array([[20.0, 20.0], [20.0, 5.0]])).all()

        assert (
            imaging_capped.noise_map
            == np.array([[(20.0 / 3.0), (20.0 / 3.0)], [(20.0 / 3.0), 2.0]])
        ).all()

        assert (
            imaging_capped.signal_to_noise_map == np.array([[3.0, 3.0], [3.0, 2.5]])
        ).all()

    def test__from_fits__loads_arrays_and_psf_is_renormalized(self):

        imaging = ds.Imaging.from_fits(
            pixel_scales=0.1,
            image_path=test_data_path + "3x3_ones.fits",
            noise_map_path=test_data_path + "3x3_threes.fits",
        )

        assert (imaging.image == np.ones((3, 3))).all()
        assert (imaging.noise_map == 3.0 * np.ones((3, 3))).all()

        assert imaging.pixel_scales == (0.1, 0.1)
        assert imaging.noise_map.mask.pixel_scales == (0.1, 0.1)

    def test__optional_array_paths_included__loads_optional_array(self):
        imaging = ds.Imaging.from_fits(
            pixel_scales=0.1,
            image_path=test_data_path + "3x3_ones.fits",
            noise_map_path=test_data_path + "3x3_threes.fits",
        )

        assert (imaging.image == np.ones((3, 3))).all()
        assert (imaging.noise_map == 3.0 * np.ones((3, 3))).all()

        assert imaging.pixel_scales == (0.1, 0.1)
        assert imaging.noise_map.mask.pixel_scales == (0.1, 0.1)

    def test__from_fits__all_files_in_one_fits__load_using_different_hdus(self):
        imaging = ds.Imaging.from_fits(
            pixel_scales=0.1,
            image_path=test_data_path + "3x3_multiple_hdu.fits",
            image_hdu=0,
            noise_map_path=test_data_path + "3x3_multiple_hdu.fits",
            noise_map_hdu=2,
        )

        assert (imaging.image == np.ones((3, 3))).all()
        assert (imaging.noise_map == 3.0 * np.ones((3, 3))).all()

        assert imaging.pixel_scales == (0.1, 0.1)
        assert imaging.noise_map.mask.pixel_scales == (0.1, 0.1)

    def test__output_to_fits__outputs_all_imaging_arrays(self):
        imaging = ds.Imaging.from_fits(
            pixel_scales=0.1,
            image_path=test_data_path + "3x3_ones.fits",
            noise_map_path=test_data_path + "3x3_threes.fits",
        )

        output_data_dir = "{}/../files/array/output_test/".format(
            os.path.dirname(os.path.realpath(__file__))
        )
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        imaging.output_to_fits(
            image_path=output_data_dir + "image.fits",
            noise_map_path=output_data_dir + "noise_map.fits",
        )

        imaging = ds.Imaging.from_fits(
            pixel_scales=0.1,
            image_path=output_data_dir + "image.fits",
            noise_map_path=output_data_dir + "noise_map.fits",
        )

        assert (imaging.image == np.ones((3, 3))).all()
        assert (imaging.noise_map == 3.0 * np.ones((3, 3))).all()

        assert imaging.pixel_scales == (0.1, 0.1)
        assert imaging.noise_map.mask.pixel_scales == (0.1, 0.1)


class TestMaskedImaging:
    def test__masked_dataset(self, imaging_7x7, mask_7x7):

        masked_imaging_7x7 = ds.MaskedImaging(imaging=imaging_7x7, mask=mask_7x7)

        assert (masked_imaging_7x7.image == np.ones((7, 7)) * np.invert(mask_7x7)).all()

        assert (
            masked_imaging_7x7.noise_map == 2.0 * np.ones((7, 7)) * np.invert(mask_7x7)
        ).all()

    def test__different_imaging_without_mock_objects__customize_constructor_inputs(
        self
    ):

        imaging = ds.Imaging(
            image=struct.Array.ones(shape_2d=(19, 19), pixel_scales=3.0),
            noise_map=struct.Array.full(
                fill_value=2.0, shape_2d=(19, 19), pixel_scales=3.0
            ),
        )
        mask = struct.Mask.unmasked(shape_2d=(19, 19), pixel_scales=1.0, invert=True)
        mask[9, 9] = False

        masked_imaging = ds.MaskedImaging(imaging=imaging, mask=mask)

        assert (masked_imaging.imaging.image == np.ones((19, 19))).all()
        assert (masked_imaging.imaging.noise_map == 2.0 * np.ones((19, 19))).all()
        assert (masked_imaging.image[9, 9] == np.array([1.0])).all()
        assert (masked_imaging.noise_map[9, 9] == np.array([2.0])).all()

    def test__modified_noise_map(self, noise_map_7x7, imaging_7x7, mask_7x7):

        masked_imaging_7x7 = ds.MaskedImaging(imaging=imaging_7x7, mask=mask_7x7)

        noise_map_7x7[0, 0] = 11.0

        masked_imaging_7x7 = masked_imaging_7x7.modify_noise_map(
            noise_map=noise_map_7x7
        )

        assert masked_imaging_7x7.noise_map[0, 0] == 11.0
