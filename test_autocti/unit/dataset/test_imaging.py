import os
import shutil

import numpy as np
import autocti as ac

test_data_path = "{}/files/array/".format(os.path.dirname(os.path.realpath(__file__)))


class TestImaging:
    def test__from_fits__loads_arrays_and_psf_is_normalized(self):

        imaging = ac.Imaging.from_fits(
            pixel_scales=0.1,
            image_path=test_data_path + "3x3_ones.fits",
            noise_map_path=test_data_path + "3x3_threes.fits",
        )

        assert (imaging.image == np.ones((3, 3))).all()
        assert (imaging.noise_map == 3.0 * np.ones((3, 3))).all()

        assert imaging.pixel_scales == (0.1, 0.1)
        assert imaging.noise_map.mask.pixel_scales == (0.1, 0.1)

    def test__optional_array_paths_included__loads_optional_array(self):
        imaging = ac.Imaging.from_fits(
            pixel_scales=0.1,
            image_path=test_data_path + "3x3_ones.fits",
            noise_map_path=test_data_path + "3x3_threes.fits",
        )

        assert (imaging.image == np.ones((3, 3))).all()
        assert (imaging.noise_map == 3.0 * np.ones((3, 3))).all()

        assert imaging.pixel_scales == (0.1, 0.1)
        assert imaging.noise_map.mask.pixel_scales == (0.1, 0.1)

    def test__from_fits__all_files_in_one_fits__load_using_different_hdus(self):
        imaging = ac.Imaging.from_fits(
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
        imaging = ac.Imaging.from_fits(
            pixel_scales=0.1,
            image_path=test_data_path + "3x3_ones.fits",
            noise_map_path=test_data_path + "3x3_threes.fits",
        )

        output_data_dir = "{}/files/array/output_test/".format(
            os.path.dirname(os.path.realpath(__file__))
        )
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        imaging.output_to_fits(
            image_path=output_data_dir + "image.fits",
            noise_map_path=output_data_dir + "noise_map.fits",
        )

        imaging = ac.Imaging.from_fits(
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

        masked_imaging_7x7 = ac.MaskedImaging(imaging=imaging_7x7, mask=mask_7x7)

        assert (masked_imaging_7x7.image == np.ones((7, 7)) * np.invert(mask_7x7)).all()

        assert (
            masked_imaging_7x7.noise_map == 2.0 * np.ones((7, 7)) * np.invert(mask_7x7)
        ).all()

    def test__different_imaging_without_mock_objects__customize_constructor_inputs(
        self
    ):

        imaging = ac.Imaging(
            image=ac.Array.ones(shape_2d=(19, 19), pixel_scales=3.0),
            noise_map=ac.Array.full(
                fill_value=2.0, shape_2d=(19, 19), pixel_scales=3.0
            ),
        )
        mask = ac.Mask2D.unmasked(shape_2d=(19, 19), pixel_scales=1.0, invert=True)
        mask[9, 9] = False

        masked_imaging = ac.MaskedImaging(imaging=imaging, mask=mask)

        assert (masked_imaging.imaging.image == np.ones((19, 19))).all()
        assert (masked_imaging.imaging.noise_map == 2.0 * np.ones((19, 19))).all()
        assert (masked_imaging.image[9, 9] == np.array([1.0])).all()
        assert (masked_imaging.noise_map[9, 9] == np.array([2.0])).all()
