import os
from os import path
import shutil

import numpy as np
import pytest
import autocti as ac

test_data_path = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "arrays"
)


class TestSettingsImagingCI:
    def test__modify_via_fit_type(self):

        settings = ac.ci.SettingsImagingCI(parallel_columns=None, serial_rows=None)
        settings = settings.modify_via_fit_type(
            is_parallel_fit=False, is_serial_fit=False
        )
        assert settings.parallel_columns is None
        assert settings.serial_rows is None

        settings = ac.ci.SettingsImagingCI(parallel_columns=1, serial_rows=1)
        settings = settings.modify_via_fit_type(
            is_parallel_fit=False, is_serial_fit=False
        )
        assert settings.parallel_columns == 1
        assert settings.serial_rows == 1

        settings = ac.ci.SettingsImagingCI(parallel_columns=1, serial_rows=1)
        settings = settings.modify_via_fit_type(
            is_parallel_fit=True, is_serial_fit=False
        )
        assert settings.parallel_columns == 1
        assert settings.serial_rows is None

        settings = ac.ci.SettingsImagingCI(parallel_columns=1, serial_rows=1)
        settings = settings.modify_via_fit_type(
            is_parallel_fit=False, is_serial_fit=True
        )
        assert settings.parallel_columns is None
        assert settings.serial_rows == 1

        settings = ac.ci.SettingsImagingCI(parallel_columns=1, serial_rows=1)
        settings = settings.modify_via_fit_type(
            is_parallel_fit=True, is_serial_fit=True
        )
        assert settings.parallel_columns is None
        assert settings.serial_rows is None


class TestImagingCI:
    def test__parallel_calibration_imaging_ci_from(self, imaging_ci_7x7):

        # The ci layout_ci starts at column 1, so the left most column is removed below

        parallel_calibration_imaging = imaging_ci_7x7.parallel_calibration_imaging_from(
            columns=(0, 6)
        )

        assert (
            parallel_calibration_imaging.image.native
            == imaging_ci_7x7.image.native[:, 1:7]
        ).all()
        assert (
            parallel_calibration_imaging.noise_map.native
            == imaging_ci_7x7.noise_map.native[:, 1:7]
        ).all()
        assert (
            parallel_calibration_imaging.pre_cti_image.native
            == imaging_ci_7x7.pre_cti_image.native[:, 1:7]
        ).all()
        assert (
            parallel_calibration_imaging.cosmic_ray_map.native
            == imaging_ci_7x7.cosmic_ray_map.native[:, 1:7]
        ).all()

        assert parallel_calibration_imaging.layout.region_list == [(1, 5, 0, 4)]

    def test__serial_calibration_imaging_ci_from_rows(self, imaging_ci_7x7):

        # The ci layout_ci spans 2 rows, so two rows are extracted

        serial_calibration_imaging = imaging_ci_7x7.serial_calibration_imaging_for_rows(
            rows=(0, 2)
        )

        assert (
            serial_calibration_imaging.image.native
            == imaging_ci_7x7.image.native[0:2, :]
        ).all()
        assert (
            serial_calibration_imaging.noise_map.native
            == imaging_ci_7x7.noise_map.native[0:2, :]
        ).all()
        assert (
            serial_calibration_imaging.pre_cti_image.native
            == imaging_ci_7x7.pre_cti_image.native[0:2, :]
        ).all()
        assert (
            serial_calibration_imaging.cosmic_ray_map.native
            == imaging_ci_7x7.cosmic_ray_map.native[1:3, :]
        ).all()

        assert serial_calibration_imaging.layout.region_list == [(0, 2, 1, 5)]

    def test__from_fits__load_all_data_components__has_correct_attributes(
        self, layout_ci_7x7
    ):

        imaging = ac.ci.ImagingCI.from_fits(
            pixel_scales=1.0,
            layout=layout_ci_7x7,
            image_path=path.join(test_data_path, "3x3_ones.fits"),
            image_hdu=0,
            noise_map_path=path.join(test_data_path, "3x3_twos.fits"),
            noise_map_hdu=0,
            pre_cti_image_path=path.join(test_data_path, "3x3_threes.fits"),
            pre_cti_image_hdu=0,
            cosmic_ray_map_path=path.join(test_data_path, "3x3_fours.fits"),
            cosmic_ray_map_hdu=0,
        )

        assert (imaging.image.native == np.ones((3, 3))).all()
        assert (imaging.noise_map.native == 2.0 * np.ones((3, 3))).all()
        assert (imaging.pre_cti_image.native == 3.0 * np.ones((3, 3))).all()
        assert (imaging.cosmic_ray_map.native == 4.0 * np.ones((3, 3))).all()

        assert imaging.layout == layout_ci_7x7

    def test__from_fits__load_all_image_components__load_from_multi_hdu_fits(
        self, layout_ci_7x7
    ):

        imaging = ac.ci.ImagingCI.from_fits(
            pixel_scales=1.0,
            layout=layout_ci_7x7,
            image_path=path.join(test_data_path, "3x3_multiple_hdu.fits"),
            image_hdu=0,
            noise_map_path=path.join(test_data_path, "3x3_multiple_hdu.fits"),
            noise_map_hdu=1,
            pre_cti_image_path=path.join(test_data_path, "3x3_multiple_hdu.fits"),
            pre_cti_image_hdu=2,
            cosmic_ray_map_path=path.join(test_data_path, "3x3_multiple_hdu.fits"),
            cosmic_ray_map_hdu=3,
        )

        assert (imaging.image.native == np.ones((3, 3))).all()
        assert (imaging.noise_map.native == 2.0 * np.ones((3, 3))).all()
        assert (imaging.pre_cti_image.native == 3.0 * np.ones((3, 3))).all()
        assert (imaging.cosmic_ray_map.native == 4.0 * np.ones((3, 3))).all()

        assert imaging.layout == layout_ci_7x7

    def test__from_fits__noise_map_from_single_value(self, layout_ci_7x7):

        imaging = ac.ci.ImagingCI.from_fits(
            pixel_scales=1.0,
            layout=layout_ci_7x7,
            image_path=path.join(test_data_path, "3x3_ones.fits"),
            image_hdu=0,
            noise_map_from_single_value=10.0,
            pre_cti_image_path=path.join(test_data_path, "3x3_threes.fits"),
            pre_cti_image_hdu=0,
        )

        assert (imaging.image.native == np.ones((3, 3))).all()
        assert (imaging.noise_map.native == 10.0 * np.ones((3, 3))).all()
        assert (imaging.pre_cti_image.native == 3.0 * np.ones((3, 3))).all()
        assert imaging.cosmic_ray_map == None

        assert imaging.layout == layout_ci_7x7

    def test__from_fits__load_pre_cti_image_image_from_the_layout_ci_and_image(self):

        layout_ci = ac.ci.Layout2DCIUniform(
            shape_2d=(3, 3), region_list=[(0, 3, 0, 3)], normalization=10.0
        )

        imaging = ac.ci.ImagingCI.from_fits(
            pixel_scales=1.0,
            layout=layout_ci,
            image_path=path.join(test_data_path, "3x3_ones.fits"),
            image_hdu=0,
            noise_map_path=path.join(test_data_path, "3x3_twos.fits"),
            noise_map_hdu=0,
        )

        assert (imaging.image.native == np.ones((3, 3))).all()
        assert (imaging.noise_map.native == 2.0 * np.ones((3, 3))).all()
        assert (imaging.pre_cti_image.native == 10.0 * np.ones((3, 3))).all()
        assert imaging.cosmic_ray_map == None

        assert imaging.layout == layout_ci

    def test__output_to_fits___all_arrays(self, layout_ci_7x7):

        imaging = ac.ci.ImagingCI.from_fits(
            pixel_scales=1.0,
            layout=layout_ci_7x7,
            image_path=path.join(test_data_path, "3x3_ones.fits"),
            image_hdu=0,
            noise_map_path=path.join(test_data_path, "3x3_twos.fits"),
            noise_map_hdu=0,
            pre_cti_image_path=path.join(test_data_path, "3x3_threes.fits"),
            pre_cti_image_hdu=0,
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
            pre_cti_image_path=path.join(output_data_dir, "pre_cti_image.fits"),
            cosmic_ray_map_path=path.join(output_data_dir, "cosmic_ray_map.fits"),
        )

        imaging = ac.ci.ImagingCI.from_fits(
            pixel_scales=1.0,
            layout=layout_ci_7x7,
            image_path=path.join(output_data_dir, "image.fits"),
            image_hdu=0,
            noise_map_path=path.join(output_data_dir, "noise_map.fits"),
            noise_map_hdu=0,
            pre_cti_image_path=path.join(output_data_dir, "pre_cti_image.fits"),
            pre_cti_image_hdu=0,
            cosmic_ray_map_path=path.join(output_data_dir, "cosmic_ray_map.fits"),
            cosmic_ray_map_hdu=0,
        )

        assert (imaging.image.native == np.ones((3, 3))).all()
        assert (imaging.noise_map.native == 2.0 * np.ones((3, 3))).all()
        assert (imaging.pre_cti_image.native == 3.0 * np.ones((3, 3))).all()
        assert (imaging.cosmic_ray_map.native == 4.0 * np.ones((3, 3))).all()


class TestApplyMask:
    def test__construtor__masks_arrays_correctly(self, imaging_ci_7x7):

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

        assert (masked_imaging_ci.pre_cti_image == imaging_ci_7x7.pre_cti_image).all()

        masked_cosmic_ray_map = imaging_ci_7x7.cosmic_ray_map
        masked_cosmic_ray_map[0, 0] = 0.0

        assert (masked_imaging_ci.cosmic_ray_map == masked_cosmic_ray_map).all()


class TestApplySettings:
    def test__include_parallel_columns_extraction(
        self, imaging_ci_7x7, mask_2d_7x7_unmasked, ci_noise_scaling_map_list_7x7
    ):

        mask = ac.Mask2D.unmasked(
            shape_native=imaging_ci_7x7.shape_native, pixel_scales=1.0
        )
        mask[0, 2] = True

        masked_imaging_ci = imaging_ci_7x7.apply_mask(mask=mask)
        masked_imaging_ci = masked_imaging_ci.apply_settings(
            settings=ac.ci.SettingsImagingCI(parallel_columns=(1, 3))
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

        pre_cti_image = 10.0 * np.ones((7, 2))

        assert masked_imaging_ci.pre_cti_image == pytest.approx(pre_cti_image, 1.0e-4)

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

    def test__serial_masked_imaging_ci(
        self, imaging_ci_7x7, mask_2d_7x7_unmasked, ci_noise_scaling_map_list_7x7
    ):

        mask = ac.Mask2D.unmasked(
            shape_native=imaging_ci_7x7.shape_native, pixel_scales=1.0
        )
        mask[1, 0] = True

        masked_imaging_ci = imaging_ci_7x7.apply_mask(mask=mask)
        masked_imaging_ci = masked_imaging_ci.apply_settings(
            settings=ac.ci.SettingsImagingCI(serial_rows=(0, 1))
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

        pre_cti_image = 10.0 * np.ones((1, 7))

        assert masked_imaging_ci.pre_cti_image == pytest.approx(pre_cti_image, 1.0e-4)

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


class TestSimulatorImagingCI(object):
    def test__no_instrumental_effects_input__only_cti_simulated(
        self, parallel_clocker, traps_x2, ccd
    ):

        layout_ci = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 5),
            normalization=10.0,
            region_list=[(0, 1, 0, 5)],
            serial_overscan=ac.Region2D((1, 2, 1, 2)),
        )

        simulator = ac.ci.SimulatorImagingCI(pixel_scales=1.0, add_poisson_noise=False)

        imaging = simulator.from_layout(
            layout=layout_ci,
            clocker=parallel_clocker,
            parallel_traps=traps_x2,
            parallel_ccd=ccd,
        )

        assert imaging.image[0, 0:5] == pytest.approx(
            np.array([9.43, 9.43, 9.43, 9.43, 9.43]), 1e-1
        )
        assert imaging.layout == layout_ci

    def test__include_read_noise__is_added_after_cti(
        self, parallel_clocker, traps_x2, ccd
    ):

        layout_ci = ac.ci.Layout2DCIUniform(
            shape_2d=(3, 3),
            normalization=10.0,
            region_list=[(0, 1, 0, 3)],
            serial_overscan=ac.Region2D((1, 2, 1, 2)),
        )

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0, read_noise=1.0, add_poisson_noise=True, noise_seed=1
        )

        imaging = simulator.from_layout(
            layout=layout_ci, clocker=parallel_clocker
        )

        image_no_noise = layout_ci.pre_cti_image_from(
            shape_native=(3, 3), pixel_scales=1.0
        )

        # Use seed to give us a known read noises map we'll test_autocti for

        assert imaging.image - image_no_noise.native == pytest.approx(
            np.array([[1.62, -0.61, -0.53], [-1.07, 0.87, -2.30], [1.74, -0.76, 0.32]]),
            1e-2,
        )
        assert imaging.layout == layout_ci

    def test__include_cosmics__is_added_to_image_and_trailed(
        self, parallel_clocker, traps_x2, ccd
    ):

        layout_ci = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 5),
            normalization=10.0,
            region_list=[(0, 1, 0, 5)],
            serial_overscan=ac.Region2D((1, 2, 1, 2)),
        )

        simulator = ac.ci.SimulatorImagingCI(pixel_scales=1.0, add_poisson_noise=False)

        cosmic_ray_map = np.zeros((5, 5))
        cosmic_ray_map[2, 2] = 100.0

        imaging = simulator.from_layout(
            layout=layout_ci,
            clocker=parallel_clocker,
            parallel_traps=traps_x2,
            parallel_ccd=ccd,
            cosmic_ray_map=cosmic_ray_map,
        )

        assert imaging.image[0, 0:5] == pytest.approx(
            np.array([9.43, 9.43, 9.43, 9.43, 9.43]), 1e-1
        )
        assert 0.0 < imaging.image[1, 1] < 100.0
        assert imaging.image[2, 2] > 94.0
        assert (imaging.image[1, 1:4] > 0.0).all()
        assert (
            imaging.cosmic_ray_map
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 100.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()
        assert imaging.layout == layout_ci

    def test__from_pre_cti_image(self, parallel_clocker, traps_x2, ccd):

        layout_ci = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 5),
            normalization=10.0,
            region_list=[(0, 1, 0, 5)],
            serial_overscan=ac.Region2D((1, 2, 1, 2)),
        )

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0, read_noise=4.0, add_poisson_noise=False, noise_seed=1
        )

        cosmic_ray_map = np.zeros((5, 5))
        cosmic_ray_map[2, 2] = 100.0

        imaging = simulator.from_layout(
            layout=layout_ci,
            clocker=parallel_clocker,
            parallel_traps=traps_x2,
            parallel_ccd=ccd,
            cosmic_ray_map=cosmic_ray_map,
        )

        pre_cti_image = layout_ci.pre_cti_image_from(shape_native=(5, 5), pixel_scales=1.0)

        imaging_via_pre_cti_image = simulator.from_pre_cti_image(
            pre_cti_image=pre_cti_image.native,
            layout=layout_ci,
            clocker=parallel_clocker,
            parallel_traps=traps_x2,
            parallel_ccd=ccd,
            cosmic_ray_map=cosmic_ray_map,
        )

        assert (imaging.image == imaging_via_pre_cti_image.image).all()
        assert (imaging.noise_map == imaging_via_pre_cti_image.noise_map).all()
        assert (imaging.pre_cti_image == imaging_via_pre_cti_image.pre_cti_image).all()
        assert (imaging.cosmic_ray_map == imaging_via_pre_cti_image.cosmic_ray_map).all()

    def test__from_post_cti_image(self, parallel_clocker, traps_x2, ccd):

        layout_ci = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 5),
            normalization=10.0,
            region_list=[(0, 1, 0, 5)],
            serial_overscan=ac.Region2D((1, 2, 1, 2)),
        )

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0, read_noise=4.0, add_poisson_noise=False, noise_seed=1
        )

        cosmic_ray_map = np.zeros((5, 5))
        cosmic_ray_map[2, 2] = 100.0

        imaging = simulator.from_layout(
            layout=layout_ci,
            clocker=parallel_clocker,
            parallel_traps=traps_x2,
            parallel_ccd=ccd,
            cosmic_ray_map=cosmic_ray_map,
        )

        pre_cti_image = layout_ci.pre_cti_image_from(
            shape_native=(5, 5), pixel_scales=1.0
        ).native
        pre_cti_image += cosmic_ray_map

        post_cti_image = parallel_clocker.add_cti(
            image=pre_cti_image, parallel_traps=traps_x2, parallel_ccd=ccd
        )

        imaging_via_post_cti_image = simulator.from_post_cti_image(
            post_cti_image=post_cti_image,
            pre_cti_image=pre_cti_image.native,
            layout=layout_ci,
            cosmic_ray_map=cosmic_ray_map,
        )

        assert (imaging.image == imaging_via_post_cti_image.image).all()
        assert (imaging.noise_map == imaging_via_post_cti_image.noise_map).all()
        assert (imaging.pre_cti_image == imaging_via_post_cti_image.pre_cti_image).all()
        assert (imaging.cosmic_ray_map == imaging_via_post_cti_image.cosmic_ray_map).all()
