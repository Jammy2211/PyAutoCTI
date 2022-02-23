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

        settings = ac.ci.SettingsImagingCI(parallel_pixels=None, serial_pixels=None)
        settings = settings.modify_via_fit_type(
            is_parallel_fit=False, is_serial_fit=False
        )
        assert settings.parallel_pixels is None
        assert settings.serial_pixels is None

        settings = ac.ci.SettingsImagingCI(parallel_pixels=1, serial_pixels=1)
        settings = settings.modify_via_fit_type(
            is_parallel_fit=False, is_serial_fit=False
        )
        assert settings.parallel_pixels == 1
        assert settings.serial_pixels == 1

        settings = ac.ci.SettingsImagingCI(parallel_pixels=1, serial_pixels=1)
        settings = settings.modify_via_fit_type(
            is_parallel_fit=True, is_serial_fit=False
        )
        assert settings.parallel_pixels == 1
        assert settings.serial_pixels is None

        settings = ac.ci.SettingsImagingCI(parallel_pixels=1, serial_pixels=1)
        settings = settings.modify_via_fit_type(
            is_parallel_fit=False, is_serial_fit=True
        )
        assert settings.parallel_pixels is None
        assert settings.serial_pixels == 1

        settings = ac.ci.SettingsImagingCI(parallel_pixels=1, serial_pixels=1)
        settings = settings.modify_via_fit_type(
            is_parallel_fit=True, is_serial_fit=True
        )
        assert settings.parallel_pixels is None
        assert settings.serial_pixels is None


class TestImagingCI:
    def test__serial_calibration_imaging_ci_from_rows(self, imaging_ci_7x7):

        # The ci layout spans 2 rows, so two rows are extracted

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
            serial_calibration_imaging.pre_cti_data.native
            == imaging_ci_7x7.pre_cti_data.native[0:2, :]
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

    def test__from_fits__noise_map_from_single_value(self, layout_ci_7x7):

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

    def test__output_to_fits___all_arrays(self, layout_ci_7x7):

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

        assert (masked_imaging_ci.pre_cti_data == imaging_ci_7x7.pre_cti_data).all()

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

    def test__serial_masked_imaging_ci(
        self, imaging_ci_7x7, mask_2d_7x7_unmasked, ci_noise_scaling_map_list_7x7
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


class TestSimulatorImagingCI:
    def test__pre_cti_data_from(self):

        simulator = ac.ci.SimulatorImagingCI(normalization=30.0, pixel_scales=1.0)

        layout = ac.ci.Layout2DCI(
            shape_2d=(4, 3), region_list=[(0, 3, 0, 2), (2, 3, 2, 3)]
        )

        pre_cti_data = simulator.pre_cti_data_uniform_from(layout=layout)

        assert (
            pre_cti_data.native
            == np.array(
                [
                    [30.0, 30.0, 0.0],
                    [30.0, 30.0, 0.0],
                    [30.0, 30.0, 30.0],
                    [0.0, 0.0, 0.0],
                ]
            )
        ).all()

    def test__region_ci_from(self):

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0,
            normalization=100.0,
            row_slope=0.0,
            column_sigma=0.0,
            ci_seed=1,
        )

        layout = ac.ci.Layout2DCI(shape_2d=(5, 3), region_list=[(0, 1, 0, 1)])

        region = simulator.region_ci_from(region_dimensions=(3, 3))

        assert (
            region
            == np.array(
                [[100.0, 100.0, 100.0], [100.0, 100.0, 100.0], [100.0, 100.0, 100.0]]
            )
        ).all()

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0,
            normalization=100.0,
            row_slope=0.0,
            column_sigma=1.0,
            ci_seed=1,
        )

        layout = ac.ci.Layout2DCI(shape_2d=(5, 3), region_list=[(0, 1, 0, 1)])

        region = simulator.region_ci_from(region_dimensions=(3, 3))

        region = np.round(region, 1)

        assert (
            region
            == np.array([[101.6, 99.4, 99.5], [101.6, 99.4, 99.5], [101.6, 99.4, 99.5]])
        ).all()

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0,
            normalization=100.0,
            row_slope=-0.01,
            column_sigma=0.0,
            ci_seed=1,
        )

        layout = ac.ci.Layout2DCI(shape_2d=(5, 3), region_list=[(0, 1, 0, 1)])

        region = simulator.region_ci_from(region_dimensions=(3, 3))

        region = np.round(region, 1)

        assert (
            region
            == np.array([[100.0, 100.0, 100.0], [99.3, 99.3, 99.3], [98.9, 98.9, 98.9]])
        ).all()

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0,
            normalization=100.0,
            row_slope=-0.01,
            column_sigma=1.0,
            ci_seed=1,
        )

        layout = ac.ci.Layout2DCI(shape_2d=(5, 3), region_list=[(0, 1, 0, 1)])

        region = simulator.region_ci_from(region_dimensions=(3, 3))

        region = np.round(region, 1)

        assert (
            region
            == np.array([[101.6, 99.4, 99.5], [100.9, 98.7, 98.8], [100.5, 98.3, 98.4]])
        ).all()

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0,
            normalization=100.0,
            row_slope=0.0,
            column_sigma=100.0,
            ci_seed=1,
        )

        layout = ac.ci.Layout2DCI(shape_2d=(5, 3), region_list=[(0, 1, 0, 1)])

        region = simulator.region_ci_from(region_dimensions=(10, 10))

        assert (region > 0).all()

    def test__pre_cti_data_from__compare_uniform_to_non_uniform(self):

        simulator = ac.ci.SimulatorImagingCI(pixel_scales=1.0, normalization=10.0)

        layout = ac.ci.Layout2DCI(shape_2d=(5, 5), region_list=[(2, 4, 0, 5)])

        pre_cti_data_0 = simulator.pre_cti_data_uniform_from(layout=layout)

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0, normalization=10.0, row_slope=0.0, column_sigma=0.0
        )

        pre_cti_data_1 = simulator.pre_cti_data_non_uniform_from(layout=layout)

        assert (pre_cti_data_0 == pre_cti_data_1).all()

    def test__pre_cti_data_from__non_uniformity_in_columns(self):

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0,
            normalization=100.0,
            row_slope=0.0,
            column_sigma=1.0,
            ci_seed=1,
        )

        layout = ac.ci.Layout2DCI(shape_2d=(5, 5), region_list=[(0, 3, 0, 3)])

        image = simulator.pre_cti_data_non_uniform_from(layout=layout)

        image[:] = np.round(image[:], 1)

        assert (
            image.native
            == np.array(
                [
                    [101.6, 99.4, 99.5, 0.0, 0.0],
                    [101.6, 99.4, 99.5, 0.0, 0.0],
                    [101.6, 99.4, 99.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0,
            normalization=100.0,
            row_slope=0.0,
            column_sigma=1.0,
            ci_seed=1,
        )

        layout = ac.ci.Layout2DCI(
            shape_2d=(5, 5), region_list=[(1, 4, 1, 3), (1, 4, 4, 5)]
        )

        image = simulator.pre_cti_data_non_uniform_from(layout=layout)

        image[:] = np.round(image[:], 1)

        assert (
            image.native
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 101.6, 99.4, 0.0, 101.6],
                    [0.0, 101.6, 99.4, 0.0, 101.6],
                    [0.0, 101.6, 99.4, 0.0, 101.6],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0,
            normalization=100.0,
            row_slope=0.0,
            column_sigma=100.0,
            max_normalization=100.0,
            ci_seed=1,
        )

        layout = ac.ci.Layout2DCI(shape_2d=(5, 5), region_list=[(0, 5, 0, 5)])

        image = simulator.pre_cti_data_non_uniform_from(layout=layout)

        image = np.round(image, 1)

        # Checked ci_seed to ensure the max is above 100.0 without a max_normalization
        assert np.max(image) < 100.0

    def test__pre_cti_data_from__non_uniformity_in_rows(self):

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0, normalization=100.0, row_slope=-0.01, column_sigma=0.0
        )

        layout = ac.ci.Layout2DCI(shape_2d=(5, 5), region_list=[(0, 3, 0, 3)])

        image = simulator.pre_cti_data_non_uniform_from(layout=layout)

        image[:] = np.round(image[:], 1)

        assert (
            image.native
            == np.array(
                [
                    [100.0, 100.0, 100.0, 0.0, 0.0],
                    [99.3, 99.3, 99.3, 0.0, 0.0],
                    [98.9, 98.9, 98.9, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0, normalization=100.0, row_slope=-0.01, column_sigma=0.0
        )

        layout = ac.ci.Layout2DCI(
            shape_2d=(5, 5), region_list=[(1, 5, 1, 4), (0, 5, 4, 5)]
        )

        image = simulator.pre_cti_data_non_uniform_from(layout=layout)

        image[:] = np.round(image[:], 1)

        assert (
            image.native
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 100.0],
                    [0.0, 100.0, 100.0, 100.0, 99.3],
                    [0.0, 99.3, 99.3, 99.3, 98.9],
                    [0.0, 98.9, 98.9, 98.9, 98.6],
                    [0.0, 98.6, 98.6, 98.6, 98.4],
                ]
            )
        ).all()

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0,
            normalization=100.0,
            row_slope=-0.01,
            column_sigma=1.0,
            ci_seed=1,
        )

        layout = ac.ci.Layout2DCI(
            shape_2d=(5, 5), region_list=[(1, 5, 1, 4), (0, 5, 4, 5)]
        )

        image = simulator.pre_cti_data_non_uniform_from(layout=layout)

        image[:] = np.round(image[:], 1)

        assert (
            image.native
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 101.6],
                    [0.0, 101.6, 99.4, 99.5, 100.9],
                    [0.0, 100.9, 98.7, 98.8, 100.5],
                    [0.0, 100.5, 98.3, 98.4, 100.2],
                    [0.0, 100.2, 98.0, 98.1, 100.0],
                ]
            )
        ).all()

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0,
            normalization=100.0,
            row_slope=-0.01,
            column_sigma=1.0,
            read_noise=0.0,
            ci_seed=1,
        )

        layout = ac.ci.Layout2DCI(
            shape_2d=(5, 5), region_list=[(0, 2, 0, 5), (3, 5, 0, 5)]
        )

        image = simulator.pre_cti_data_non_uniform_from(layout=layout)

        image[:] = np.round(image[:], 1)

        print(image.native)

        assert (image.native[0:2, 0:5] == image.native[3:5, 0:5]).all()

    def test__no_instrumental_effects_input__only_cti_simulated(
        self, parallel_clocker_2d, traps_x2, ccd
    ):

        layout = ac.ci.Layout2DCI(
            shape_2d=(5, 5),
            region_list=[(0, 1, 0, 5)],
            serial_overscan=ac.Region2D((1, 2, 1, 2)),
        )

        simulator = ac.ci.SimulatorImagingCI(normalization=10.0, pixel_scales=1.0)

        imaging = simulator.from_layout(
            layout=layout,
            clocker=parallel_clocker_2d,
            parallel_trap_list=traps_x2,
            parallel_ccd=ccd,
        )

        assert imaging.image[0, 0:5] == pytest.approx(
            np.array([9.43, 9.43, 9.43, 9.43, 9.43]), 1e-1
        )
        assert imaging.layout == layout

    def test__include_read_noise__is_added_after_cti(
        self, parallel_clocker_2d, traps_x2, ccd
    ):

        layout = ac.ci.Layout2DCI(
            shape_2d=(3, 3),
            region_list=[(0, 1, 0, 3)],
            serial_overscan=ac.Region2D((1, 2, 1, 2)),
        )

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0, normalization=10.0, read_noise=1.0, noise_seed=1
        )

        imaging = simulator.from_layout(
            layout=layout,
            clocker=parallel_clocker_2d,
            parallel_trap_list=traps_x2,
            parallel_ccd=ccd,
        )

        image_no_noise = simulator.pre_cti_data_uniform_from(layout=layout)

        # Use seed to give us a known read noises map we'll test_autocti for

        assert imaging.image - image_no_noise.native == pytest.approx(
            np.array(
                [
                    [1.055, -1.180, -1.097],
                    [-0.780, 1.1574, -2.009],
                    [1.863, -0.642, 0.437],
                ]
            ),
            1e-2,
        )
        assert imaging.layout == layout

    def test__include_cosmics__is_added_to_image_and_trailed(
        self, parallel_clocker_2d, traps_x2, ccd
    ):

        layout = ac.ci.Layout2DCI(
            shape_2d=(5, 5),
            region_list=[(0, 1, 0, 5)],
            serial_overscan=ac.Region2D((1, 2, 1, 2)),
        )

        simulator = ac.ci.SimulatorImagingCI(pixel_scales=1.0, normalization=10.0)

        cosmic_ray_map = ac.Array2D.zeros(shape_native=(5, 5), pixel_scales=0.1).native
        cosmic_ray_map[2, 2] = 100.0

        imaging = simulator.from_layout(
            layout=layout,
            clocker=parallel_clocker_2d,
            parallel_trap_list=traps_x2,
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
        assert imaging.layout == layout

    def test__from_pre_cti_data(self, parallel_clocker_2d, traps_x2, ccd):

        layout = ac.ci.Layout2DCI(
            shape_2d=(5, 5),
            region_list=[(0, 1, 0, 5)],
            serial_overscan=ac.Region2D((1, 2, 1, 2)),
        )

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0, normalization=1.0, read_noise=4.0, noise_seed=1
        )

        cosmic_ray_map = ac.Array2D.zeros(shape_native=(5, 5), pixel_scales=0.1).native
        cosmic_ray_map[2, 2] = 100.0

        imaging = simulator.from_layout(
            layout=layout,
            clocker=parallel_clocker_2d,
            parallel_trap_list=traps_x2,
            parallel_ccd=ccd,
            cosmic_ray_map=cosmic_ray_map,
        )

        pre_cti_data = simulator.pre_cti_data_uniform_from(layout=layout)

        imaging_via_pre_cti_data = simulator.from_pre_cti_data(
            pre_cti_data=pre_cti_data.native,
            layout=layout,
            clocker=parallel_clocker_2d,
            parallel_trap_list=traps_x2,
            parallel_ccd=ccd,
            cosmic_ray_map=cosmic_ray_map,
        )

        assert (imaging.image == imaging_via_pre_cti_data.image).all()
        assert (imaging.noise_map == imaging_via_pre_cti_data.noise_map).all()
        assert (imaging.pre_cti_data == imaging_via_pre_cti_data.pre_cti_data).all()
        assert (imaging.cosmic_ray_map == imaging_via_pre_cti_data.cosmic_ray_map).all()

    def test__from_post_cti_data(self, parallel_clocker_2d, traps_x2, ccd):

        layout = ac.ci.Layout2DCI(
            shape_2d=(5, 5),
            region_list=[(0, 1, 0, 5)],
            serial_overscan=ac.Region2D((1, 2, 1, 2)),
        )

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0, normalization=1.0, read_noise=4.0, noise_seed=1
        )

        cosmic_ray_map = ac.Array2D.zeros(shape_native=(5, 5), pixel_scales=0.1).native
        cosmic_ray_map[2, 2] = 100.0

        imaging = simulator.from_layout(
            layout=layout,
            clocker=parallel_clocker_2d,
            parallel_trap_list=traps_x2,
            parallel_ccd=ccd,
            cosmic_ray_map=cosmic_ray_map,
        )

        pre_cti_data = simulator.pre_cti_data_uniform_from(layout=layout).native
        pre_cti_data += cosmic_ray_map

        post_cti_data = parallel_clocker_2d.add_cti(
            data=pre_cti_data, parallel_trap_list=traps_x2, parallel_ccd=ccd
        )

        pre_cti_data -= cosmic_ray_map

        imaging_via_post_cti_data = simulator.from_post_cti_data(
            post_cti_data=post_cti_data,
            pre_cti_data=pre_cti_data.native,
            layout=layout,
            cosmic_ray_map=cosmic_ray_map,
        )

        assert (imaging.image == imaging_via_post_cti_data.image).all()
        assert (imaging.noise_map == imaging_via_post_cti_data.noise_map).all()
        assert (imaging.pre_cti_data == imaging_via_post_cti_data.pre_cti_data).all()
        assert (
            imaging.cosmic_ray_map == imaging_via_post_cti_data.cosmic_ray_map
        ).all()
