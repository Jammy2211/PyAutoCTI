import os
from os import path
import shutil

import numpy as np
import pytest
import autocti as ac

test_data_path = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "arrays"
)


class TestSettingsCIImaging:
    def test__modify_via_fit_type(self):

        settings = ac.ci.SettingsCIImaging(parallel_columns=None, serial_rows=None)
        settings = settings.modify_via_fit_type(
            is_parallel_fit=False, is_serial_fit=False
        )
        assert settings.parallel_columns is None
        assert settings.serial_rows is None

        settings = ac.ci.SettingsCIImaging(parallel_columns=1, serial_rows=1)
        settings = settings.modify_via_fit_type(
            is_parallel_fit=False, is_serial_fit=False
        )
        assert settings.parallel_columns == 1
        assert settings.serial_rows == 1

        settings = ac.ci.SettingsCIImaging(parallel_columns=1, serial_rows=1)
        settings = settings.modify_via_fit_type(
            is_parallel_fit=True, is_serial_fit=False
        )
        assert settings.parallel_columns == 1
        assert settings.serial_rows is None

        settings = ac.ci.SettingsCIImaging(parallel_columns=1, serial_rows=1)
        settings = settings.modify_via_fit_type(
            is_parallel_fit=False, is_serial_fit=True
        )
        assert settings.parallel_columns is None
        assert settings.serial_rows == 1

        settings = ac.ci.SettingsCIImaging(parallel_columns=1, serial_rows=1)
        settings = settings.modify_via_fit_type(
            is_parallel_fit=True, is_serial_fit=True
        )
        assert settings.parallel_columns is None
        assert settings.serial_rows is None


class TestCIImaging:
    def test__parallel_calibration_ci_imaging_from_columns(self, ci_imaging_7x7):

        # The ci pattern starts at column 1, so the left most column is removed below

        parallel_calibration_imaging = ci_imaging_7x7.parallel_calibration_ci_imaging_for_columns(
            columns=(0, 6)
        )

        assert (
            parallel_calibration_imaging.image == ci_imaging_7x7.image[:, 1:7]
        ).all()
        assert (
            parallel_calibration_imaging.noise_map == ci_imaging_7x7.noise_map[:, 1:7]
        ).all()
        assert (
            parallel_calibration_imaging.pre_cti_ci == ci_imaging_7x7.pre_cti_ci[:, 1:7]
        ).all()
        assert (
            parallel_calibration_imaging.cosmic_ray_map
            == ci_imaging_7x7.cosmic_ray_map[:, 1:7]
        ).all()

    def test__serial_calibration_ci_imaging_from_rows(self, ci_imaging_7x7):

        # The ci pattern spans 2 rows, so two rows are extracted

        serial_calibration_imaging = ci_imaging_7x7.serial_calibration_ci_imaging_for_rows(
            rows=(0, 2)
        )

        assert (serial_calibration_imaging.image == ci_imaging_7x7.image[0:2, :]).all()
        assert (
            serial_calibration_imaging.noise_map == ci_imaging_7x7.noise_map[0:2, :]
        ).all()
        assert (
            serial_calibration_imaging.pre_cti_ci == ci_imaging_7x7.pre_cti_ci[0:2, :]
        ).all()
        assert (
            serial_calibration_imaging.cosmic_ray_map
            == ci_imaging_7x7.cosmic_ray_map[1:3, :]
        ).all()

    def test__signal_to_noise_map_and_max(self):

        image = ac.Array2D.ones(shape_native=(2, 2), pixel_scales=0.1).native
        image[0, 0] = 6.0

        imaging = ac.ci.CIImaging(
            image=image, noise_map=2.0 * np.ones((2, 2)), pre_cti_ci=None
        )

        assert (imaging.signal_to_noise_map == np.array([[3.0, 0.5], [0.5, 0.5]])).all()

        assert imaging.signal_to_noise_max == 3.0

    def test__from_fits__load_all_data_components__has_correct_attributes(
        self, ci_pattern_7x7
    ):

        imaging = ac.ci.CIImaging.from_fits(
            pixel_scales=1.0,
            roe_corner=(1, 0),
            scans=ac.Scans(
                parallel_overscan=(1, 2, 3, 4),
                serial_prescan=(5, 6, 7, 8),
                serial_overscan=(2, 4, 6, 8),
            ),
            ci_pattern=ci_pattern_7x7,
            image_path=path.join(test_data_path, "3x3_ones.fits"),
            image_hdu=0,
            noise_map_path=path.join(test_data_path, "3x3_twos.fits"),
            noise_map_hdu=0,
            pre_cti_ci_path=path.join(test_data_path, "3x3_threes.fits"),
            pre_cti_ci_hdu=0,
            cosmic_ray_map_path=path.join(test_data_path, "3x3_fours.fits"),
            cosmic_ray_map_hdu=0,
        )

        assert imaging.image.original_roe_corner == (1, 0)
        assert imaging.ci_pattern.regions == ci_pattern_7x7.regions
        assert imaging.image.scans.parallel_overscan == (1, 2, 3, 4)
        assert imaging.image.scans.serial_prescan == (5, 6, 7, 8)
        assert imaging.image.scans.serial_overscan == (2, 4, 6, 8)
        assert (imaging.image == np.ones((3, 3))).all()
        assert (imaging.noise_map == 2.0 * np.ones((3, 3))).all()
        assert (imaging.pre_cti_ci == 3.0 * np.ones((3, 3))).all()
        assert (imaging.cosmic_ray_map == 4.0 * np.ones((3, 3))).all()

    def test__from_fits__load_all_image_components__load_from_multi_hdu_fits(
        self, ci_pattern_7x7
    ):

        imaging = ac.ci.CIImaging.from_fits(
            pixel_scales=1.0,
            roe_corner=(1, 0),
            ci_pattern=ci_pattern_7x7,
            image_path=path.join(test_data_path, "3x3_multiple_hdu.fits"),
            image_hdu=0,
            noise_map_path=path.join(test_data_path, "3x3_multiple_hdu.fits"),
            noise_map_hdu=1,
            pre_cti_ci_path=path.join(test_data_path, "3x3_multiple_hdu.fits"),
            pre_cti_ci_hdu=2,
            cosmic_ray_map_path=path.join(test_data_path, "3x3_multiple_hdu.fits"),
            cosmic_ray_map_hdu=3,
        )

        assert imaging.image.original_roe_corner == (1, 0)
        assert imaging.ci_pattern.regions == ci_pattern_7x7.regions
        assert (imaging.image == np.ones((3, 3))).all()
        assert (imaging.noise_map == 2.0 * np.ones((3, 3))).all()
        assert (imaging.pre_cti_ci == 3.0 * np.ones((3, 3))).all()
        assert (imaging.cosmic_ray_map == 4.0 * np.ones((3, 3))).all()

    def test__from_fits__noise_map_from_single_value(self, ci_pattern_7x7):

        imaging = ac.ci.CIImaging.from_fits(
            pixel_scales=1.0,
            roe_corner=(1, 0),
            ci_pattern=ci_pattern_7x7,
            image_path=path.join(test_data_path, "3x3_ones.fits"),
            image_hdu=0,
            noise_map_from_single_value=10.0,
            pre_cti_ci_path=path.join(test_data_path, "3x3_threes.fits"),
            pre_cti_ci_hdu=0,
        )

        assert imaging.image.original_roe_corner == (1, 0)
        assert imaging.ci_pattern.regions == ci_pattern_7x7.regions
        assert (imaging.image == np.ones((3, 3))).all()
        assert (imaging.noise_map == 10.0 * np.ones((3, 3))).all()
        assert (imaging.pre_cti_ci == 3.0 * np.ones((3, 3))).all()
        assert imaging.cosmic_ray_map == None

    def test__from_fits__load_pre_cti_ci_image_from_the_pattern_and_image(self):

        pattern = ac.ci.CIPatternUniform(regions=[(0, 3, 0, 3)], normalization=10.0)

        imaging = ac.ci.CIImaging.from_fits(
            pixel_scales=1.0,
            roe_corner=(1, 0),
            ci_pattern=pattern,
            image_path=path.join(test_data_path, "3x3_ones.fits"),
            image_hdu=0,
            noise_map_path=path.join(test_data_path, "3x3_twos.fits"),
            noise_map_hdu=0,
        )

        assert imaging.image.original_roe_corner == (1, 0)
        assert imaging.ci_pattern.regions == pattern.regions
        assert (imaging.image == np.ones((3, 3))).all()
        assert (imaging.noise_map == 2.0 * np.ones((3, 3))).all()
        assert (imaging.pre_cti_ci == 10.0 * np.ones((3, 3))).all()
        assert imaging.cosmic_ray_map == None

    def test__output_to_fits___all_arrays(self, ci_pattern_7x7):

        imaging = ac.ci.CIImaging.from_fits(
            pixel_scales=1.0,
            roe_corner=(1, 0),
            ci_pattern=ci_pattern_7x7,
            image_path=path.join(test_data_path, "3x3_ones.fits"),
            image_hdu=0,
            noise_map_path=path.join(test_data_path, "3x3_twos.fits"),
            noise_map_hdu=0,
            pre_cti_ci_path=path.join(test_data_path, "3x3_threes.fits"),
            pre_cti_ci_hdu=0,
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
            pre_cti_ci_path=path.join(output_data_dir, "pre_cti_ci.fits"),
            cosmic_ray_map_path=path.join(output_data_dir, "cosmic_ray_map.fits"),
        )

        imaging = ac.ci.CIImaging.from_fits(
            pixel_scales=1.0,
            roe_corner=(1, 0),
            ci_pattern=ci_pattern_7x7,
            image_path=path.join(output_data_dir, "image.fits"),
            image_hdu=0,
            noise_map_path=path.join(output_data_dir, "noise_map.fits"),
            noise_map_hdu=0,
            pre_cti_ci_path=path.join(output_data_dir, "pre_cti_ci.fits"),
            pre_cti_ci_hdu=0,
            cosmic_ray_map_path=path.join(output_data_dir, "cosmic_ray_map.fits"),
            cosmic_ray_map_hdu=0,
        )

        assert (imaging.image == np.ones((3, 3))).all()
        assert (imaging.noise_map == 2.0 * np.ones((3, 3))).all()
        assert (imaging.pre_cti_ci == 3.0 * np.ones((3, 3))).all()
        assert (imaging.cosmic_ray_map == 4.0 * np.ones((3, 3))).all()


class TestApplyMask:
    def test__construtor__masks_arrays_correctly(self, ci_imaging_7x7):

        mask = ac.Mask2D.unmasked(
            shape_native=ci_imaging_7x7.shape_native, pixel_scales=1.0
        )

        mask[0, 0] = True

        masked_ci_imaging = ci_imaging_7x7.apply_mask(mask=mask)

        assert (masked_ci_imaging.mask == mask).all()

        masked_image = ci_imaging_7x7.image
        masked_image[0, 0] = 0.0

        assert (masked_ci_imaging.image == masked_image).all()

        masked_noise_map = ci_imaging_7x7.noise_map
        masked_noise_map[0, 0] = 0.0

        assert (masked_ci_imaging.noise_map == masked_noise_map).all()

        assert (masked_ci_imaging.pre_cti_ci == ci_imaging_7x7.pre_cti_ci).all()

        masked_cosmic_ray_map = ci_imaging_7x7.cosmic_ray_map
        masked_cosmic_ray_map[0, 0] = 0.0

        assert (masked_ci_imaging.cosmic_ray_map == masked_cosmic_ray_map).all()

    def test__include_parallel_columns_extraction(
        self, ci_imaging_7x7, mask_7x7_unmasked, ci_noise_scaling_maps_7x7
    ):

        mask = ac.Mask2D.unmasked(
            shape_native=ci_imaging_7x7.shape_native, pixel_scales=1.0
        )
        mask[0, 2] = True

        masked_ci_imaging = ci_imaging_7x7.apply_mask(mask=mask)
        masked_ci_imaging = masked_ci_imaging.apply_settings(
            settings=ac.ci.SettingsCIImaging(parallel_columns=(1, 3))
        )

        mask = ac.Mask2D.unmasked(shape_native=(7, 2), pixel_scales=1.0)
        mask[0, 0] = True

        assert (masked_ci_imaging.mask == mask).all()

        image = np.ones((7, 2))
        image[0, 0] = 0.0

        assert masked_ci_imaging.image == pytest.approx(image, 1.0e-4)

        noise_map = 2.0 * np.ones((7, 2))
        noise_map[0, 0] = 0.0

        assert masked_ci_imaging.noise_map == pytest.approx(noise_map, 1.0e-4)

        pre_cti_ci = 10.0 * np.ones((7, 2))

        assert masked_ci_imaging.pre_cti_ci == pytest.approx(pre_cti_ci, 1.0e-4)

        assert masked_ci_imaging.cosmic_ray_map.shape == (7, 2)

        noise_scaling_map_0 = np.ones((7, 2))
        noise_scaling_map_0[0, 0] = 0.0

        assert masked_ci_imaging.noise_scaling_maps[0] == pytest.approx(
            noise_scaling_map_0, 1.0e-4
        )

        noise_scaling_map_1 = 2.0 * np.ones((7, 2))
        noise_scaling_map_1[0, 0] = 0.0

        assert masked_ci_imaging.noise_scaling_maps[1] == pytest.approx(
            noise_scaling_map_1, 1.0e-4
        )

    def test__serial_masked_ci_imaging(
        self, ci_imaging_7x7, mask_7x7_unmasked, ci_noise_scaling_maps_7x7
    ):

        mask = ac.Mask2D.unmasked(
            shape_native=ci_imaging_7x7.shape_native, pixel_scales=1.0
        )
        mask[1, 0] = True

        masked_ci_imaging = ci_imaging_7x7.apply_mask(mask=mask)
        masked_ci_imaging = masked_ci_imaging.apply_settings(
            settings=ac.ci.SettingsCIImaging(serial_rows=(0, 1))
        )

        mask = ac.Mask2D.unmasked(shape_native=(1, 7), pixel_scales=1.0)
        mask[0, 0] = True

        assert (masked_ci_imaging.mask == mask).all()

        image = np.ones((1, 7))
        image[0, 0] = 0.0

        assert masked_ci_imaging.image == pytest.approx(image, 1.0e-4)

        noise_map = 2.0 * np.ones((1, 7))
        noise_map[0, 0] = 0.0

        assert masked_ci_imaging.noise_map == pytest.approx(noise_map, 1.0e-4)

        pre_cti_ci = 10.0 * np.ones((1, 7))

        assert masked_ci_imaging.pre_cti_ci == pytest.approx(pre_cti_ci, 1.0e-4)

        assert masked_ci_imaging.cosmic_ray_map.shape == (1, 7)

        noise_scaling_map_0 = np.ones((1, 7))
        noise_scaling_map_0[0, 0] = 0.0

        assert masked_ci_imaging.noise_scaling_maps[0] == pytest.approx(
            noise_scaling_map_0, 1.0e-4
        )

        noise_scaling_map_1 = 2.0 * np.ones((1, 7))
        noise_scaling_map_1[0, 0] = 0.0

        assert masked_ci_imaging.noise_scaling_maps[1] == pytest.approx(
            noise_scaling_map_1, 1.0e-4
        )


class TestSimulatorCIImaging(object):
    def test__no_instrumental_effects_input__only_cti_simulated(
        self, parallel_clocker, traps_x2, ccd
    ):

        pattern = ac.ci.CIPatternUniform(normalization=10.0, regions=[(0, 1, 0, 5)])

        simulator = ac.ci.SimulatorCIImaging(
            shape_native=(5, 5),
            pixel_scales=1.0,
            add_poisson_noise=False,
            scans=ac.Scans(serial_overscan=ac.Region2D((1, 2, 1, 2))),
        )

        imaging = simulator.from_ci_pattern(
            ci_pattern=pattern,
            clocker=parallel_clocker,
            parallel_traps=traps_x2,
            parallel_ccd=ccd,
        )

        assert imaging.image[0, 0:5] == pytest.approx(
            np.array([9.43, 9.43, 9.43, 9.43, 9.43]), 1e-1
        )
        assert imaging.image.scans.serial_overscan == (1, 2, 1, 2)
        assert imaging.pre_cti_ci.scans.serial_overscan == (1, 2, 1, 2)

    def test__include_read_noise__is_added_after_cti(
        self, parallel_clocker, traps_x2, ccd
    ):

        pattern = ac.ci.CIPatternUniform(normalization=10.0, regions=[(0, 1, 0, 3)])

        simulator = ac.ci.SimulatorCIImaging(
            shape_native=(3, 3),
            pixel_scales=1.0,
            scans=ac.Scans(serial_overscan=ac.Region2D((1, 2, 1, 2))),
            read_noise=1.0,
            add_poisson_noise=True,
            noise_seed=1,
        )

        imaging = simulator.from_ci_pattern(
            ci_pattern=pattern, clocker=parallel_clocker
        )

        image_no_noise = pattern.pre_cti_ci_from(shape_native=(3, 3), pixel_scales=1.0)

        # Use seed to give us a known read noises map we'll test_autocti for

        assert imaging.image - image_no_noise == pytest.approx(
            np.array([[1.62, -0.61, -0.53], [-1.07, 0.87, -2.30], [1.74, -0.76, 0.32]]),
            1e-2,
        )
        assert imaging.noise_map.scans.serial_overscan == (1, 2, 1, 2)

    def test__include_cosmics__is_added_to_image_and_trailed(
        self, parallel_clocker, traps_x2, ccd
    ):

        pattern = ac.ci.CIPatternUniform(normalization=10.0, regions=[(0, 1, 0, 5)])

        simulator = ac.ci.SimulatorCIImaging(
            shape_native=(5, 5),
            pixel_scales=1.0,
            scans=ac.Scans(serial_overscan=ac.Region2D((1, 2, 1, 2))),
            add_poisson_noise=False,
        )

        cosmic_ray_map = np.zeros((5, 5))
        cosmic_ray_map[2, 2] = 100.0

        imaging = simulator.from_ci_pattern(
            ci_pattern=pattern,
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
        assert imaging.cosmic_ray_map.scans.serial_overscan == (1, 2, 1, 2)

    def test__from_pre_cti_ci(self, parallel_clocker, traps_x2, ccd):

        pattern = ac.ci.CIPatternUniform(normalization=10.0, regions=[(0, 1, 0, 5)])

        simulator = ac.ci.SimulatorCIImaging(
            shape_native=(5, 5),
            pixel_scales=1.0,
            read_noise=4.0,
            scans=ac.Scans(serial_overscan=ac.Region2D((1, 2, 1, 2))),
            add_poisson_noise=False,
            noise_seed=1,
        )

        cosmic_ray_map = np.zeros((5, 5))
        cosmic_ray_map[2, 2] = 100.0

        imaging = simulator.from_ci_pattern(
            ci_pattern=pattern,
            clocker=parallel_clocker,
            parallel_traps=traps_x2,
            parallel_ccd=ccd,
            cosmic_ray_map=cosmic_ray_map,
        )

        pre_cti_ci = pattern.pre_cti_ci_from(shape_native=(5, 5), pixel_scales=1.0)

        imaging_via_pre_cti_ci = simulator.from_pre_cti_ci(
            pre_cti_ci=pre_cti_ci,
            ci_pattern=pattern,
            clocker=parallel_clocker,
            parallel_traps=traps_x2,
            parallel_ccd=ccd,
            cosmic_ray_map=cosmic_ray_map,
        )

        assert (imaging.image == imaging_via_pre_cti_ci.image).all()
        assert (imaging.noise_map == imaging_via_pre_cti_ci.noise_map).all()
        assert (imaging.pre_cti_ci == imaging_via_pre_cti_ci.pre_cti_ci).all()
        assert (imaging.cosmic_ray_map == imaging_via_pre_cti_ci.cosmic_ray_map).all()

    def test__from_post_cti_ci(self, parallel_clocker, traps_x2, ccd):

        pattern = ac.ci.CIPatternUniform(normalization=10.0, regions=[(0, 1, 0, 5)])

        simulator = ac.ci.SimulatorCIImaging(
            shape_native=(5, 5),
            pixel_scales=1.0,
            read_noise=4.0,
            scans=ac.Scans(serial_overscan=ac.Region2D((1, 2, 1, 2))),
            add_poisson_noise=False,
            noise_seed=1,
        )

        cosmic_ray_map = np.zeros((5, 5))
        cosmic_ray_map[2, 2] = 100.0

        imaging = simulator.from_ci_pattern(
            ci_pattern=pattern,
            clocker=parallel_clocker,
            parallel_traps=traps_x2,
            parallel_ccd=ccd,
            cosmic_ray_map=cosmic_ray_map,
        )

        pre_cti_ci = pattern.pre_cti_ci_from(shape_native=(5, 5), pixel_scales=1.0)
        pre_cti_ci += cosmic_ray_map

        post_cti_ci = parallel_clocker.add_cti(
            image=pre_cti_ci, parallel_traps=traps_x2, parallel_ccd=ccd
        )

        imaging_via_post_cti_ci = simulator.from_post_cti_ci(
            post_cti_ci=post_cti_ci,
            pre_cti_ci=pre_cti_ci,
            ci_pattern=pattern,
            cosmic_ray_map=cosmic_ray_map,
        )

        assert (imaging.image == imaging_via_post_cti_ci.image).all()
        assert (imaging.noise_map == imaging_via_post_cti_ci.noise_map).all()
        assert (imaging.pre_cti_ci == imaging_via_post_cti_ci.pre_cti_ci).all()
        assert (imaging.cosmic_ray_map == imaging_via_post_cti_ci.cosmic_ray_map).all()
