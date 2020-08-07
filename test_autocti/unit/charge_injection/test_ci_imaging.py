import os
import shutil

import numpy as np
import pytest
import autocti as ac

test_data_path = "{}/files/arrays/".format(os.path.dirname(os.path.realpath(__file__)))


class TestCIImaging(object):
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
            parallel_calibration_imaging.ci_pre_cti == ci_imaging_7x7.ci_pre_cti[:, 1:7]
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
            serial_calibration_imaging.ci_pre_cti == ci_imaging_7x7.ci_pre_cti[0:2, :]
        ).all()
        assert (
            serial_calibration_imaging.cosmic_ray_map
            == ci_imaging_7x7.cosmic_ray_map[1:3, :]
        ).all()

    def test__signal_to_noise_map_and_max(self):
        image = np.ones((2, 2))
        image[0, 0] = 6.0

        imaging = ac.ci.CIImaging(
            image=image, noise_map=2.0 * np.ones((2, 2)), ci_pre_cti=None
        )

        assert (imaging.signal_to_noise_map == np.array([[3.0, 0.5], [0.5, 0.5]])).all()

        assert imaging.signal_to_noise_max == 3.0

    def test__from_fits__load_all_data_components__has_correct_attributes(
        self, ci_pattern_7x7
    ):

        imaging = ac.ci.CIImaging.from_fits(
            roe_corner=(1, 0),
            scans=ac.Scans(
                parallel_overscan=(1, 2, 3, 4),
                serial_prescan=(5, 6, 7, 8),
                serial_overscan=(2, 4, 6, 8),
            ),
            ci_pattern=ci_pattern_7x7,
            image_path=test_data_path + "3x3_ones.fits",
            image_hdu=0,
            noise_map_path=test_data_path + "3x3_twos.fits",
            noise_map_hdu=0,
            ci_pre_cti_path=test_data_path + "3x3_threes.fits",
            ci_pre_cti_hdu=0,
            cosmic_ray_map_path=test_data_path + "3x3_fours.fits",
            cosmic_ray_map_hdu=0,
        )

        assert imaging.image.original_roe_corner == (1, 0)
        assert imaging.ci_pattern.regions == ci_pattern_7x7.regions
        assert imaging.image.scans.parallel_overscan == (1, 2, 3, 4)
        assert imaging.image.scans.serial_prescan == (5, 6, 7, 8)
        assert imaging.image.scans.serial_overscan == (2, 4, 6, 8)
        assert (imaging.image == np.ones((3, 3))).all()
        assert (imaging.noise_map == 2.0 * np.ones((3, 3))).all()
        assert (imaging.ci_pre_cti == 3.0 * np.ones((3, 3))).all()
        assert (imaging.cosmic_ray_map == 4.0 * np.ones((3, 3))).all()

    def test__from_fits__load_all_image_components__load_from_multi_hdu_fits(
        self, ci_pattern_7x7
    ):

        imaging = ac.ci.CIImaging.from_fits(
            roe_corner=(1, 0),
            ci_pattern=ci_pattern_7x7,
            image_path=test_data_path + "3x3_multiple_hdu.fits",
            image_hdu=0,
            noise_map_path=test_data_path + "3x3_multiple_hdu.fits",
            noise_map_hdu=1,
            ci_pre_cti_path=test_data_path + "3x3_multiple_hdu.fits",
            ci_pre_cti_hdu=2,
            cosmic_ray_map_path=test_data_path + "3x3_multiple_hdu.fits",
            cosmic_ray_map_hdu=3,
        )

        assert imaging.image.original_roe_corner == (1, 0)
        assert imaging.ci_pattern.regions == ci_pattern_7x7.regions
        assert (imaging.image == np.ones((3, 3))).all()
        assert (imaging.noise_map == 2.0 * np.ones((3, 3))).all()
        assert (imaging.ci_pre_cti == 3.0 * np.ones((3, 3))).all()
        assert (imaging.cosmic_ray_map == 4.0 * np.ones((3, 3))).all()

    def test__from_fits__noise_map_from_single_value(self, ci_pattern_7x7):

        imaging = ac.ci.CIImaging.from_fits(
            roe_corner=(1, 0),
            ci_pattern=ci_pattern_7x7,
            image_path=test_data_path + "3x3_ones.fits",
            image_hdu=0,
            noise_map_from_single_value=10.0,
            ci_pre_cti_path=test_data_path + "3x3_threes.fits",
            ci_pre_cti_hdu=0,
        )

        assert imaging.image.original_roe_corner == (1, 0)
        assert imaging.ci_pattern.regions == ci_pattern_7x7.regions
        assert (imaging.image == np.ones((3, 3))).all()
        assert (imaging.noise_map == 10.0 * np.ones((3, 3))).all()
        assert (imaging.ci_pre_cti == 3.0 * np.ones((3, 3))).all()
        assert imaging.cosmic_ray_map == None

    def test__from_fits__load_ci_pre_cti_image_from_the_pattern_and_image(self):

        pattern = ac.ci.CIPatternUniform(regions=[(0, 3, 0, 3)], normalization=10.0)

        imaging = ac.ci.CIImaging.from_fits(
            roe_corner=(1, 0),
            ci_pattern=pattern,
            image_path=test_data_path + "3x3_ones.fits",
            image_hdu=0,
            noise_map_path=test_data_path + "3x3_twos.fits",
            noise_map_hdu=0,
        )

        assert imaging.image.original_roe_corner == (1, 0)
        assert imaging.ci_pattern.regions == pattern.regions
        assert (imaging.image == np.ones((3, 3))).all()
        assert (imaging.noise_map == 2.0 * np.ones((3, 3))).all()
        assert (imaging.ci_pre_cti == 10.0 * np.ones((3, 3))).all()
        assert imaging.cosmic_ray_map == None

    def test__output_to_fits___all_arrays(self, ci_pattern_7x7):

        imaging = ac.ci.CIImaging.from_fits(
            roe_corner=(1, 0),
            ci_pattern=ci_pattern_7x7,
            image_path=test_data_path + "3x3_ones.fits",
            image_hdu=0,
            noise_map_path=test_data_path + "3x3_twos.fits",
            noise_map_hdu=0,
            ci_pre_cti_path=test_data_path + "3x3_threes.fits",
            ci_pre_cti_hdu=0,
            cosmic_ray_map_path=test_data_path + "3x3_fours.fits",
            cosmic_ray_map_hdu=0,
        )

        output_data_dir = "{}/files/arrays/output_test/".format(
            os.path.dirname(os.path.realpath(__file__))
        )
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        imaging.output_to_fits(
            image_path=output_data_dir + "image.fits",
            noise_map_path=output_data_dir + "noise_map.fits",
            ci_pre_cti_path=output_data_dir + "ci_pre_cti.fits",
            cosmic_ray_map_path=output_data_dir + "cosmic_ray_map.fits",
        )

        imaging = ac.ci.CIImaging.from_fits(
            roe_corner=(1, 0),
            ci_pattern=ci_pattern_7x7,
            image_path=output_data_dir + "image.fits",
            image_hdu=0,
            noise_map_path=output_data_dir + "noise_map.fits",
            noise_map_hdu=0,
            ci_pre_cti_path=output_data_dir + "ci_pre_cti.fits",
            ci_pre_cti_hdu=0,
            cosmic_ray_map_path=output_data_dir + "cosmic_ray_map.fits",
            cosmic_ray_map_hdu=0,
        )

        assert (imaging.image == np.ones((3, 3))).all()
        assert (imaging.noise_map == 2.0 * np.ones((3, 3))).all()
        assert (imaging.ci_pre_cti == 3.0 * np.ones((3, 3))).all()
        assert (imaging.cosmic_ray_map == 4.0 * np.ones((3, 3))).all()


class TestMaskedCIImaging:
    def test__construtor__masks_arrays_correctly(self, ci_imaging_7x7):

        mask = ac.Mask.unmasked(shape_2d=ci_imaging_7x7.shape_2d)

        mask[0, 0] = True

        masked_ci_imaging = ac.ci.MaskedCIImaging(ci_imaging=ci_imaging_7x7, mask=mask)

        assert (masked_ci_imaging.mask == mask).all()

        masked_image = ci_imaging_7x7.image
        masked_image[0, 0] = 0.0

        assert (masked_ci_imaging.image == masked_image).all()

        masked_noise_map = ci_imaging_7x7.noise_map
        masked_noise_map[0, 0] = 0.0

        assert (masked_ci_imaging.noise_map == masked_noise_map).all()

        assert (masked_ci_imaging.ci_pre_cti == ci_imaging_7x7.ci_pre_cti).all()

        masked_cosmic_ray_map = ci_imaging_7x7.cosmic_ray_map
        masked_cosmic_ray_map[0, 0] = 0.0

        assert (masked_ci_imaging.cosmic_ray_map == masked_cosmic_ray_map).all()

    def test__include_parallel_columns_extraction(
        self, ci_imaging_7x7, mask_7x7, ci_noise_scaling_maps_7x7
    ):

        mask = ac.Mask.unmasked(shape_2d=ci_imaging_7x7.shape_2d)
        mask[0, 2] = True

        masked_ci_imaging = ac.ci.MaskedCIImaging(
            ci_imaging=ci_imaging_7x7,
            mask=mask,
            parallel_columns=(1, 3),
            noise_scaling_maps=ci_noise_scaling_maps_7x7,
        )

        mask = ac.Mask.unmasked(shape_2d=(7, 2))
        mask[0, 0] = True

        assert (masked_ci_imaging.mask == mask).all()

        image = np.ones((7, 2))
        image[0, 0] = 0.0

        assert masked_ci_imaging.image == pytest.approx(image, 1.0e-4)

        noise_map = 2.0 * np.ones((7, 2))
        noise_map[0, 0] = 0.0

        assert masked_ci_imaging.noise_map == pytest.approx(noise_map, 1.0e-4)

        ci_pre_cti = 10.0 * np.ones((7, 2))

        assert masked_ci_imaging.ci_pre_cti == pytest.approx(ci_pre_cti, 1.0e-4)

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
        self, ci_imaging_7x7, mask_7x7, ci_noise_scaling_maps_7x7
    ):

        mask = ac.Mask.unmasked(shape_2d=ci_imaging_7x7.shape_2d)
        mask[1, 0] = True

        masked_ci_imaging = ac.ci.MaskedCIImaging(
            ci_imaging=ci_imaging_7x7,
            mask=mask,
            serial_rows=(0, 1),
            noise_scaling_maps=ci_noise_scaling_maps_7x7,
        )

        mask = ac.Mask.unmasked(shape_2d=(1, 7))
        mask[0, 0] = True

        assert (masked_ci_imaging.mask == mask).all()

        image = np.ones((1, 7))
        image[0, 0] = 0.0

        assert masked_ci_imaging.image == pytest.approx(image, 1.0e-4)

        noise_map = 2.0 * np.ones((1, 7))
        noise_map[0, 0] = 0.0

        assert masked_ci_imaging.noise_map == pytest.approx(noise_map, 1.0e-4)

        ci_pre_cti = 10.0 * np.ones((1, 7))

        assert masked_ci_imaging.ci_pre_cti == pytest.approx(ci_pre_cti, 1.0e-4)

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
            shape_2d=(5, 5),
            add_noise=False,
            scans=ac.Scans(serial_overscan=ac.Region((1, 2, 1, 2))),
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
        assert imaging.ci_pre_cti.scans.serial_overscan == (1, 2, 1, 2)

    def test__include_read_noise__is_added_after_cti(
        self, parallel_clocker, traps_x2, ccd
    ):

        pattern = ac.ci.CIPatternUniform(normalization=10.0, regions=[(0, 1, 0, 3)])

        simulator = ac.ci.SimulatorCIImaging(
            shape_2d=(3, 3),
            scans=ac.Scans(serial_overscan=ac.Region((1, 2, 1, 2))),
            read_noise=1.0,
            add_noise=True,
            noise_seed=1,
        )

        imaging = simulator.from_ci_pattern(
            ci_pattern=pattern, clocker=parallel_clocker
        )

        image_no_noise = pattern.ci_pre_cti_from_shape_2d(shape_2d=(3, 3))

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
            shape_2d=(5, 5),
            scans=ac.Scans(serial_overscan=ac.Region((1, 2, 1, 2))),
            add_noise=False,
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

    # def test__include_parallel_poisson_trap_densities(self, arctic_parallel):
    #
    #     pattern = ac.ci.CIPatternUniform(normalization=10.0, regions=[(2, 3, 0, 5)])
    #
    #     ci_pre_cti = pattern.simulate_ci_pre_cti(shape=(5, 5))
    #
    #     # Densities for this seed are [9.6, 8.2, 8.6, 9.6, 9.6]
    #
    #     parallel_traps = ac.TrapInstantCapture(trap_density=10.0, trap_release_timescale=1.0)
    #     parallel_traps = ac.TrapInstantCapture.poisson_species(
    #         species=[parallel_traps], shape=(5, 5), seed=1
    #     )
    #     parallel_ccd = ac.CCD(
    #         well_notch_depth=1.0e-4,
    #         well_fill_power=0.58,
    #         well_fill_gamma=0.0,
    #         well_fill_alpha=1.0,
    #     )
    #
    #     params_parallel = ac.ArcticParams(
    #         parallel_traps=parallel_traps, parallel_ccd=parallel_ccd
    #     )
    #
    #     imaging = ac.ci.CIImaging.simulate(
    #         ci_pre_cti=ci_pre_cti,
    #         frame_geometry=ac.FrameGeometry.bottom_left(),
    #         ci_pattern=pattern,
    #         clocker=arctic_parallel,
    #         cti_params=params_parallel,
    #         use_parallel_poisson_densities=True,
    #     )
    #
    #     assert imaging.image[2, 0] == imaging.image[2, 3]
    #     assert imaging.image[2, 0] == imaging.image[2, 4]
    #     assert imaging.image[2, 0] < imaging.image[2, 1]
    #     assert imaging.image[2, 0] < imaging.image[2, 2]
    #     assert imaging.image[2, 1] > imaging.image[2, 2]
