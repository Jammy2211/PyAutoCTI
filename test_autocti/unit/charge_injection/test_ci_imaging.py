import os

import numpy as np
import pytest
import shutil

import autocti as ac
from autocti.masked import masked_dataset

test_data_dir = "{}/../test_files/arrays/".format(
    os.path.dirname(os.path.realpath(__file__))
)


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
            rows=(0, 6)
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

        ci_imaging = ac.ci_imaging(
            image=image, noise_map=2.0 * np.ones((2, 2)), ci_pre_cti=None
        )

        assert (
            ci_imaging.signal_to_noise_map == np.array([[3.0, 0.5], [0.5, 0.5]])
        ).all()

        assert ci_imaging.signal_to_noise_max == 3.0


class TestCIDataSimulate(object):
    def test__no_instrumental_effects_input__only_cti_simulated(
        self, arctic_parallel, params_parallel
    ):

        pattern = ac.CIPatternUniform(normalization=10.0, regions=[(0, 1, 0, 5)])

        ci_pre_cti = pattern.simulate_ci_pre_cti(shape=(5, 5))

        ci_data_simulate = ac.ci_imaging.simulate(
            ci_pre_cti=ci_pre_cti,
            frame_geometry=ac.FrameGeometry.bottom_left(),
            ci_pattern=pattern,
            cti_settings=arctic_parallel,
            cti_params=params_parallel,
        )

        assert ci_data_simulate.image[0, 0:5] == pytest.approx(
            np.array([10.0, 10.0, 10.0, 10.0, 10.0]), 1e-2
        )

    def test__include_read_noise__is_added_after_cti(
        self, arctic_parallel, params_parallel
    ):

        pattern = ac.CIPatternUniform(normalization=10.0, regions=[(0, 1, 0, 3)])

        ci_pre_cti = pattern.simulate_ci_pre_cti(shape=(3, 3))

        ci_data_simulate = ac.ci_imaging.simulate(
            ci_pre_cti=ci_pre_cti,
            frame_geometry=ac.FrameGeometry.bottom_left(),
            ci_pattern=pattern,
            cti_settings=arctic_parallel,
            cti_params=params_parallel,
            read_noise=1.0,
            noise_seed=1,
        )

        image_no_noise = pattern.ci_pre_cti_from_shape(shape=(3, 3))

        # Use seed to give us a known read noises map we'll test_autoarray for

        assert ci_data_simulate.image - image_no_noise == pytest.approx(
            np.array([[1.62, -0.61, -0.53], [-1.07, 0.87, -2.30], [1.74, -0.76, 0.32]]),
            1e-2,
        )

    def test__include_cosmics__is_added_to_image_and_trailed(
        self, arctic_parallel, params_parallel
    ):

        pattern = ac.CIPatternUniform(normalization=10.0, regions=[(0, 1, 0, 5)])

        ci_pre_cti = pattern.simulate_ci_pre_cti(shape=(5, 5))

        cosmic_ray_map = np.zeros((5, 5))
        cosmic_ray_map[2, 2] = 100.0

        ci_data_simulate = ac.ci_imaging.simulate(
            ci_pre_cti=ci_pre_cti,
            frame_geometry=ac.FrameGeometry.bottom_left(),
            ci_pattern=pattern,
            cti_settings=arctic_parallel,
            cti_params=params_parallel,
            cosmic_ray_map=cosmic_ray_map,
        )

        assert ci_data_simulate.image[0, 0:5] == pytest.approx(
            np.array([10.0, 10.0, 10.0, 10.0, 10.0]), 1e-2
        )
        assert 0.0 < ci_data_simulate.image[1, 1] < 100.0
        assert ci_data_simulate.image[2, 2] > 98.0
        assert (ci_data_simulate.image[1, 1:4] > 0.0).all()
        assert (
            ci_data_simulate.cosmic_ray_map
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

    def test__include_parallel_poisson_trap_densities(self, arctic_parallel):

        pattern = ac.CIPatternUniform(normalization=10.0, regions=[(2, 3, 0, 5)])

        ci_pre_cti = pattern.simulate_ci_pre_cti(shape=(5, 5))

        # Densities for this seed are [9.6, 8.2, 8.6, 9.6, 9.6]

        parallel_traps = ac.Trap(trap_density=10.0, trap_lifetime=1.0)
        parallel_traps = ac.Trap.poisson_species(
            species=[parallel_traps], shape=(5, 5), seed=1
        )
        parallel_ccd_volume = ac.CCDVolume(
            well_notch_depth=1.0e-4,
            well_fill_beta=0.58,
            well_fill_gamma=0.0,
            well_fill_alpha=1.0,
        )

        params_parallel = ac.ArcticParams(
            parallel_traps=parallel_traps, parallel_ccd_volume=parallel_ccd_volume
        )

        ci_data_simulate = ac.ci_imaging.simulate(
            ci_pre_cti=ci_pre_cti,
            frame_geometry=ac.FrameGeometry.bottom_left(),
            ci_pattern=pattern,
            cti_settings=arctic_parallel,
            cti_params=params_parallel,
            use_parallel_poisson_densities=True,
        )

        assert ci_data_simulate.image[2, 0] == ci_data_simulate.image[2, 3]
        assert ci_data_simulate.image[2, 0] == ci_data_simulate.image[2, 4]
        assert ci_data_simulate.image[2, 0] < ci_data_simulate.image[2, 1]
        assert ci_data_simulate.image[2, 0] < ci_data_simulate.image[2, 2]
        assert ci_data_simulate.image[2, 1] > ci_data_simulate.image[2, 2]


class TestCIImagingFromFits(object):
    def test__load_all_data_components__has_correct_attributes(self, ci_pattern_7x7):

        ci_imaging = ac.ci_imaging.from_fits(
            roe_corner=(1, 0),
            parallel_overscan=(1, 2, 3, 4),
            serial_prescan=(5, 6, 7, 8),
            serial_overscan=(2, 4, 6, 8),
            ci_pattern=ci_pattern_7x7,
            image_path=test_data_dir + "3x3_ones.fits",
            image_hdu=0,
            noise_map_path=test_data_dir + "3x3_twos.fits",
            noise_map_hdu=0,
            ci_pre_cti_path=test_data_dir + "3x3_threes.fits",
            ci_pre_cti_hdu=0,
            cosmic_ray_map_path=test_data_dir + "3x3_fours.fits",
            cosmic_ray_map_hdu=0,
        )

        assert ci_imaging.image.original_roe_corner == (1, 0)
        assert ci_imaging.ci_pattern.regions == ci_pattern_7x7.regions
        assert ci_imaging.image.parallel_overscan == (1, 2, 3, 4)
        assert ci_imaging.image.serial_prescan == (5, 6, 7, 8)
        assert ci_imaging.image.serial_overscan == (2, 4, 6, 8)
        assert (ci_imaging.image == np.ones((3, 3))).all()
        assert (ci_imaging.noise_map == 2.0 * np.ones((3, 3))).all()
        assert (ci_imaging.ci_pre_cti == 3.0 * np.ones((3, 3))).all()
        assert (ci_imaging.cosmic_ray_map == 4.0 * np.ones((3, 3))).all()

    def test__load_all_image_components__load_from_multi_hdu_fits(self, ci_pattern_7x7):

        ci_imaging = ac.ci_imaging.from_fits(
            roe_corner=(1, 0),
            ci_pattern=ci_pattern_7x7,
            image_path=test_data_dir + "3x3_multiple_hdu.fits",
            image_hdu=0,
            noise_map_path=test_data_dir + "3x3_multiple_hdu.fits",
            noise_map_hdu=1,
            ci_pre_cti_path=test_data_dir + "3x3_multiple_hdu.fits",
            ci_pre_cti_hdu=2,
            cosmic_ray_map_path=test_data_dir + "3x3_multiple_hdu.fits",
            cosmic_ray_map_hdu=3,
        )

        assert ci_imaging.image.original_roe_corner == (1, 0)
        assert ci_imaging.ci_pattern.regions == ci_pattern_7x7.regions
        assert (ci_imaging.image == np.ones((3, 3))).all()
        assert (ci_imaging.noise_map == 2.0 * np.ones((3, 3))).all()
        assert (ci_imaging.ci_pre_cti == 3.0 * np.ones((3, 3))).all()
        assert (ci_imaging.cosmic_ray_map == 4.0 * np.ones((3, 3))).all()

    def test__load_noise_map_from_single_value(self, ci_pattern_7x7):

        ci_imaging = ac.ci_imaging.from_fits(
            roe_corner=(1, 0),
            ci_pattern=ci_pattern_7x7,
            image_path=test_data_dir + "3x3_ones.fits",
            image_hdu=0,
            noise_map_from_single_value=10.0,
            ci_pre_cti_path=test_data_dir + "3x3_threes.fits",
            ci_pre_cti_hdu=0,
        )

        assert ci_imaging.image.original_roe_corner == (1, 0)
        assert ci_imaging.ci_pattern.regions == ci_pattern_7x7.regions
        assert (ci_imaging.image == np.ones((3, 3))).all()
        assert (ci_imaging.noise_map == 10.0 * np.ones((3, 3))).all()
        assert (ci_imaging.ci_pre_cti == 3.0 * np.ones((3, 3))).all()
        assert ci_imaging.cosmic_ray_map == None

    def test__load_ci_pre_cti_image_from_the_pattern_and_image(self):

        ci_pattern = ac.CIPatternUniform(regions=[(0, 3, 0, 3)], normalization=10.0)

        ci_imaging = ac.ci_imaging.from_fits(
            roe_corner=(1, 0),
            ci_pattern=ci_pattern,
            image_path=test_data_dir + "3x3_ones.fits",
            image_hdu=0,
            noise_map_path=test_data_dir + "3x3_twos.fits",
            noise_map_hdu=0,
        )

        assert ci_imaging.image.original_roe_corner == (1, 0)
        assert ci_imaging.ci_pattern.regions == ci_pattern.regions
        assert (ci_imaging.image == np.ones((3, 3))).all()
        assert (ci_imaging.noise_map == 2.0 * np.ones((3, 3))).all()
        assert (ci_imaging.ci_pre_cti == 10.0 * np.ones((3, 3))).all()
        assert ci_imaging.cosmic_ray_map == None

    def test__output_all_arrays(self, ci_pattern_7x7):

        ci_imaging = ac.ci_imaging.from_fits(
            roe_corner=(1, 0),
            ci_pattern=ci_pattern_7x7,
            image_path=test_data_dir + "3x3_ones.fits",
            image_hdu=0,
            noise_map_path=test_data_dir + "3x3_twos.fits",
            noise_map_hdu=0,
            ci_pre_cti_path=test_data_dir + "3x3_threes.fits",
            ci_pre_cti_hdu=0,
            cosmic_ray_map_path=test_data_dir + "3x3_fours.fits",
            cosmic_ray_map_hdu=0,
        )

        output_data_dir = "{}/../test_files/arrays/output_test/".format(
            os.path.dirname(os.path.realpath(__file__))
        )
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        ac.output_ci_data_to_fits(
            ci_data=ci_imaging,
            image_path=output_data_dir + "image.fits",
            noise_map_path=output_data_dir + "noise_map.fits",
            ci_pre_cti_path=output_data_dir + "ci_pre_cti.fits",
            cosmic_ray_map_path=output_data_dir + "cosmic_ray_map.fits",
        )

        ci_imaging = ac.ci_imaging.from_fits(
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

        assert (ci_imaging.image == np.ones((3, 3))).all()
        assert (ci_imaging.noise_map == 2.0 * np.ones((3, 3))).all()
        assert (ci_imaging.ci_pre_cti == 3.0 * np.ones((3, 3))).all()
        assert (ci_imaging.cosmic_ray_map == 4.0 * np.ones((3, 3))).all()


class TestCIImage(object):
    class TestSetupCIPreCTI:
        def test__uniform_pattern_1_region_normalization_10__correct_pre_clock_image(
            self
        ):
            pattern = ac.CIPatternUniform(normalization=10.0, regions=[(0, 2, 0, 2)])
            image = 10.0 * np.ones((3, 3))

            ci_pre_cti = ac.ci_pre_cti_from_ci_pattern_geometry_image_and_mask(
                ci_pattern=pattern, image=image
            )

            assert (
                ci_pre_cti
                == np.array([[10.0, 10.0, 0.0], [10.0, 10.0, 0.0], [00.0, 00.0, 0.0]])
            ).all()

        def test__same_as_above_but_different_normalization_and_regions(self):
            pattern = ac.CIPatternUniform(
                normalization=20.0, regions=[(0, 2, 0, 1), (2, 3, 2, 3)]
            )
            image = 10.0 * np.ones((3, 3))

            ci_pre_cti = ac.ci_pre_cti_from_ci_pattern_geometry_image_and_mask(
                ci_pattern=pattern, image=image
            )

            assert (
                ci_pre_cti
                == np.array([[20.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 0.0, 20.0]])
            ).all()

        def test__non_uniform_pattern__image_is_same_as_computed_image(self):
            pattern = ac.CIPatternNonUniform(
                normalization=100.0,
                regions=[(0, 2, 0, 2), (2, 3, 0, 3)],
                row_slope=-1.0,
            )
            image = np.array([[10.0, 10.0, 10.0], [2.0, 2.0, 2.0], [8.0, 12.0, 10.0]])
            mask = ac.Mask.unmasked(shape_2d=(3, 3))

            ci_pre_cti = ac.ci_pre_cti_from_ci_pattern_geometry_image_and_mask(
                mask=mask, ci_pattern=pattern, image=image
            )

            pattern_ci_pre_cti = pattern.ci_pre_cti_from_ci_image_and_mask(image, mask)

            # noinspection PyUnresolvedReferences
            assert (ci_pre_cti == pattern_ci_pre_cti).all()

    class TestCreateReadNoiseMap(object):
        def test__read_noise_sigma_0__read_noise_map_all_0__image_is_identical_to_input(
            self
        ):
            simulate_read_noise = ac.read_noise_map_from_shape_and_sigma(
                shape=(3, 3), sigma=0.0, noise_seed=1
            )

            assert (simulate_read_noise == np.zeros((3, 3))).all()

        def test__read_noise_sigma_1__read_noise_map_all_non_0__image_has_noise_added(
            self
        ):
            simulate_read_noise = ac.read_noise_map_from_shape_and_sigma(
                shape=(3, 3), sigma=1.0, noise_seed=1
            )

            # Use seed to give us a known read noises map we'll test_autoarray for

            assert simulate_read_noise == pytest.approx(
                np.array(
                    [[1.62, -0.61, -0.53], [-1.07, 0.87, -2.30], [1.74, -0.76, 0.32]]
                ),
                1e-2,
            )
