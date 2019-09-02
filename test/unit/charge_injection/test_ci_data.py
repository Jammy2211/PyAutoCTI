import os

import numpy as np
import pytest
import shutil

import autocti as ac

from test.unit.mock.mock import MockGeometry, MockPattern

test_data_dir = "{}/../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)


@pytest.fixture(scope="class", name="arctic_parallel")
def make_arctic_parallel():
    parallel_settings = ac.Settings(
        well_depth=84700, niter=1, express=5, n_levels=2000, readout_offset=0
    )
    parallel = ac.ArcticSettings(neomode="NEO", parallel=parallel_settings)

    return parallel


@pytest.fixture(scope="class", name="arctic_serial")
def make_arctic_serial():
    serial_settings = ac.Settings(
        well_depth=84700, niter=1, express=5, n_levels=2000, readout_offset=0
    )

    serial = ac.ArcticSettings(neomode="NEO", serial=serial_settings)

    return serial


@pytest.fixture(scope="class", name="arctic_both")
def make_arctic_both():
    parallel_settings = ac.Settings(
        well_depth=84700, niter=1, express=5, n_levels=2000, readout_offset=0
    )

    serial_settings = ac.Settings(
        well_depth=84700, niter=1, express=5, n_levels=2000, readout_offset=0
    )

    both = ac.ArcticSettings(
        neomode="NEO", parallel=parallel_settings, serial=serial_settings
    )

    return both


@pytest.fixture(scope="class", name="params_parallel")
def make_params_parallel():
    parallel = ac.Species(trap_density=0.1, trap_lifetime=1.0)

    ccd = ac.CCDVolume(well_notch_depth=0.000001, well_fill_beta=0.8)

    parallel = ac.ArcticParams(parallel_species=[parallel], parallel_ccd_volume=ccd)

    return parallel


@pytest.fixture(scope="class", name="params_serial")
def make_params_serial():
    serial = ac.Species(trap_density=0.2, trap_lifetime=2.0)

    ccd = ac.CCDVolume(well_notch_depth=0.000001, well_fill_beta=0.4)

    serial = ac.ArcticParams(serial_species=[serial], serial_ccd_volume=ccd)

    return serial


@pytest.fixture(scope="class", name="params_both")
def make_params_both():
    parallel = ac.Species(trap_density=0.4, trap_lifetime=1.0)

    parallel_ccd_volume = ac.CCDVolume(well_notch_depth=0.000001, well_fill_beta=0.8)

    serial = ac.Species(trap_density=0.2, trap_lifetime=2.0)

    serial_ccd_volume = ac.CCDVolume(well_notch_depth=0.000001, well_fill_beta=0.4)

    both = ac.ArcticParams(
        parallel_species=[parallel],
        serial_species=[serial],
        parallel_ccd_volume=parallel_ccd_volume,
        serial_ccd_volume=serial_ccd_volume,
    )

    return both


class TestCIData(object):
    def test__map(self):

        data = ac.CIData(
            image=1,
            noise_map=3,
            ci_pre_cti=4,
            ci_pattern=None,
            ci_frame=None,
            cosmic_ray_image=None,
        )

        result = data.map_to_ci_data_masked(func=lambda x: 2 * x, mask=1)
        assert isinstance(result, ac.CIDataMasked)
        assert result.image == 2
        assert result.noise_map == 6
        assert result.ci_pre_cti == 8
        assert result.cosmic_ray_image == None

        data = ac.CIData(
            image=1,
            noise_map=3,
            ci_pre_cti=4,
            ci_pattern=None,
            ci_frame=None,
            cosmic_ray_image=10,
        )
        result = data.map_to_ci_data_masked(func=lambda x: 2 * x, mask=1)
        assert isinstance(result, ac.CIDataMasked)
        assert result.image == 2
        assert result.noise_map == 6
        assert result.ci_pre_cti == 8
        assert result.cosmic_ray_image == 10

    def test__map_including_noise_scaling_maps(self):

        data = ac.CIData(
            image=1,
            noise_map=3,
            ci_pre_cti=4,
            ci_pattern=None,
            ci_frame=None,
            cosmic_ray_image=None,
        )
        result = data.map_to_ci_data_masked(
            func=lambda x: 2 * x, mask=1, noise_scaling_maps=[1, 2, 3]
        )
        assert isinstance(result, ac.CIDataMasked)
        assert result.image == 2
        assert result.noise_map == 6
        assert result.ci_pre_cti == 8
        assert result.noise_scaling_maps == [2, 4, 6]
        assert result.cosmic_ray_image == None

        data = ac.CIData(
            image=1,
            noise_map=3,
            ci_pre_cti=4,
            ci_pattern=None,
            ci_frame=None,
            cosmic_ray_image=10,
        )
        result = data.map_to_ci_data_masked(
            func=lambda x: 2 * x, mask=1, noise_scaling_maps=[1, 2, 3]
        )
        assert isinstance(result, ac.CIDataMasked)
        assert result.image == 2
        assert result.noise_map == 6
        assert result.ci_pre_cti == 8
        assert result.noise_scaling_maps == [2, 4, 6]
        assert result.cosmic_ray_image == 10

    def test__parallel_serial_ci_data_fit_from_mask(self):

        data = ac.CIData(
            image=1,
            noise_map=3,
            ci_pre_cti=4,
            ci_pattern=None,
            ci_frame=None,
            cosmic_ray_image=10,
        )

        def parallel_serial_extractor():
            def extractor(obj):
                return 2 * obj

            return extractor

        data.parallel_serial_extractor = parallel_serial_extractor
        result = data.parallel_serial_ci_data_masked_from_mask(mask=1)

        assert isinstance(result, ac.CIDataMasked)
        assert result.image == 2
        assert result.noise_map == 6
        assert result.ci_pre_cti == 8
        assert result.cosmic_ray_image == 10

    def test__parallel_serial_ci_data_fit_from_mask__include_noise_scaling_maps(self):
        data = ac.CIData(
            image=1,
            noise_map=3,
            ci_pre_cti=4,
            ci_pattern=None,
            ci_frame=None,
            cosmic_ray_image=10,
        )

        def parallel_serial_extractor():
            def extractor(obj):
                return 2 * obj

            return extractor

        data.parallel_serial_extractor = parallel_serial_extractor
        result = data.parallel_serial_ci_data_masked_from_mask(
            mask=1, noise_scaling_maps=[2, 3]
        )

        assert isinstance(result, ac.CIDataMasked)
        assert result.image == 2
        assert result.noise_map == 6
        assert result.ci_pre_cti == 8
        assert result.noise_scaling_maps == [4, 6]
        assert result.cosmic_ray_image == 10

    def test__signal_to_noise_map_and_max(self):
        image = np.ones((2, 2))
        image[0, 0] = 6.0

        data = ac.CIData(
            image=image,
            noise_map=2.0 * np.ones((2, 2)),
            ci_pre_cti=None,
            ci_pattern=None,
            ci_frame=None,
        )

        assert (data.signal_to_noise_map == np.array([[3.0, 0.5], [0.5, 0.5]])).all()

        assert data.signal_to_noise_max == 3.0


class TestCIDataSimulate(object):
    def test__no_instrumental_effects_input__only_cti_simulated(
        self, arctic_parallel, params_parallel
    ):

        pattern = ac.CIPatternUniform(normalization=10.0, regions=[(0, 1, 0, 5)])

        ci_pre_cti = pattern.simulate_ci_pre_cti(shape=(5, 5))

        ci_data_simulate = ac.CIData.simulate(
            ci_pre_cti=ci_pre_cti,
            frame_geometry=ac.FrameGeometry.euclid_bottom_left(),
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

        ci_data_simulate = ac.CIData.simulate(
            ci_pre_cti=ci_pre_cti,
            frame_geometry=ac.FrameGeometry.euclid_bottom_left(),
            ci_pattern=pattern,
            cti_settings=arctic_parallel,
            cti_params=params_parallel,
            read_noise=1.0,
            noise_seed=1,
        )

        image_no_noise = pattern.ci_pre_cti_from_shape(shape=(3, 3))

        # Use seed to give us a known read noises map we'll test for

        assert ci_data_simulate.image - image_no_noise == pytest.approx(
            np.array([[1.62, -0.61, -0.53], [-1.07, 0.87, -2.30], [1.74, -0.76, 0.32]]),
            1e-2,
        )

    def test__include_cosmics__is_added_to_image_and_trailed(
        self, arctic_parallel, params_parallel
    ):

        pattern = ac.CIPatternUniform(normalization=10.0, regions=[(0, 1, 0, 5)])

        ci_pre_cti = pattern.simulate_ci_pre_cti(shape=(5, 5))

        cosmic_ray_image = np.zeros((5, 5))
        cosmic_ray_image[2, 2] = 100.0

        ci_data_simulate = ac.CIData.simulate(
            ci_pre_cti=ci_pre_cti,
            frame_geometry=ac.FrameGeometry.euclid_bottom_left(),
            ci_pattern=pattern,
            cti_settings=arctic_parallel,
            cti_params=params_parallel,
            cosmic_ray_image=cosmic_ray_image,
        )

        assert ci_data_simulate.image[0, 0:5] == pytest.approx(
            np.array([10.0, 10.0, 10.0, 10.0, 10.0]), 1e-2
        )
        assert 0.0 < ci_data_simulate.image[1, 1] < 100.0
        assert ci_data_simulate.image[2, 2] > 98.0
        assert (ci_data_simulate.image[1, 1:4] > 0.0).all()
        assert (
            ci_data_simulate.cosmic_ray_image
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

        parallel_species = ac.Species(trap_density=10.0, trap_lifetime=1.0)
        parallel_species = ac.Species.poisson_species(
            species=[parallel_species], shape=(5, 5), seed=1
        )
        parallel_ccd_volume = ac.CCDVolume(
            well_notch_depth=1.0e-4,
            well_fill_beta=0.58,
            well_fill_gamma=0.0,
            well_fill_alpha=1.0,
        )

        params_parallel = ac.ArcticParams(
            parallel_species=parallel_species, parallel_ccd_volume=parallel_ccd_volume
        )

        ci_data_simulate = ac.CIData.simulate(
            ci_pre_cti=ci_pre_cti,
            frame_geometry=ac.FrameGeometry.euclid_bottom_left(),
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
            mask = ac.Mask.empty_for_shape(shape=(3, 3))

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

            # Use seed to give us a known read noises map we'll test for

            assert simulate_read_noise == pytest.approx(
                np.array(
                    [[1.62, -0.61, -0.53], [-1.07, 0.87, -2.30], [1.74, -0.76, 0.32]]
                ),
                1e-2,
            )


class TestCIPreCTI(object):
    def test__simple_case__sets_up_post_cti_correctly(self, arctic_both, params_both):
        frame_geometry = ac.FrameGeometry.euclid_bottom_left()

        ci_pre_cti = np.zeros((5, 5))
        ci_pre_cti[2, 2] = 10.0

        ci_post_cti = frame_geometry.add_cti(ci_pre_cti, params_both, arctic_both)

        image_difference = ci_post_cti - ci_pre_cti

        assert (image_difference[2, 2] < 0.0).all()  # dot loses charge
        assert (image_difference[3:5, 2] > 0.0).all()  # parallel trail behind dot
        assert (image_difference[2, 3:5] > 0.0).all()  # serial trail to right of dot


class TestLoadCIData(object):
    def test__load_all_data_components__has_correct_attributes(self):

        frame_geometry = MockGeometry()
        pattern = MockPattern()

        data = ac.ci_data_from_fits(
            frame_geometry=frame_geometry,
            ci_pattern=pattern,
            image_path=test_data_dir + "3x3_ones.fits",
            image_hdu=0,
            noise_map_path=test_data_dir + "3x3_twos.fits",
            noise_map_hdu=0,
            ci_pre_cti_path=test_data_dir + "3x3_threes.fits",
            ci_pre_cti_hdu=0,
            cosmic_ray_image_path=test_data_dir + "3x3_fours.fits",
            cosmic_ray_image_hdu=0,
        )

        assert (data.image == np.ones((3, 3))).all()
        assert (data.noise_map == 2.0 * np.ones((3, 3))).all()
        assert (data.ci_pre_cti == 3.0 * np.ones((3, 3))).all()
        assert data.ci_frame.frame_geometry == frame_geometry
        assert data.ci_frame.ci_pattern == pattern
        assert (data.cosmic_ray_image == 4.0 * np.ones((3, 3))).all()

    def test__load_all_image_components__load_from_multi_hdu_fits(self):

        frame_geometry = MockGeometry()
        pattern = MockPattern()

        data = ac.ci_data_from_fits(
            frame_geometry=frame_geometry,
            ci_pattern=pattern,
            image_path=test_data_dir + "3x3_multiple_hdu.fits",
            image_hdu=0,
            noise_map_path=test_data_dir + "3x3_multiple_hdu.fits",
            noise_map_hdu=1,
            ci_pre_cti_path=test_data_dir + "3x3_multiple_hdu.fits",
            ci_pre_cti_hdu=2,
            cosmic_ray_image_path=test_data_dir + "3x3_multiple_hdu.fits",
            cosmic_ray_image_hdu=3,
        )

        assert (data.image == np.ones((3, 3))).all()
        assert (data.noise_map == 2.0 * np.ones((3, 3))).all()
        assert (data.ci_pre_cti == 3.0 * np.ones((3, 3))).all()
        assert data.ci_frame.frame_geometry == frame_geometry
        assert data.ci_frame.ci_pattern == pattern
        assert (data.cosmic_ray_image == 4.0 * np.ones((3, 3))).all()

    def test__load_noise_map_from_single_value(self):

        frame_geometry = MockGeometry()
        pattern = MockPattern()

        data = ac.ci_data_from_fits(
            frame_geometry=frame_geometry,
            ci_pattern=pattern,
            image_path=test_data_dir + "3x3_ones.fits",
            image_hdu=0,
            noise_map_from_single_value=10.0,
            ci_pre_cti_path=test_data_dir + "3x3_threes.fits",
            ci_pre_cti_hdu=0,
        )

        assert (data.image == np.ones((3, 3))).all()
        assert (data.noise_map == 10.0 * np.ones((3, 3))).all()
        assert (data.ci_pre_cti == 3.0 * np.ones((3, 3))).all()
        assert data.ci_frame.frame_geometry == frame_geometry
        assert data.ci_frame.ci_pattern == pattern
        assert data.cosmic_ray_image == None

    def test__load_ci_pre_cti_image_from_the_pattern_and_image(self):

        frame_geometry = MockGeometry()
        pattern = ac.CIPatternUniform(regions=[(0, 3, 0, 3)], normalization=10.0)

        data = ac.ci_data_from_fits(
            frame_geometry=frame_geometry,
            ci_pattern=pattern,
            image_path=test_data_dir + "3x3_ones.fits",
            image_hdu=0,
            noise_map_path=test_data_dir + "3x3_twos.fits",
            noise_map_hdu=0,
        )

        assert (data.image == np.ones((3, 3))).all()
        assert (data.noise_map == 2.0 * np.ones((3, 3))).all()
        assert (data.ci_pre_cti == 10.0 * np.ones((3, 3))).all()
        assert data.ci_frame.frame_geometry == frame_geometry
        assert data.ci_frame.ci_pattern == pattern
        assert data.cosmic_ray_image == None

    def test__output_all_arrays(self):

        data = ac.ci_data_from_fits(
            frame_geometry=MockGeometry(),
            ci_pattern=MockPattern(),
            image_path=test_data_dir + "3x3_ones.fits",
            image_hdu=0,
            noise_map_path=test_data_dir + "3x3_twos.fits",
            noise_map_hdu=0,
            ci_pre_cti_path=test_data_dir + "3x3_threes.fits",
            ci_pre_cti_hdu=0,
            cosmic_ray_image_path=test_data_dir + "3x3_fours.fits",
            cosmic_ray_image_hdu=0,
        )

        output_data_dir = "{}/../test_files/array/output_test/".format(
            os.path.dirname(os.path.realpath(__file__))
        )
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        ac.output_ci_data_to_fits(
            ci_data=data,
            image_path=output_data_dir + "image.fits",
            noise_map_path=output_data_dir + "noise_map.fits",
            ci_pre_cti_path=output_data_dir + "ci_pre_cti.fits",
            cosmic_ray_image_path=output_data_dir + "cosmic_ray_image.fits",
        )

        data = ac.ci_data_from_fits(
            frame_geometry=MockGeometry(),
            ci_pattern=MockPattern(),
            image_path=output_data_dir + "image.fits",
            image_hdu=0,
            noise_map_path=output_data_dir + "noise_map.fits",
            noise_map_hdu=0,
            ci_pre_cti_path=output_data_dir + "ci_pre_cti.fits",
            ci_pre_cti_hdu=0,
            cosmic_ray_image_path=output_data_dir + "cosmic_ray_image.fits",
            cosmic_ray_image_hdu=0,
        )

        assert (data.image == np.ones((3, 3))).all()
        assert (data.noise_map == 2.0 * np.ones((3, 3))).all()
        assert (data.ci_pre_cti == 3.0 * np.ones((3, 3))).all()
        assert (data.cosmic_ray_image == 4.0 * np.ones((3, 3))).all()
