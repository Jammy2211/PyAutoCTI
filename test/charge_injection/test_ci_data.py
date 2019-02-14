#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
#
# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3.0 of the License, or (at your option)
# any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
#

"""
File: tests/python/CTICIData_test.py

Created on: 02/14/18
Author: user
"""

import os

import numpy as np
import pytest

from autocti.charge_injection import ci_data, ci_frame, ci_pattern
from autocti.charge_injection.ci_data import load_ci_data_from_fits
from autocti.data import mask as msk
from autocti.model import arctic_params
from autocti.model import arctic_settings
from test.mock.mock import MockGeometry, MockPattern

test_data_dir = "{}/../test_files/array/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(scope='class', name='empty_mask')
def make_empty_mask():
    parallel_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                                 readout_offset=0)
    parallel = arctic_settings.ArcticSettings(neomode='NEO', parallel=parallel_settings)

    return parallel


@pytest.fixture(scope='class', name='arctic_parallel')
def make_arctic_parallel():
    parallel_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                                 readout_offset=0)
    parallel = arctic_settings.ArcticSettings(neomode='NEO', parallel=parallel_settings)

    return parallel


@pytest.fixture(scope='class', name='arctic_serial')
def make_arctic_serial():
    serial_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                               readout_offset=0)

    serial = arctic_settings.ArcticSettings(neomode='NEO', serial=serial_settings)

    return serial


@pytest.fixture(scope='class', name='arctic_both')
def make_arctic_both():
    parallel_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                                 readout_offset=0)

    serial_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                               readout_offset=0)

    both = arctic_settings.ArcticSettings(neomode='NEO', parallel=parallel_settings,
                                          serial=serial_settings)

    return both


@pytest.fixture(scope='class', name='params_parallel')
def make_params_parallel():
    parallel = arctic_params.Species(trap_density=0.1, trap_lifetime=1.0)

    ccd = arctic_params.CCD(well_notch_depth=0.000001, well_fill_beta=0.8)

    parallel = arctic_params.ArcticParams(parallel_species=[parallel], parallel_ccd=ccd)

    return parallel


@pytest.fixture(scope='class', name='params_serial')
def make_params_serial():
    serial = arctic_params.Species(trap_density=0.2, trap_lifetime=2.0)

    ccd = arctic_params.CCD(well_notch_depth=0.000001, well_fill_beta=0.4)

    serial = arctic_params.ArcticParams(serial_species=[serial], serial_ccd=ccd)

    return serial


@pytest.fixture(scope='class', name='params_both')
def make_params_both():
    parallel = arctic_params.Species(trap_density=0.4, trap_lifetime=1.0)

    parallel_ccd = arctic_params.CCD(well_notch_depth=0.000001, well_fill_beta=0.8)

    serial = arctic_params.Species(trap_density=0.2, trap_lifetime=2.0)

    serial_ccd = arctic_params.CCD(well_notch_depth=0.000001, well_fill_beta=0.4)

    both = arctic_params.ArcticParams(parallel_species=[parallel],
                                      serial_species=[serial],
                                      parallel_ccd=parallel_ccd,
                                      serial_ccd=serial_ccd)

    return both


class TestCIData(object):

    def test_map(self):
        data = ci_data.CIData(image=1, noise_map=3, ci_pre_cti=4, ci_pattern=None, ci_frame=None)
        result = data.map_to_ci_data_fit(lambda x: 2 * x, 1)
        assert isinstance(result, ci_data.CIDataFit)
        assert result.image == 2
        assert result.noise_map == 6
        assert result.ci_pre_cti == 8

    def test__signal_to_noise_map_and_max(self):
        image = np.ones((2, 2))
        image[0, 0] = 6.0

        data = ci_data.CIData(image=image, noise_map=2.0 * np.ones((2, 2)), ci_pre_cti=None, ci_pattern=None,
                              ci_frame=None)

        assert (data.signal_to_noise_map == np.array([[3.0, 0.5],
                                                      [0.5, 0.5]])).all()

        assert data.signal_to_noise_max == 3.0


class TestCIImage(object):

    class TestSetupCIPreCTI:

        def test__uniform_pattern_1_region_normalization_10__correct_pre_clock_image(self):
            pattern = ci_pattern.CIPatternUniform(normalization=10.0, regions=[(0, 2, 0, 2)])
            image = 10.0 * np.ones((3, 3))

            ci_pre_cti = ci_data.ci_pre_cti_from_ci_pattern_geometry_image_and_mask(ci_pattern=pattern, image=image)

            assert (ci_pre_cti == np.array([[10.0, 10.0, 0.0],
                                            [10.0, 10.0, 0.0],
                                            [00.0, 00.0, 0.0]])).all()

        def test__same_as_above_but_different_normalization_and_regions(self):
            pattern = ci_pattern.CIPatternUniform(normalization=20.0,
                                                  regions=[(0, 2, 0, 1), (2, 3, 2, 3)])
            image = 10.0 * np.ones((3, 3))

            ci_pre_cti = ci_data.ci_pre_cti_from_ci_pattern_geometry_image_and_mask(ci_pattern=pattern, image=image)

            assert (ci_pre_cti == np.array([[20.0, 0.0, 0.0],
                                            [20.0, 0.0, 0.0],
                                            [0.0, 0.0, 20.0]])).all()

        def test__non_uniform_pattern__image_is_same_as_computed_image(self):
            pattern = ci_pattern.CIPatternNonUniform(
                normalization=100.0, regions=[(0, 2, 0, 2), (2, 3, 0, 3)],
                row_slope=-1.0)
            image = np.array([[10.0, 10.0, 10.0],
                              [2.0, 2.0, 2.0],
                              [8.0, 12.0, 10.0]])
            mask = msk.Mask.empty_for_shape(shape=(3, 3))

            ci_pre_cti = ci_data.ci_pre_cti_from_ci_pattern_geometry_image_and_mask(mask=mask,
                                                                                    ci_pattern=pattern, image=image)

            pattern_ci_pre_cti = pattern.ci_pre_cti_from_ci_image_and_mask(image, mask)

            # noinspection PyUnresolvedReferences
            assert (ci_pre_cti == pattern_ci_pre_cti).all()

    class TestCISimulate(object):

        def test__no_instrumental_effects_input__only_cti_simulated(self, arctic_parallel, params_parallel):

            pattern = ci_pattern.CIPatternUniform(normalization=10.0, regions=[(0, 1, 0, 5)])

            ci_pre_cti = pattern.simulate_ci_pre_cti(shape=(5,5))

            ci_data_simulate = ci_data.simulate(ci_pre_cti=ci_pre_cti,
                                           frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                           ci_pattern=pattern, cti_settings=arctic_parallel,
                                           cti_params=params_parallel)

            assert ci_data_simulate.image[0, 0:5] == pytest.approx(np.array([10.0, 10.0, 10.0, 10.0, 10.0]), 1e-2)

        def test__include_read_noise__is_added_after_cti(self, arctic_parallel, params_parallel):

            pattern = ci_pattern.CIPatternUniform(normalization=10.0, regions=[(0, 1, 0, 3)])

            ci_pre_cti = pattern.simulate_ci_pre_cti(shape=(3,3))

            ci_data_simulate = ci_data.simulate(ci_pre_cti=ci_pre_cti,
                                           frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                           ci_pattern=pattern, cti_settings=arctic_parallel,
                                           cti_params=params_parallel, read_noise=1.0, noise_seed=1)

            image_no_noise = pattern.ci_pre_cti_from_shape(shape=(3, 3))

            # Use seed to give us a known read noises map we'll test for

            assert (ci_data_simulate.image - image_no_noise == pytest.approx(np.array([[1.62, -0.61, -0.53],
                                                                            [-1.07, 0.87, -2.30],
                                                                            [1.74, -0.76, 0.32]]), 1e-2))

        def test__include_cosmics__is_added_to_image_and_trailed(self, arctic_parallel, params_parallel):

            pattern = ci_pattern.CIPatternUniform(normalization=10.0, regions=[(0, 1, 0, 5)])

            ci_pre_cti = pattern.simulate_ci_pre_cti(shape=(5,5))

            cosmics = np.zeros((5, 5))
            cosmics[1, 1] = 100.0

            ci_data_simulate = ci_data.simulate(ci_pre_cti=ci_pre_cti,
                                           frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                           ci_pattern=pattern, cti_settings=arctic_parallel,
                                           cti_params=params_parallel)

            assert ci_data_simulate.image[0, 0:5] == pytest.approx(np.array([10.0, 10.0, 10.0, 10.0, 10.0]), 1e-2)
            assert 0.0 < ci_data_simulate.image[1, 1] < 100.0
            assert (ci_data_simulate.image[1, 1:4] > 0.0).all()

    class TestCreateReadNoiseMap(object):

        def test__read_noise_sigma_0__read_noise_map_all_0__image_is_identical_to_input(self):
            simulate_read_noise = ci_data.read_noise_map_from_shape_and_sigma(shape=(3, 3), sigma=0.0, noise_seed=1)

            assert (simulate_read_noise == np.zeros((3, 3))).all()

        def test__read_noise_sigma_1__read_noise_map_all_non_0__image_has_noise_added(self):
            simulate_read_noise = ci_data.read_noise_map_from_shape_and_sigma(shape=(3, 3), sigma=1.0, noise_seed=1)

            # Use seed to give us a known read noises map we'll test for

            assert simulate_read_noise == pytest.approx(np.array([[1.62, -0.61, -0.53],
                                                                  [-1.07, 0.87, -2.30],
                                                                  [1.74, -0.76, 0.32]]), 1e-2)


class TestCIMask(object):
    class TestMaskRemoveRegions:

        def test__remove_one_region(self):
            mask = msk.Mask.create(shape=(3, 3), frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                   regions=[(0, 3, 2, 3)])

            assert (mask == np.array([[False, False, True],
                                      [False, False, True],
                                      [False, False, True]])).all()

        def test__remove_two_regions(self):
            mask = msk.Mask.create(shape=(3, 3), frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                   regions=[(0, 3, 2, 3), (0, 2, 0, 2)])

            assert (mask == np.array([[True, True, True],
                                      [True, True, True],
                                      [False, False, True]])).all()

    class TestCosmicRayMask:

        def test__cosmic_ray_mask_included_in_total_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = msk.Mask.create(shape=(3, 3), frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                   cosmic_rays=cosmic_rays)

            assert (mask == np.array([[False, False, False],
                                      [False, True, False],
                                      [False, False, False]])).all()

        def test__cosmic_ray_includes_trail_regions__and_a_mask_region(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = msk.Mask.create(shape=(3, 3), frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                   regions=[(0, 1, 0, 1)], cosmic_rays=cosmic_rays,
                                   cr_parallel=1, cr_serial=1)

            assert (mask == np.array([[True, False, False],
                                      [False, True, True],
                                      [False, True, False]])).all()

    class TestMaskCosmicsBottomLeftGeometry:

        def test__mask_one_cosmic_ray_with_parallel_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = msk.Mask.create(shape=(3, 3), frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                   cosmic_rays=cosmic_rays, cr_parallel=1)

            assert (mask == np.array([[False, False, False],
                                      [False, True, False],
                                      [False, True, False]])).all()

        def test__mask_one_cosmic_ray_with_longer_parallel_mask(self):
            cosmic_rays = np.array([[False, True, False],
                                    [False, False, False],
                                    [False, False, False]])

            mask = msk.Mask.create(shape=(3, 3), frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                   cosmic_rays=cosmic_rays, cr_parallel=2)

            assert (mask == np.array([[False, True, False],
                                      [False, True, False],
                                      [False, True, False]])).all()

        def test__mask_one_cosmic_ray_with_serial_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = msk.Mask.create(shape=(3, 3), frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                   cosmic_rays=cosmic_rays, cr_serial=1)

            assert (mask == np.array([[False, False, False],
                                      [False, True, True],
                                      [False, False, False]])).all()

        def test__mask_one_cosmic_ray_with_longer_serial_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [True, False, False],
                                    [False, False, False]])

            mask = msk.Mask.create(shape=(3, 3), frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                   cosmic_rays=cosmic_rays, cr_serial=2)

            assert (mask == np.array([[False, False, False],
                                      [True, True, True],
                                      [False, False, False]])).all()

        def test__mask_one_cosmic_ray_with_diagonal_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = msk.Mask.create(shape=(3, 3), frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                   cosmic_rays=cosmic_rays, cr_diagonal=1)

            assert (mask == np.array([[False, False, False],
                                      [False, True, True],
                                      [False, True, True]])).all()

        def test__mask_one_cosmic_ray_with_bigger_diagonal_mask(self):
            cosmic_rays = np.array([[False, False, False, False],
                                    [False, True, False, False],
                                    [False, False, False, False],
                                    [False, False, False, False]])

            mask = msk.Mask.create(shape=(4, 4), frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                   cosmic_rays=cosmic_rays, cr_diagonal=2)

            assert (mask == np.array([[False, False, False, False],
                                      [False, True, True, True],
                                      [False, True, True, True],
                                      [False, True, True, True]])).all()

    class TestMaskCosmicsBottomRightGeometry:

        def test__mask_one_cosmic_ray_with_parallel_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = msk.Mask.create(shape=(3, 3), frame_geometry=ci_frame.QuadGeometryEuclid.bottom_right(),
                                   cosmic_rays=cosmic_rays, cr_parallel=1)

            assert (mask == np.array([[False, False, False],
                                      [False, True, False],
                                      [False, True, False]])).all()

        def test__mask_one_cosmic_ray_with_longer_parallel_mask(self):
            cosmic_rays = np.array([[False, True, False],
                                    [False, False, False],
                                    [False, False, False]])

            mask = msk.Mask.create(shape=(3, 3), frame_geometry=ci_frame.QuadGeometryEuclid.bottom_right(),
                                   cosmic_rays=cosmic_rays, cr_parallel=2)

            assert (mask == np.array([[False, True, False],
                                      [False, True, False],
                                      [False, True, False]])).all()

        def test__mask_one_cosmic_ray_with_serial_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = msk.Mask.create(shape=(3, 3), frame_geometry=ci_frame.QuadGeometryEuclid.bottom_right(),
                                   cosmic_rays=cosmic_rays, cr_serial=1)

            assert (mask == np.array([[False, False, False],
                                      [True, True, False],
                                      [False, False, False]])).all()

        def test__mask_one_cosmic_ray_with_longer_serial_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, False, True],
                                    [False, False, False]])

            mask = msk.Mask.create(shape=(3, 3), frame_geometry=ci_frame.QuadGeometryEuclid.bottom_right(),
                                   cosmic_rays=cosmic_rays, cr_serial=2)

            assert (mask == np.array([[False, False, False],
                                      [True, True, True],
                                      [False, False, False]])).all()

        def test__mask_one_cosmic_ray_with_diagonal_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = msk.Mask.create(shape=(3, 3), frame_geometry=ci_frame.QuadGeometryEuclid.bottom_right(),
                                   cosmic_rays=cosmic_rays, cr_diagonal=1)

            assert (mask == np.array([[False, False, False],
                                      [True, True, False],
                                      [True, True, False]])).all()

        def test__mask_one_cosmic_ray_with_bigger_diagonal_mask(self):
            cosmic_rays = np.array([[False, False, False, False],
                                    [False, False, False, True],
                                    [False, False, False, False],
                                    [False, False, False, False]])

            mask = msk.Mask.create(shape=(4, 4), frame_geometry=ci_frame.QuadGeometryEuclid.bottom_right(),
                                   cosmic_rays=cosmic_rays, cr_diagonal=2)

            assert (mask == np.array([[False, False, False, False],
                                      [False, True, True, True],
                                      [False, True, True, True],
                                      [False, True, True, True]])).all()

    class TestMaskCosmicsTopLeftGeometry:

        def test__mask_one_cosmic_ray_with_parallel_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = msk.Mask.create(shape=(3, 3), frame_geometry=ci_frame.QuadGeometryEuclid.top_left(),
                                   cosmic_rays=cosmic_rays, cr_parallel=1)

            assert (mask == np.array([[False, True, False],
                                      [False, True, False],
                                      [False, False, False]])).all()

        def test__mask_one_cosmic_ray_with_longer_parallel_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, False, False],
                                    [False, True, False]])

            mask = msk.Mask.create(shape=(3, 3), frame_geometry=ci_frame.QuadGeometryEuclid.top_left(),
                                   cosmic_rays=cosmic_rays, cr_parallel=2)

            assert (mask == np.array([[False, True, False],
                                      [False, True, False],
                                      [False, True, False]])).all()

        def test__mask_one_cosmic_ray_with_serial_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = msk.Mask.create(shape=(3, 3), frame_geometry=ci_frame.QuadGeometryEuclid.top_left(),
                                   cosmic_rays=cosmic_rays, cr_serial=1)

            assert (mask == np.array([[False, False, False],
                                      [False, True, True],
                                      [False, False, False]])).all()

        def test__mask_one_cosmic_ray_with_longer_serial_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [True, False, False],
                                    [False, False, False]])

            mask = msk.Mask.create(shape=(3, 3), frame_geometry=ci_frame.QuadGeometryEuclid.top_left(),
                                   cosmic_rays=cosmic_rays, cr_serial=2)

            assert (mask == np.array([[False, False, False],
                                      [True, True, True],
                                      [False, False, False]])).all()

        def test__mask_one_cosmic_ray_with_diagonal_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = msk.Mask.create(shape=(3, 3), frame_geometry=ci_frame.QuadGeometryEuclid.top_left(),
                                   cosmic_rays=cosmic_rays, cr_diagonal=1)

            assert (mask == np.array([[False, True, True],
                                      [False, True, True],
                                      [False, False, False]])).all()

        def test__mask_one_cosmic_ray_with_bigger_diagonal_mask(self):
            cosmic_rays = np.array([[False, False, False, False],
                                    [False, False, False, False],
                                    [False, False, False, False],
                                    [False, True, False, False]])

            mask = msk.Mask.create(shape=(4, 4), frame_geometry=ci_frame.QuadGeometryEuclid.top_left(),
                                   cosmic_rays=cosmic_rays, cr_diagonal=2)

            assert (mask == np.array([[False, False, False, False],
                                      [False, True, True, True],
                                      [False, True, True, True],
                                      [False, True, True, True]])).all()

    class TestMaskCosmicsTopRightGeometry:

        def test__mask_one_cosmic_ray_with_parallel_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = msk.Mask.create(shape=(3, 3), frame_geometry=ci_frame.QuadGeometryEuclid.top_right(),
                                   cosmic_rays=cosmic_rays, cr_parallel=1)

            assert (mask == np.array([[False, True, False],
                                      [False, True, False],
                                      [False, False, False]])).all()

        def test__mask_one_cosmic_ray_with_longer_parallel_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, False, False],
                                    [False, True, False]])

            mask = msk.Mask.create(shape=(3, 3), frame_geometry=ci_frame.QuadGeometryEuclid.top_right(),
                                   cosmic_rays=cosmic_rays, cr_parallel=2)
            assert (mask == np.array([[False, True, False],
                                      [False, True, False],
                                      [False, True, False]])).all()

        def test__mask_one_cosmic_ray_with_serial_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = msk.Mask.create(shape=(3, 3), frame_geometry=ci_frame.QuadGeometryEuclid.top_right(),
                                   cosmic_rays=cosmic_rays, cr_serial=1)

            assert (mask == np.array([[False, False, False],
                                      [True, True, False],
                                      [False, False, False]])).all()

        def test__mask_one_cosmic_ray_with_longer_serial_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, False, True],
                                    [False, False, False]])

            mask = msk.Mask.create(shape=(3, 3), frame_geometry=ci_frame.QuadGeometryEuclid.top_right(),
                                   cosmic_rays=cosmic_rays, cr_serial=2)

            assert (mask == np.array([[False, False, False],
                                      [True, True, True],
                                      [False, False, False]])).all()

        def test__mask_one_cosmic_ray_with_diagonal_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = msk.Mask.create(shape=(3, 3), frame_geometry=ci_frame.QuadGeometryEuclid.top_right(),
                                   cosmic_rays=cosmic_rays, cr_diagonal=1)

            assert (mask == np.array([[True, True, False],
                                      [True, True, False],
                                      [False, False, False]])).all()

        def test__mask_one_cosmic_ray_with_bigger_diagonal_mask(self):
            cosmic_rays = np.array([[False, False, False, False],
                                    [False, False, False, False],
                                    [False, False, False, False],
                                    [False, False, False, True]])

            mask = msk.Mask.create(shape=(4, 4), frame_geometry=ci_frame.QuadGeometryEuclid.top_right(),
                                   cosmic_rays=cosmic_rays, cr_diagonal=2)

            assert (mask == np.array([[False, False, False, False],
                                      [False, True, True, True],
                                      [False, True, True, True],
                                      [False, True, True, True]])).all()


class TestCIPreCTI(object):

    def test__simple_case__sets_up_post_cti_correctly(self, arctic_both, params_both):
        frame_geometry = ci_frame.QuadGeometryEuclid.bottom_left()

        ci_pre_cti = np.zeros((5, 5))
        ci_pre_cti[2, 2] = 10.0

        ci_post_cti = frame_geometry.add_cti(ci_pre_cti, params_both, arctic_both)

        image_difference = ci_post_cti - ci_pre_cti

        assert (image_difference[2, 2] < 0.0).all()  # dot loses charge
        assert (image_difference[3:5, 2] > 0.0).all()  # parallel trail behind dot
        assert (image_difference[2, 3:5] > 0.0).all()  # serial trail to right of dot


class TestLoadCIDataList(object):

    def test__load_all_data_components_as_list__gives_correct_components(self):
        datas = load_ci_data_list_from_fits(frame_geometries=[MockGeometry(), MockGeometry()],
                                            ci_patterns=[MockPattern(), MockPattern()],
                                            ci_image_paths=[test_data_dir + '3x3_ones.fits',
                                                            test_data_dir + '3x3_fours.fits'],
                                            ci_noise_map_paths=[test_data_dir + '3x3_twos.fits',
                                                                test_data_dir + '3x3_fives.fits'],
                                            ci_pre_cti_paths=[test_data_dir + '3x3_threes.fits',
                                                              test_data_dir + '3x3_sixes.fits'])

        assert (datas[0].image == np.ones((3, 3))).all()
        assert (datas[0].noise_map == 2.0 * np.ones((3, 3))).all()
        assert (datas[0].ci_pre_cti == 3.0 * np.ones((3, 3))).all()
        assert (datas[1].image == 4.0 * np.ones((3, 3))).all()
        assert (datas[1].noise_map == 5.0 * np.ones((3, 3))).all()
        assert (datas[1].ci_pre_cti == 6.0 * np.ones((3, 3))).all()

    def test__load_all_data_components_as_list__use_multi_hdu_image(self):
        datas = load_ci_data_list_from_fits(frame_geometries=[MockGeometry(), MockGeometry()],
                                            ci_patterns=[MockPattern(), MockPattern()],
                                            ci_image_paths=[test_data_dir + '3x3_multiple_hdu.fits',
                                                            test_data_dir + '3x3_multiple_hdu.fits'],
                                            ci_image_hdus=[0, 3],
                                            ci_noise_map_paths=[test_data_dir + '3x3_multiple_hdu.fits',
                                                                test_data_dir + '3x3_multiple_hdu.fits'],
                                            ci_noise_map_hdus=[1, 4],
                                            ci_pre_cti_paths=[test_data_dir + '3x3_multiple_hdu.fits',
                                                              test_data_dir + '3x3_multiple_hdu.fits'],
                                            ci_pre_cti_hdus=[2, 5])

        assert (datas[0].image == np.ones((3, 3))).all()
        assert (datas[0].noise_map == 2.0 * np.ones((3, 3))).all()
        assert (datas[0].ci_pre_cti == 3.0 * np.ones((3, 3))).all()
        assert (datas[1].image == 4.0 * np.ones((3, 3))).all()
        assert (datas[1].noise_map == 5.0 * np.ones((3, 3))).all()
        assert (datas[1].ci_pre_cti == 6.0 * np.ones((3, 3))).all()

    def test__lload_ci_pre_cti_image_from_the_pattern_and_image(self):
        datas = load_ci_data_list_from_fits(frame_geometries=[MockGeometry(), MockGeometry()],
                                            ci_patterns=[
                                                ci_pattern.CIPatternUniform(regions=[(0, 3, 0, 3)],
                                                                            normalization=10.0),
                                                ci_pattern.CIPatternUniform(regions=[(0, 3, 0, 3)],
                                                                            normalization=11.0)],
                                            ci_image_paths=[test_data_dir + '3x3_ones.fits',
                                                            test_data_dir + '3x3_fours.fits'],
                                            ci_noise_map_paths=[test_data_dir + '3x3_twos.fits',
                                                                test_data_dir + '3x3_fives.fits'])

        assert (datas[0].image == np.ones((3, 3))).all()
        assert (datas[0].noise_map == 2.0 * np.ones((3, 3))).all()
        assert (datas[0].ci_pre_cti == 10.0 * np.ones((3, 3))).all()
        assert (datas[1].image == 4.0 * np.ones((3, 3))).all()
        assert (datas[1].noise_map == 5.0 * np.ones((3, 3))).all()
        assert (datas[1].ci_pre_cti == 11.0 * np.ones((3, 3))).all()


class TestLoadCIData(object):

    def test__load_all_data_components__has_correct_attributes(self):
        data = ci_data.load_ci_data_from_fits(frame_geometry=MockGeometry(), ci_pattern=MockPattern(),
                                              ci_image_path=test_data_dir + '3x3_ones.fits', ci_image_hdu=0,
                                              ci_noise_map_path=test_data_dir + '3x3_twos.fits', ci_noise_map_hdu=0,
                                              ci_pre_cti_path=test_data_dir + '3x3_threes.fits', ci_pre_cti_hdu=0)

        assert (data.image == np.ones((3, 3))).all()
        assert (data.noise_map == 2.0 * np.ones((3, 3))).all()
        assert (data.ci_pre_cti == 3.0 * np.ones((3, 3))).all()

    def test__load_all_image_components__load_from_multi_hdu_fits(self):
        data = ci_data.load_ci_data_from_fits(frame_geometry=MockGeometry(), ci_pattern=MockPattern(),
                                              ci_image_path=test_data_dir + '3x3_multiple_hdu.fits', ci_image_hdu=0,
                                              ci_noise_map_path=test_data_dir + '3x3_multiple_hdu.fits',
                                              ci_noise_map_hdu=1,
                                              ci_pre_cti_path=test_data_dir + '3x3_multiple_hdu.fits', ci_pre_cti_hdu=2)

        assert (data.image == np.ones((3, 3))).all()
        assert (data.noise_map == 2.0 * np.ones((3, 3))).all()
        assert (data.ci_pre_cti == 3.0 * np.ones((3, 3))).all()

    def test__load_noise_map_from_single_value(self):
        data = ci_data.load_ci_data_from_fits(frame_geometry=MockGeometry(), ci_pattern=MockPattern(),
                                              ci_image_path=test_data_dir + '3x3_ones.fits', ci_image_hdu=0,
                                              ci_noise_map_from_single_value=10.0,
                                              ci_pre_cti_path=test_data_dir + '3x3_threes.fits', ci_pre_cti_hdu=0)

        assert (data.image == np.ones((3, 3))).all()
        assert (data.noise_map == 10.0 * np.ones((3, 3))).all()
        assert (data.ci_pre_cti == 3.0 * np.ones((3, 3))).all()

    def test__load_ci_pre_cti_image_from_the_pattern_and_image(self):
        data = ci_data.load_ci_data_from_fits(frame_geometry=MockGeometry(),
                                              ci_pattern=ci_pattern.CIPatternUniform(regions=[(0, 3, 0, 3)],
                                                                                     normalization=10.0),
                                              ci_image_path=test_data_dir + '3x3_ones.fits', ci_image_hdu=0,
                                              ci_noise_map_path=test_data_dir + '3x3_twos.fits', ci_noise_map_hdu=0)

        assert (data.image == np.ones((3, 3))).all()
        assert (data.noise_map == 2.0 * np.ones((3, 3))).all()
        assert (data.ci_pre_cti == 10.0 * np.ones((3, 3))).all()


def load_ci_data_list_from_fits(frame_geometries, ci_patterns,
                                ci_image_paths, ci_image_hdus=None,
                                ci_noise_map_paths=None, ci_noise_map_hdus=None,
                                ci_pre_cti_paths=None, ci_pre_cti_hdus=None,
                                masks=None):
    list_size = len(ci_image_paths)

    ci_datas = []

    if ci_pre_cti_paths is None:
        ci_pre_cti_paths = list_size * [None]

    if masks is None:
        masks = list_size * [None]

    if ci_image_hdus is None:
        ci_image_hdus = list_size * [0]

    if ci_noise_map_hdus is None:
        ci_noise_map_hdus = list_size * [0]

    if ci_pre_cti_hdus is None:
        ci_pre_cti_hdus = list_size * [0]

    for data_index in range(list_size):
        cid = load_ci_data_from_fits(frame_geometry=frame_geometries[data_index],
                                     ci_pattern=ci_patterns[data_index],
                                     ci_image_path=ci_image_paths[data_index],
                                     ci_image_hdu=ci_image_hdus[data_index],
                                     ci_noise_map_path=ci_noise_map_paths[data_index],
                                     ci_noise_map_hdu=ci_noise_map_hdus[data_index],
                                     ci_pre_cti_path=ci_pre_cti_paths[data_index],
                                     ci_pre_cti_hdu=ci_pre_cti_hdus[data_index],
                                     mask=masks[data_index])

        ci_datas.append(cid)

    return ci_datas
