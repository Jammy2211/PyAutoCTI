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

from __future__ import division, print_function

import numpy as np
import pytest

from autocti import exc
from autocti.data.charge_injection import ci_data
from autocti.data.charge_injection import ci_frame, ci_pattern
from autocti.model import arctic_params
from autocti.model import arctic_settings


@pytest.fixture(scope='class')
def empty_mask():
    parallel_settings = arctic_settings.ParallelSettings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                                         readout_offset=0)
    arctic_parallel = arctic_settings.ArcticSettings(neomode='NEO', parallel=parallel_settings)

    return arctic_parallel


@pytest.fixture(scope='class')
def arctic_parallel():
    parallel_settings = arctic_settings.ParallelSettings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                                         readout_offset=0)
    arctic_parallel = arctic_settings.ArcticSettings(neomode='NEO', parallel=parallel_settings)

    return arctic_parallel


@pytest.fixture(scope='class')
def arctic_serial():
    serial_settings = arctic_settings.SerialSettings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                                     readout_offset=0)

    arctic_serial = arctic_settings.ArcticSettings(neomode='NEO', serial=serial_settings)

    return arctic_serial


@pytest.fixture(scope='class')
def arctic_both():
    parallel_settings = arctic_settings.ParallelSettings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                                         readout_offset=0)

    serial_settings = arctic_settings.SerialSettings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                                     readout_offset=0)

    arctic_both = arctic_settings.ArcticSettings(neomode='NEO', parallel=parallel_settings,
                                                 serial=serial_settings)

    return arctic_both


@pytest.fixture(scope='class')
def params_parallel():
    params_parallel = arctic_params.ParallelOneSpecies(trap_densities=(0.1,), trap_lifetimes=(1.0,),
                                                       well_notch_depth=0.000001, well_fill_beta=0.8)

    params_parallel = arctic_params.ArcticParams(parallel=params_parallel)

    return params_parallel


@pytest.fixture(scope='class')
def params_serial():
    params_serial = arctic_params.SerialOneSpecies(trap_densities=(0.2,), trap_lifetimes=(2.0,),
                                                   well_notch_depth=0.000001, well_fill_beta=0.4)

    params_serial = arctic_params.ArcticParams(serial=params_serial)

    return params_serial


@pytest.fixture(scope='class')
def params_both():
    params_parallel = arctic_params.ParallelOneSpecies(trap_densities=(0.4,), trap_lifetimes=(1.0,),
                                                       well_notch_depth=0.000001, well_fill_beta=0.8)

    params_serial = arctic_params.SerialOneSpecies(trap_densities=(0.2,), trap_lifetimes=(2.0,),
                                                   well_notch_depth=0.000001, well_fill_beta=0.4)

    params_both = arctic_params.ArcticParams(parallel=params_parallel,
                                             serial=params_serial)

    return params_both


class MockPattern(object):

    def __init__(self):
        pass


class MockGeometry(ci_frame.CIQuadGeometry):

    def __init__(self):
        super(MockGeometry, self).__init__()


class TestCIData(object):

    def test__constructor(self):
        data = ci_data.CIData(images=[np.ones((2, 2)), 5.0 * np.ones((2, 2))],
                              masks=[2.0 * np.ones((2, 2)), 6.0 * np.ones((2, 2))],
                              noises=[3.0 * np.ones((2, 2)), 7.0 * np.ones((2, 2))],
                              ci_pre_ctis=[4.0 * np.ones((2, 2)), 8.0 * np.ones((2, 2))])

        assert (data[0].image == np.ones((2, 2))).all()
        assert (data[0].mask == 2.0 * np.ones((2, 2))).all()
        assert (data[0].noise == 3.0 * np.ones((2, 2))).all()
        assert (data[0].ci_pre_cti == 4.0 * np.ones((2, 2))).all()

        assert (data[1].image == 5.0 * np.ones((2, 2))).all()
        assert (data[1].mask == 6.0 * np.ones((2, 2))).all()
        assert (data[1].noise == 7.0 * np.ones((2, 2))).all()
        assert (data[1].ci_pre_cti == 8.0 * np.ones((2, 2))).all()


class TestCIDataAnalysis(object):

    def test__constructor__no_scaled_noise(self):
        data = ci_data.CIDataAnalysis(images=[np.ones((2, 2)), 5.0 * np.ones((2, 2))],
                                      masks=[2.0 * np.ones((2, 2)), 6.0 * np.ones((2, 2))],
                                      noises=[3.0 * np.ones((2, 2)), 7.0 * np.ones((2, 2))],
                                      ci_pre_ctis=[4.0 * np.ones((2, 2)), 8.0 * np.ones((2, 2))])

        assert (data[0].image == np.ones((2, 2))).all()
        assert (data[0].mask == 2.0 * np.ones((2, 2))).all()
        assert (data[0].noise == 3.0 * np.ones((2, 2))).all()
        assert (data[0].ci_pre_cti == 4.0 * np.ones((2, 2))).all()

        assert (data[1].image == 5.0 * np.ones((2, 2))).all()
        assert (data[1].mask == 6.0 * np.ones((2, 2))).all()
        assert (data[1].noise == 7.0 * np.ones((2, 2))).all()
        assert (data[1].ci_pre_cti == 8.0 * np.ones((2, 2))).all()

    def test__constructor__include_scaled_noise__x1_scaled_noise_per_image(self):
        data = ci_data.CIDataAnalysis(images=[np.ones((2, 2)), 5.0 * np.ones((2, 2))],
                                      masks=[2.0 * np.ones((2, 2)), 6.0 * np.ones((2, 2))],
                                      noises=[3.0 * np.ones((2, 2)), 7.0 * np.ones((2, 2))],
                                      ci_pre_ctis=[4.0 * np.ones((2, 2)), 8.0 * np.ones((2, 2))],
                                      noise_scalings=[[9.0 * np.ones((2, 2))], [10.0 * np.ones((2, 2))]])

        assert (data[0].image == np.ones((2, 2))).all()
        assert (data[0].mask == 2.0 * np.ones((2, 2))).all()
        assert (data[0].noise == 3.0 * np.ones((2, 2))).all()
        assert (data[0].ci_pre_cti == 4.0 * np.ones((2, 2))).all()
        assert (data[0].noise_scalings[0] == 9.0 * np.ones((2, 2))).all()

        assert (data[1].image == 5.0 * np.ones((2, 2))).all()
        assert (data[1].mask == 6.0 * np.ones((2, 2))).all()
        assert (data[1].noise == 7.0 * np.ones((2, 2))).all()
        assert (data[1].ci_pre_cti == 8.0 * np.ones((2, 2))).all()
        assert (data[1].noise_scalings[0] == 10.0 * np.ones((2, 2))).all()

    def test__constructor__include_scaled_noise__x2_scaled_noise_per_image(self):
        data = ci_data.CIDataAnalysis(images=[np.ones((2, 2)), 5.0 * np.ones((2, 2))],
                                      masks=[2.0 * np.ones((2, 2)), 6.0 * np.ones((2, 2))],
                                      noises=[3.0 * np.ones((2, 2)), 7.0 * np.ones((2, 2))],
                                      ci_pre_ctis=[4.0 * np.ones((2, 2)), 8.0 * np.ones((2, 2))],
                                      noise_scalings=[[9.0 * np.ones((2, 2)), 11.0 * np.ones((2, 2))],
                                                      [10.0 * np.ones((2, 2)), 12.0 * np.ones((2, 2))]])

        assert (data[0].image == np.ones((2, 2))).all()
        assert (data[0].mask == 2.0 * np.ones((2, 2))).all()
        assert (data[0].noise == 3.0 * np.ones((2, 2))).all()
        assert (data[0].ci_pre_cti == 4.0 * np.ones((2, 2))).all()
        assert (data[0].noise_scalings[0] == 9.0 * np.ones((2, 2))).all()
        assert (data[0].noise_scalings[1] == 11.0 * np.ones((2, 2))).all()

        assert (data[1].image == 5.0 * np.ones((2, 2))).all()
        assert (data[1].mask == 6.0 * np.ones((2, 2))).all()
        assert (data[1].noise == 7.0 * np.ones((2, 2))).all()
        assert (data[1].ci_pre_cti == 8.0 * np.ones((2, 2))).all()
        assert (data[1].noise_scalings[0] == 10.0 * np.ones((2, 2))).all()
        assert (data[1].noise_scalings[1] == 12.0 * np.ones((2, 2))).all()


class TestCIImage(object):
    class TestConstructor:

        def test__setup_all_attributes_correctly__noise_is_generated_by_read_noise(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 1, 0, 1)])
            image = np.array([[10.0, 10.0, 10.0],
                              [2.0, 2.0, 2.0],
                              [8.0, 12.0, 10.0]])

            data = ci_data.CIImage(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(), ci_pattern=pattern, array=image)

            assert data.frame_geometry.parallel_overscan == (2066, 2086, 51, 2099)
            assert data.frame_geometry.serial_prescan == (0, 2086, 0, 51)
            assert data.frame_geometry.serial_overscan == (0, 2086, 2099, 2119)
            assert type(data.ci_pattern) == ci_pattern.CIPattern
            assert data.shape == (3, 3)
            assert (data == np.array([[10.0, 10.0, 10.0],
                                      [2.0, 2.0, 2.0],
                                      [8.0, 12.0, 10.0]])).all()

    class TestSetupCIPreCTI:

        def test__uniform_pattern_1_region_normalization_10__correct_pre_clock_image(self):
            pattern = ci_pattern.CIPatternUniform(normalization=10.0, regions=[(0, 2, 0, 2)])
            image = 10.0 * np.ones((3, 3))

            data = ci_data.CIImage(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(), ci_pattern=pattern,
                                   array=image)

            ci_pre_cti = data.create_ci_pre_cti()

            assert type(ci_pre_cti.frame_geometry) == ci_frame.CIQuadGeometryEuclidBL
            assert (ci_pre_cti == np.array([[10.0, 10.0, 0.0],
                                            [10.0, 10.0, 0.0],
                                            [00.0, 00.0, 0.0]])).all()
            assert (ci_pre_cti.ci_pattern.normalization == pattern.normalization == 10.0)
            assert (ci_pre_cti.ci_pattern.regions == pattern.regions == [(0, 2, 0, 2)])

        def test__same_as_above_but_different_normalization_and_regions(self):
            frame_geometry = ci_frame.CIQuadGeometryEuclidBL()
            pattern = ci_pattern.CIPatternUniform(normalization=20.0,
                                                  regions=[(0, 2, 0, 1), (2, 3, 2, 3)])
            image = 10.0 * np.ones((3, 3))

            data = ci_data.CIImage(frame_geometry=frame_geometry, ci_pattern=pattern, array=image)

            ci_pre_cti = data.create_ci_pre_cti()

            assert type(ci_pre_cti.frame_geometry) == ci_frame.CIQuadGeometryEuclidBL
            assert (ci_pre_cti == np.array([[20.0, 0.0, 0.0],
                                            [20.0, 0.0, 0.0],
                                            [0.0, 0.0, 20.0]])).all()
            assert (ci_pre_cti.ci_pattern.normalization == pattern.normalization == 20.0)
            assert (ci_pre_cti.ci_pattern.regions == pattern.regions == [(0, 2, 0, 1), (2, 3, 2, 3)])

        def test__non_uniform_pattern__image_is_same_as_computed_image(self):
            frame_geometry = ci_frame.CIQuadGeometryEuclidBL()
            pattern = ci_pattern.CIPatternNonUniform(
                normalization=100.0, regions=[(0, 2, 0, 2), (2, 3, 0, 3)],
                row_slope=-1.0)
            image = np.array([[10.0, 10.0, 10.0],
                              [2.0, 2.0, 2.0],
                              [8.0, 12.0, 10.0]])
            mask = ci_data.CIMask.empty_for_shape(shape=(3, 3), frame_geometry=frame_geometry, ci_pattern=MockPattern())

            data = ci_data.CIImage(frame_geometry=frame_geometry, ci_pattern=pattern, array=image)

            ci_pre_cti = data.create_ci_pre_cti(mask=mask)

            pattern_ci_pre_cti = pattern.compute_ci_pre_cti(image, mask)

            assert type(ci_pre_cti.frame_geometry) == ci_frame.CIQuadGeometryEuclidBL
            assert (ci_pre_cti == pattern_ci_pre_cti).all()
            assert (ci_pre_cti.ci_pattern.normalization == pattern.normalization == 100.0)
            assert (ci_pre_cti.ci_pattern.regions == pattern.regions == [(0, 2, 0, 2), (2, 3, 0, 3)])

        def test__fast_uniform_pattern__fast_ci_pre_cti(self):
            frame_geometry = ci_frame.CIQuadGeometryEuclidBL()
            pattern = ci_pattern.CIPatternUniformFast(
                normalization=10.0, regions=[(0, 2, 0, 2)])
            image = 10.0 * np.ones((3, 3))
            mask = ci_data.CIMask.empty_for_shape(shape=(3, 3), frame_geometry=frame_geometry, ci_pattern=MockPattern())

            data = ci_data.CIImage(frame_geometry=frame_geometry, ci_pattern=pattern, array=image)

            ci_pre_cti = data.create_ci_pre_cti()

            assert type(ci_pre_cti) == ci_data.CIPreCTIFast
            assert (ci_pre_cti.fast_column_pre_cti == np.array([[10.0],
                                                                [10.0],
                                                                [0.0]])).all()
            assert (ci_pre_cti.fast_row_pre_cti == np.array([[10.0],
                                                             [10.0],
                                                             [0.0]])).all()

            assert type(ci_pre_cti.frame_geometry) == ci_frame.CIQuadGeometryEuclidBL
            assert (ci_pre_cti == np.array([[10.0, 10.0, 0.0],
                                            [10.0, 10.0, 0.0],
                                            [00.0, 00.0, 0.0]])).all()
            assert (ci_pre_cti.ci_pattern.normalization == pattern.normalization == 10.0)
            assert (ci_pre_cti.ci_pattern.regions == pattern.regions == [(0, 2, 0, 2)])

    class TestCISimulate(object):

        def test__no_instrumental_effects_input__only_cti_simulated(self, arctic_parallel, params_parallel):
            pattern = ci_pattern.CIPatternUniformSimulate(normalization=10.0, regions=[(0, 1, 0, 5)])

            ci_simulate = ci_data.CIImage.simulate(shape=(5, 5), frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                                   ci_pattern=pattern, cti_settings=arctic_parallel,
                                                   cti_params=params_parallel)

            assert ci_simulate[0, 0:5] == pytest.approx(np.array([10.0, 10.0, 10.0, 10.0, 10.0]), 1e-2)
            # assert ci_simulate[1:5, 0:5] == pytest.approx(np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
            #                                                         [0.0, 0.0, 0.0, 0.0, 0.0],
            #                                                         [0.0, 0.0, 0.0, 0.0, 0.0],
            #                                                         [0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-2)

        def test__include_read_noise__is_added_after_cti(self, arctic_parallel, params_parallel):
            pattern = ci_pattern.CIPatternUniformSimulate(normalization=10.0, regions=[(0, 1, 0, 3)])

            ci_simulate = ci_data.CIImage.simulate(shape=(3, 3), frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                                   ci_pattern=pattern, cti_settings=arctic_parallel,
                                                   cti_params=params_parallel, read_noise=1.0, noise_seed=1)

            image_no_noise = pattern.compute_ci_pre_cti(shape=(3, 3))

            # Use seed to give us a known read noises map we'll test for

            assert (ci_simulate - image_no_noise == pytest.approx(np.array([[1.62, -0.61, -0.53],
                                                                            [-1.07, 0.87, -2.30],
                                                                            [1.74, -0.76, 0.32]]), 1e-2))

        def test__include_cosmics__is_added_to_image_and_trailed(self, arctic_parallel, params_parallel):
            pattern = ci_pattern.CIPatternUniformSimulate(normalization=10.0, regions=[(0, 1, 0, 5)])

            cosmics = np.zeros((5, 5))
            cosmics[1, 1] = 100.0

            ci_simulate = ci_data.CIImage.simulate(shape=(5, 5), frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                                   ci_pattern=pattern, cti_settings=arctic_parallel,
                                                   cti_params=params_parallel)

            assert ci_simulate[0, 0:5] == pytest.approx(np.array([10.0, 10.0, 10.0, 10.0, 10.0]), 1e-2)
            assert 0.0 < ci_simulate[1, 1] < 100.0
            assert (ci_simulate[1, 1:4] > 0.0).all()

    class TestCreateReadNoiseMap(object):

        def test__read_noise_sigma_0__read_noise_map_all_0__image_is_identical_to_input(self):
            simulate_read_noise = ci_data.create_read_noise_map(shape=(3, 3), read_noise=0.0, noise_seed=1)

            assert (simulate_read_noise == np.zeros((3, 3))).all()

        def test__read_noise_sigma_1__read_noise_map_all_non_0__image_has_noise_added(self):
            simulate_read_noise = ci_data.create_read_noise_map(shape=(3, 3), read_noise=1.0, noise_seed=1)

            # Use seed to give us a known read noises map we'll test for

            assert simulate_read_noise == pytest.approx(np.array([[1.62, -0.61, -0.53],
                                                                  [-1.07, 0.87, -2.30],
                                                                  [1.74, -0.76, 0.32]]), 1e-2)


class TestCIMask(object):
    class TestMaskRemoveRegions:

        def test__remove_one_region(self):
            mask = ci_data.CIMask.create(shape=(3, 3), frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                         ci_pattern=MockPattern(), regions=[(0, 3, 2, 3)])

            assert (mask == np.array([[False, False, True],
                                      [False, False, True],
                                      [False, False, True]])).all()

        def test__remove_two_regions(self):
            mask = ci_data.CIMask.create(shape=(3, 3), frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                         ci_pattern=MockPattern(), regions=[(0, 3, 2, 3), (0, 2, 0, 2)])

            assert (mask == np.array([[True, True, True],
                                      [True, True, True],
                                      [False, False, True]])).all()

    class TestCosmicRayMask:

        def test__cosmic_ray_mask_included_in_total_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = ci_data.CIMask.create(shape=(3, 3), frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                         ci_pattern=MockPattern(), cosmic_rays=cosmic_rays)

            assert (mask == np.array([[False, False, False],
                                      [False, True, False],
                                      [False, False, False]])).all()

        def test__cosmic_ray_includes_trail_regions__and_a_mask_region(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = ci_data.CIMask.create(shape=(3, 3), frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                         ci_pattern=MockPattern(), regions=[(0, 1, 0, 1)], cosmic_rays=cosmic_rays,
                                         cr_parallel=1, cr_serial=1)

            assert (mask == np.array([[True, False, False],
                                      [False, True, True],
                                      [False, True, False]])).all()

    class TestMaskCosmicsBottomLeftGeometry:

        def test__mask_one_cosmic_ray_with_parallel_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = ci_data.CIMask.create(shape=(3, 3), frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                         ci_pattern=MockPattern(), cosmic_rays=cosmic_rays, cr_parallel=1)

            assert (mask == np.array([[False, False, False],
                                      [False, True, False],
                                      [False, True, False]])).all()

        def test__mask_one_cosmic_ray_with_longer_parallel_mask(self):
            cosmic_rays = np.array([[False, True, False],
                                    [False, False, False],
                                    [False, False, False]])

            mask = ci_data.CIMask.create(shape=(3, 3), frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                         ci_pattern=MockPattern(), cosmic_rays=cosmic_rays, cr_parallel=2)

            assert (mask == np.array([[False, True, False],
                                      [False, True, False],
                                      [False, True, False]])).all()

        def test__mask_one_cosmic_ray_with_serial_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = ci_data.CIMask.create(shape=(3, 3), frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                         ci_pattern=MockPattern(), cosmic_rays=cosmic_rays, cr_serial=1)

            assert (mask == np.array([[False, False, False],
                                      [False, True, True],
                                      [False, False, False]])).all()

        def test__mask_one_cosmic_ray_with_longer_serial_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [True, False, False],
                                    [False, False, False]])

            mask = ci_data.CIMask.create(shape=(3, 3), frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                         ci_pattern=MockPattern(), cosmic_rays=cosmic_rays, cr_serial=2)

            assert (mask == np.array([[False, False, False],
                                      [True, True, True],
                                      [False, False, False]])).all()

        def test__mask_one_cosmic_ray_with_diagonal_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = ci_data.CIMask.create(shape=(3, 3), frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                         ci_pattern=MockPattern(), cosmic_rays=cosmic_rays, cr_diagonal=1)

            assert (mask == np.array([[False, False, False],
                                      [False, True, True],
                                      [False, True, True]])).all()

        def test__mask_one_cosmic_ray_with_bigger_diagonal_mask(self):
            cosmic_rays = np.array([[False, False, False, False],
                                    [False, True, False, False],
                                    [False, False, False, False],
                                    [False, False, False, False]])

            mask = ci_data.CIMask.create(shape=(4, 4), frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                         ci_pattern=MockPattern(), cosmic_rays=cosmic_rays, cr_diagonal=2)

            assert (mask == np.array([[False, False, False, False],
                                      [False, True, True, True],
                                      [False, True, True, True],
                                      [False, True, True, True]])).all()

    class TestMaskCosmicsBottomRightGeometry:

        def test__mask_one_cosmic_ray_with_parallel_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = ci_data.CIMask.create(shape=(3, 3), frame_geometry=ci_frame.CIQuadGeometryEuclidBR(),
                                         ci_pattern=MockPattern(), cosmic_rays=cosmic_rays, cr_parallel=1)

            assert (mask == np.array([[False, False, False],
                                      [False, True, False],
                                      [False, True, False]])).all()

        def test__mask_one_cosmic_ray_with_longer_parallel_mask(self):
            cosmic_rays = np.array([[False, True, False],
                                    [False, False, False],
                                    [False, False, False]])

            mask = ci_data.CIMask.create(shape=(3, 3), frame_geometry=ci_frame.CIQuadGeometryEuclidBR(),
                                         ci_pattern=MockPattern(), cosmic_rays=cosmic_rays, cr_parallel=2)

            assert (mask == np.array([[False, True, False],
                                      [False, True, False],
                                      [False, True, False]])).all()

        def test__mask_one_cosmic_ray_with_serial_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = ci_data.CIMask.create(shape=(3, 3), frame_geometry=ci_frame.CIQuadGeometryEuclidBR(),
                                         ci_pattern=MockPattern(), cosmic_rays=cosmic_rays, cr_serial=1)

            assert (mask == np.array([[False, False, False],
                                      [True, True, False],
                                      [False, False, False]])).all()

        def test__mask_one_cosmic_ray_with_longer_serial_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, False, True],
                                    [False, False, False]])

            mask = ci_data.CIMask.create(shape=(3, 3), frame_geometry=ci_frame.CIQuadGeometryEuclidBR(),
                                         ci_pattern=MockPattern(), cosmic_rays=cosmic_rays, cr_serial=2)

            assert (mask == np.array([[False, False, False],
                                      [True, True, True],
                                      [False, False, False]])).all()

        def test__mask_one_cosmic_ray_with_diagonal_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = ci_data.CIMask.create(shape=(3, 3), frame_geometry=ci_frame.CIQuadGeometryEuclidBR(),
                                         ci_pattern=MockPattern(), cosmic_rays=cosmic_rays, cr_diagonal=1)

            assert (mask == np.array([[False, False, False],
                                      [True, True, False],
                                      [True, True, False]])).all()

        def test__mask_one_cosmic_ray_with_bigger_diagonal_mask(self):
            cosmic_rays = np.array([[False, False, False, False],
                                    [False, False, False, True],
                                    [False, False, False, False],
                                    [False, False, False, False]])

            mask = ci_data.CIMask.create(shape=(4, 4), frame_geometry=ci_frame.CIQuadGeometryEuclidBR(),
                                         ci_pattern=MockPattern(), cosmic_rays=cosmic_rays, cr_diagonal=2)

            assert (mask == np.array([[False, False, False, False],
                                      [False, True, True, True],
                                      [False, True, True, True],
                                      [False, True, True, True]])).all()

    class TestMaskCosmicsTopLeftGeometry:

        def test__mask_one_cosmic_ray_with_parallel_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = ci_data.CIMask.create(shape=(3, 3), frame_geometry=ci_frame.CIQuadGeometryEuclidTL(),
                                         ci_pattern=MockPattern(), cosmic_rays=cosmic_rays, cr_parallel=1)

            assert (mask == np.array([[False, True, False],
                                      [False, True, False],
                                      [False, False, False]])).all()

        def test__mask_one_cosmic_ray_with_longer_parallel_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, False, False],
                                    [False, True, False]])

            mask = ci_data.CIMask.create(shape=(3, 3), frame_geometry=ci_frame.CIQuadGeometryEuclidTL(),
                                         ci_pattern=MockPattern(), cosmic_rays=cosmic_rays, cr_parallel=2)

            assert (mask == np.array([[False, True, False],
                                      [False, True, False],
                                      [False, True, False]])).all()

        def test__mask_one_cosmic_ray_with_serial_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = ci_data.CIMask.create(shape=(3, 3), frame_geometry=ci_frame.CIQuadGeometryEuclidTL(),
                                         ci_pattern=MockPattern(), cosmic_rays=cosmic_rays, cr_serial=1)

            assert (mask == np.array([[False, False, False],
                                      [False, True, True],
                                      [False, False, False]])).all()

        def test__mask_one_cosmic_ray_with_longer_serial_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [True, False, False],
                                    [False, False, False]])

            mask = ci_data.CIMask.create(shape=(3, 3), frame_geometry=ci_frame.CIQuadGeometryEuclidTL(),
                                         ci_pattern=MockPattern(), cosmic_rays=cosmic_rays, cr_serial=2)

            assert (mask == np.array([[False, False, False],
                                      [True, True, True],
                                      [False, False, False]])).all()

        def test__mask_one_cosmic_ray_with_diagonal_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = ci_data.CIMask.create(shape=(3, 3), frame_geometry=ci_frame.CIQuadGeometryEuclidTL(),
                                         ci_pattern=MockPattern(), cosmic_rays=cosmic_rays, cr_diagonal=1)

            assert (mask == np.array([[False, True, True],
                                      [False, True, True],
                                      [False, False, False]])).all()

        def test__mask_one_cosmic_ray_with_bigger_diagonal_mask(self):
            cosmic_rays = np.array([[False, False, False, False],
                                    [False, False, False, False],
                                    [False, False, False, False],
                                    [False, True, False, False]])

            mask = ci_data.CIMask.create(shape=(4, 4), frame_geometry=ci_frame.CIQuadGeometryEuclidTL(),
                                         ci_pattern=MockPattern(), cosmic_rays=cosmic_rays, cr_diagonal=2)

            assert (mask == np.array([[False, False, False, False],
                                      [False, True, True, True],
                                      [False, True, True, True],
                                      [False, True, True, True]])).all()

    class TestMaskCosmicsTopRightGeometry:

        def test__mask_one_cosmic_ray_with_parallel_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = ci_data.CIMask.create(shape=(3, 3), frame_geometry=ci_frame.CIQuadGeometryEuclidTR(),
                                         ci_pattern=MockPattern(), cosmic_rays=cosmic_rays, cr_parallel=1)

            assert (mask == np.array([[False, True, False],
                                      [False, True, False],
                                      [False, False, False]])).all()

        def test__mask_one_cosmic_ray_with_longer_parallel_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, False, False],
                                    [False, True, False]])

            mask = ci_data.CIMask.create(shape=(3, 3), frame_geometry=ci_frame.CIQuadGeometryEuclidTR(),
                                         ci_pattern=MockPattern(), cosmic_rays=cosmic_rays, cr_parallel=2)
            assert (mask == np.array([[False, True, False],
                                      [False, True, False],
                                      [False, True, False]])).all()

        def test__mask_one_cosmic_ray_with_serial_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = ci_data.CIMask.create(shape=(3, 3), frame_geometry=ci_frame.CIQuadGeometryEuclidTR(),
                                         ci_pattern=MockPattern(), cosmic_rays=cosmic_rays, cr_serial=1)

            assert (mask == np.array([[False, False, False],
                                      [True, True, False],
                                      [False, False, False]])).all()

        def test__mask_one_cosmic_ray_with_longer_serial_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, False, True],
                                    [False, False, False]])

            mask = ci_data.CIMask.create(shape=(3, 3), frame_geometry=ci_frame.CIQuadGeometryEuclidTR(),
                                         ci_pattern=MockPattern(), cosmic_rays=cosmic_rays, cr_serial=2)

            assert (mask == np.array([[False, False, False],
                                      [True, True, True],
                                      [False, False, False]])).all()

        def test__mask_one_cosmic_ray_with_diagonal_mask(self):
            cosmic_rays = np.array([[False, False, False],
                                    [False, True, False],
                                    [False, False, False]])

            mask = ci_data.CIMask.create(shape=(3, 3), frame_geometry=ci_frame.CIQuadGeometryEuclidTR(),
                                         ci_pattern=MockPattern(), cosmic_rays=cosmic_rays, cr_diagonal=1)

            assert (mask == np.array([[True, True, False],
                                      [True, True, False],
                                      [False, False, False]])).all()

        def test__mask_one_cosmic_ray_with_bigger_diagonal_mask(self):
            cosmic_rays = np.array([[False, False, False, False],
                                    [False, False, False, False],
                                    [False, False, False, False],
                                    [False, False, False, True]])

            mask = ci_data.CIMask.create(shape=(4, 4), frame_geometry=ci_frame.CIQuadGeometryEuclidTR(),
                                         ci_pattern=MockPattern(), cosmic_rays=cosmic_rays, cr_diagonal=2)

            assert (mask == np.array([[False, False, False, False],
                                      [False, True, True, True],
                                      [False, True, True, True],
                                      [False, True, True, True]])).all()


class TestCIPreCTI(object):
    class TestConstructor:

        def test__simple_case__sets_up_correctly(self):
            ci_pre_cti = np.array([[10.0, 10.0, 10.0],
                                   [2.0, 2.0, 2.0],
                                   [8.0, 12.0, 10.0]])
            pattern = ci_pattern.CIPatternUniform(normalization=1.0,
                                                  regions=[(0, 1, 0, 1)])

            ci_pre_cti = ci_data.CIPreCTI(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                          array=ci_pre_cti, ci_pattern=pattern)

            assert type(ci_pre_cti.frame_geometry) == ci_frame.CIQuadGeometryEuclidBL
            assert (ci_pre_cti == np.array([[10.0, 10.0, 10.0],
                                            [2.0, 2.0, 2.0],
                                            [8.0, 12.0, 10.0]])).all()
            assert ci_pre_cti.ci_pattern.normalization == pattern.normalization == 1.0
            assert ci_pre_cti.ci_pattern.regions == pattern.regions == [(0, 1, 0, 1)]

    class TestSetupCIPostCTI:

        def test__simple_case__sets_up_post_cti_correctly(self, arctic_both, params_both):
            pattern = ci_pattern.CIPatternUniform(normalization=10.0, regions=[(2, 3, 2, 3)])

            ci_pre_cti = np.zeros((5, 5))
            ci_pre_cti[2, 2] = 10.0
            ci_pre_cti = ci_data.CIPreCTI(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                          array=ci_pre_cti, ci_pattern=pattern)

            ci_post_cti = ci_pre_cti.create_ci_post_cti(params_both, arctic_both)

            image_difference = ci_post_cti - ci_pre_cti

            assert (image_difference[2, 2] < 0.0).all()  # dot loses charge
            assert (image_difference[3:5, 2] > 0.0).all()  # parallel trail behind dot
            assert (image_difference[2, 3:5] > 0.0).all()  # serial trail to right of dot


class TestCIPreCTIFast(object):
    class TestConstructor:

        def test__simple_case__sets_up_correctly(self):
            ci_pre_cti = np.array([[10.0, 10.0, 10.0],
                                   [2.0, 2.0, 2.0],
                                   [8.0, 12.0, 10.0]])

            pattern = ci_pattern.CIPatternUniformFast(normalization=1.0,
                                                      regions=[(0, 1, 0, 1)])

            ci_pre_cti = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                              array=ci_pre_cti, ci_pattern=pattern)

            assert type(ci_pre_cti.frame_geometry) == ci_frame.CIQuadGeometryEuclidBL
            assert (ci_pre_cti == np.array([[10.0, 10.0, 10.0],
                                            [2.0, 2.0, 2.0],
                                            [8.0, 12.0, 10.0]])).all()
            assert ci_pre_cti.ci_pattern.normalization == pattern.normalization == 1.0
            assert ci_pre_cti.ci_pattern.regions == pattern.regions == [(0, 1, 0, 1)]

    class TestSetupFastColumn:

        def test__1_ci_region_with_two_pixels(self):
            pattern = ci_pattern.CIPatternUniformFast(normalization=10.0, regions=[(0, 2, 0, 1)])

            ci_pre_cti = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                              array=np.ones((3, 3)), ci_pattern=pattern)

            assert (ci_pre_cti.fast_column_pre_cti == np.array([[10.0],
                                                                [10.0],
                                                                [0.0]])).all()

        def test__2_ci_regions_with_one_pixel_each(self):
            pattern = ci_pattern.CIPatternUniformFast(normalization=10.0, regions=[(0, 1, 0, 3), (2, 3, 0, 3)])

            ci_pre_cti = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                              array=np.ones((3, 3)), ci_pattern=pattern)

            assert (ci_pre_cti.fast_column_pre_cti == np.array([[10.0],
                                                                [0.0],
                                                                [10.0]])).all()

        def test__top_half_geometry_so_must_include_a_rotation(self):
            pattern = ci_pattern.CIPatternUniformFast(normalization=10.0, regions=[(0, 1, 0, 3)])

            ci_pre_cti = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidTL(),
                                              array=np.ones((3, 3)), ci_pattern=pattern)

            assert (ci_pre_cti.fast_column_pre_cti == np.array([[0.0],
                                                                [0.0],
                                                                [10.0]])).all()

        def test__rectangular_image__adding_rows_makes_fast_column_longer(self):
            pattern = ci_pattern.CIPatternUniformFast(
                normalization=10.0, regions=[(0, 2, 0, 3)])

            ci_pre_cti = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                              array=np.ones((5, 3)), ci_pattern=pattern)

            assert (ci_pre_cti.fast_column_pre_cti == np.array([[10.0],
                                                                [10.0],
                                                                [0.0],
                                                                [0.0],
                                                                [0.0]])).all()

        def test__rectangular_iamge__adding_columns_does_nothing(self):
            pattern = ci_pattern.CIPatternUniformFast(
                normalization=10.0, regions=[(0, 2, 0, 3)])

            ci_pre_cti = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                              array=np.ones((3, 3)), ci_pattern=pattern)

            assert (ci_pre_cti.fast_column_pre_cti == np.array([[10.0],
                                                                [10.0],
                                                                [0.0]])).all()

    class TestAddCtiToFastColumn:

        def test__simple_case(self, arctic_parallel, params_parallel):
            pattern = ci_pattern.CIPatternUniformFast(
                normalization=10.0, regions=[(0, 1, 0, 3)])

            ci_pre_cti = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                              array=np.ones((3, 3)), ci_pattern=pattern)

            fast_column_post_cti = ci_pre_cti.compute_fast_column_post_cti(params_parallel, arctic_parallel)

            image_difference = fast_column_post_cti - ci_pre_cti.fast_column_pre_cti

            assert image_difference[0, 0] < 0.0  # Charge was here, loses due to captures
            assert image_difference[1:-1, 0] > 0.0  # Was empty before, trails go here

    class TestMapFastColumnToPostCTIImage:

        def test__image_3x3__no_rotation__one_fast_column(self):
            pattern = ci_pattern.CIPatternUniformFast(normalization=1.0,
                                                      regions=[(0, 2, 0, 2)])

            ci_pre_cti = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                              array=np.ones((3, 3)), ci_pattern=pattern)

            fast_column_post_cti = np.array([[10.0],
                                             [5.0],
                                             [1.0]])

            ci_post_cti = ci_pre_cti.map_fast_column_post_cti_to_image(fast_column_post_cti)

            assert (ci_post_cti == np.array([[10.0, 10.0, 10.0],
                                             [5.0, 5.0, 5.0],
                                             [1.0, 1.0, 1.0]])).all()

        def test__image_3x4__no_rotation__one_fast_column(self):
            pattern = ci_pattern.CIPatternUniformFast(normalization=1.0,
                                                      regions=[(0, 2, 0, 2)])

            ci_pre_cti = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                              array=np.ones((3, 4)), ci_pattern=pattern)

            fast_column_post_cti = np.array([[10.0],
                                             [5.0],
                                             [1.0]])

            ci_post_cti = ci_pre_cti.map_fast_column_post_cti_to_image(fast_column_post_cti)

            assert (ci_post_cti == np.array([[10.0, 10.0, 10.0, 10.0],
                                             [5.0, 5.0, 5.0, 5.0],
                                             [1.0, 1.0, 1.0, 1.0]])).all()

        def test__image_4x3__no_rotation__one_fast_column(self):
            pattern = ci_pattern.CIPatternUniformFast(normalization=1.0,
                                                      regions=[(0, 2, 0, 2)])

            ci_pre_cti = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                              array=np.ones((4, 3)), ci_pattern=pattern)

            fast_column_post_cti = np.array([[10.0],
                                             [5.0],
                                             [1.0],
                                             [0.0]])

            ci_post_cti = ci_pre_cti.map_fast_column_post_cti_to_image(fast_column_post_cti)

            assert (ci_post_cti == np.array([[10.0, 10.0, 10.0],
                                             [5.0, 5.0, 5.0],
                                             [1.0, 1.0, 1.0],
                                             [0.0, 0.0, 0.0]])).all()

        def test__image_3x3__rotation_flip_horizontal(self):
            pattern = ci_pattern.CIPatternUniformFast(normalization=1.0,
                                                      regions=[(0, 2, 0, 2)])

            ci_pre_cti = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidTL(),
                                              array=np.ones((3, 3)), ci_pattern=pattern)

            fast_column_post_cti = np.array([[10.0],
                                             [5.0],
                                             [1.0]])

            ci_post_cti = ci_pre_cti.map_fast_column_post_cti_to_image(fast_column_post_cti)

            assert (ci_post_cti == np.array([[1.0, 1.0, 1.0],
                                             [5.0, 5.0, 5.0],
                                             [10.0, 10.0, 10.0]])).all()

        def test__image_3x4__rotation_flip_horizontal(self):
            pattern = ci_pattern.CIPatternUniformFast(normalization=1.0,
                                                      regions=[(0, 2, 0, 2)])

            ci_pre_cti = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidTL(),
                                              array=np.ones((3, 4)), ci_pattern=pattern)

            fast_column_post_cti = np.array([[10.0],
                                             [5.0],
                                             [1.0]])

            ci_post_cti = ci_pre_cti.map_fast_column_post_cti_to_image(fast_column_post_cti)

            assert (ci_post_cti == np.array([[1.0, 1.0, 1.0, 1.0],
                                             [5.0, 5.0, 5.0, 5.0],
                                             [10.0, 10.0, 10.0, 10.0]])).all()

        def test__image_4x3__rotation_flip_horizontal(self):
            pattern = ci_pattern.CIPatternUniformFast(normalization=1.0,
                                                      regions=[(0, 2, 0, 2)])

            ci_pre_cti = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidTL(),
                                              array=np.ones((4, 3)), ci_pattern=pattern)

            fast_column_post_cti = np.array([[10.0],
                                             [5.0],
                                             [1.0],
                                             [0.0]])

            ci_post_cti = ci_pre_cti.map_fast_column_post_cti_to_image(fast_column_post_cti)

            assert (ci_post_cti == np.array([[0.0, 0.0, 0.0],
                                             [1.0, 1.0, 1.0],
                                             [5.0, 5.0, 5.0],
                                             [10.0, 10.0, 10.0]])).all()

    class TestSetupFastRow:

        def test__1_ci_region_covers_entire_row(self):
            pattern = ci_pattern.CIPatternUniformFast(
                normalization=10.0, regions=[(0, 3, 0, 3)])

            ci_pre_cti = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                              array=np.ones((3, 3)), ci_pattern=pattern)

            assert (ci_pre_cti.fast_row_pre_cti == np.array([[10.0],
                                                             [10.0],
                                                             [10.0]])).all()

        def test__1_ci_region_covers_first_two_pixels(self):
            pattern = ci_pattern.CIPatternUniformFast(
                normalization=10.0, regions=[(0, 3, 0, 2)])

            ci_pre_cti = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                              array=np.ones((3, 3)), ci_pattern=pattern)

            assert (ci_pre_cti.fast_row_pre_cti == np.array([[10.0],
                                                             [10.0],
                                                             [0.0]])).all()

        def test__right_hand_geometry_so_must_include_rotation(self):
            pattern = ci_pattern.CIPatternUniformFast(
                normalization=10.0, regions=[(0, 3, 0, 1)])

            ci_pre_cti = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidTR(),
                                              array=np.ones((3, 3)), ci_pattern=pattern)

            assert (ci_pre_cti.fast_row_pre_cti == np.array([[0.0],
                                                             [0.0],
                                                             [10.0]])).all()

        def test__rectangular_image_4x3__adding_rows_does_nothing(self):
            pattern = ci_pattern.CIPatternUniformFast(
                normalization=10.0, regions=[(0, 5, 0, 2)])

            ci_pre_cti = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                              array=np.ones((4, 3)), ci_pattern=pattern)

            assert (ci_pre_cti.fast_row_pre_cti == np.array([[10.0],
                                                             [10.0],
                                                             [0.0]])).all()

        def test__rectangular_image_3x4__adding_rows_makes_fast_rows_longer(self):
            pattern = ci_pattern.CIPatternUniformFast(
                normalization=10.0, regions=[(0, 3, 0, 2)])

            ci_pre_cti = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                              array=np.ones((3, 5)), ci_pattern=pattern)

            assert (ci_pre_cti.fast_row_pre_cti == np.array([[10.0],
                                                             [10.0],
                                                             [0.0],
                                                             [0.0],
                                                             [0.0]])).all()

    class TestAddCTIToFastRow:

        def test__simple_case__one_image(self, arctic_serial, params_serial):
            pattern = ci_pattern.CIPatternUniformFast(
                normalization=10.0, regions=[(0, 3, 0, 1)])

            ci_pre_cti = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                              array=np.ones((3, 3)), ci_pattern=pattern)

            fast_row_post_cti = ci_pre_cti.compute_fast_row_post_cti(params_serial, arctic_serial)

            image_difference = fast_row_post_cti - ci_pre_cti.fast_row_pre_cti

            assert image_difference[0, 0] < 0.0  # Charge was here, loses due to captures
            assert image_difference[1:-1, 0] > 0.0  # Was empty before, trails go here

    class TestMapFastRowToPostCTIImage:

        def test__image_3x3__no_rotation(self):
            pattern = ci_pattern.CIPatternUniformFast(normalization=1.0,
                                                      regions=[(0, 2, 0, 2)])

            ci_pre_cti = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                              array=np.ones((3, 3)), ci_pattern=pattern)

            fast_row_post_cti = np.array([[10.0],
                                          [10.0],
                                          [0.0]])

            ci_post_cti = ci_pre_cti.map_fast_row_post_cti_to_image(fast_row_post_cti)

            assert (ci_post_cti == np.array([[10.0, 10.0, 0.0],
                                             [10.0, 10.0, 0.0],
                                             [10.0, 10.0, 0.0]])).all()

        def test__image_3x4__no_rotation(self):
            pattern = ci_pattern.CIPatternUniformFast(normalization=1.0,
                                                      regions=[(0, 2, 0, 2)])

            ci_pre_cti = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                              array=np.ones((3, 4)), ci_pattern=pattern)

            fast_row_post_cti = np.array([[10.0],
                                          [10.0],
                                          [0.0],
                                          [0.0]])

            ci_post_cti = ci_pre_cti.map_fast_row_post_cti_to_image(fast_row_post_cti)

            assert (ci_post_cti == np.array([[10.0, 10.0, 0.0, 0.0],
                                             [10.0, 10.0, 0.0, 0.0],
                                             [10.0, 10.0, 0.0, 0.0]])).all()

        def test__image_4x3__no_rotation(self):
            pattern = ci_pattern.CIPatternUniformFast(normalization=1.0,
                                                      regions=[(0, 2, 0, 2)])

            ci_pre_cti = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                              array=np.ones((4, 3)), ci_pattern=pattern)

            fast_row_post_cti = np.array([[10.0],
                                          [10.0],
                                          [0.0]])

            ci_post_cti = ci_pre_cti.map_fast_row_post_cti_to_image(fast_row_post_cti)

            assert (ci_post_cti == np.array([[10.0, 10.0, 0.0],
                                             [10.0, 10.0, 0.0],
                                             [10.0, 10.0, 0.0],
                                             [10.0, 10.0, 0.0]])).all()

        def test__image_3x3__rotation_flip_horizontal(self):
            pattern = ci_pattern.CIPatternUniformFast(normalization=1.0,
                                                      regions=[(0, 2, 0, 2)])

            ci_pre_cti = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidBR(),
                                              array=np.ones((3, 3)), ci_pattern=pattern)

            fast_row_post_cti = np.array([[10.0],
                                          [10.0],
                                          [0.0]])

            ci_post_cti = ci_pre_cti.map_fast_row_post_cti_to_image(fast_row_post_cti)

            assert (ci_post_cti == np.array([[0.0, 10.0, 10.0],
                                             [0.0, 10.0, 10.0],
                                             [0.0, 10.0, 10.0]])).all()

        def test__image_3x4__rotation_flip_horizontal(self):
            pattern = ci_pattern.CIPatternUniformFast(normalization=1.0,
                                                      regions=[(0, 2, 0, 2)])

            ci_pre_cti = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidBR(),
                                              array=np.ones((3, 4)), ci_pattern=pattern)

            fast_row_post_cti = np.array([[10.0],
                                          [10.0],
                                          [0.0],
                                          [0.0]])

            ci_post_cti = ci_pre_cti.map_fast_row_post_cti_to_image(fast_row_post_cti)

            assert (ci_post_cti == np.array([[0.0, 0.0, 10.0, 10.0],
                                             [0.0, 0.0, 10.0, 10.0],
                                             [0.0, 0.0, 10.0, 10.0]])).all()

        def test__image_4x3__rotation_flip_horizontal(self):
            pattern = ci_pattern.CIPatternUniformFast(normalization=1.0,
                                                      regions=[(0, 2, 0, 2)])

            ci_pre_cti = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidBR(),
                                              array=np.ones((4, 3)), ci_pattern=pattern)

            fast_row_post_cti = np.array([[10.0],
                                          [10.0],
                                          [0.0]])

            ci_post_cti = ci_pre_cti.map_fast_row_post_cti_to_image(fast_row_post_cti)

            assert (ci_post_cti == np.array([[0.0, 10.0, 10.0],
                                             [0.0, 10.0, 10.0],
                                             [0.0, 10.0, 10.0],
                                             [0.0, 10.0, 10.0]])).all()

    class TestComputePostCTIImage:

        def test__parallel_only__compare_to_non_fast_image(self, arctic_parallel, params_parallel):
            pattern = ci_pattern.CIPatternUniform(normalization=10.0,
                                                  regions=[(2, 3, 0, 5)])
            pattern_fast = ci_pattern.CIPatternUniformFast(
                normalization=10.0, regions=[(2, 3, 0, 5)])

            ci_pre_cti = np.zeros((5, 5))
            ci_pre_cti[2, :] = 10.0
            ci_pre_cti_normal = ci_data.CIPreCTI(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                                 array=ci_pre_cti, ci_pattern=pattern)
            ci_post_cti_normal = ci_pre_cti_normal.add_cti_to_image(params_parallel, arctic_parallel)

            ci_pre_cti_fast = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                                   array=ci_pre_cti, ci_pattern=pattern_fast)
            ci_post_cti_fast = ci_pre_cti_fast.create_ci_post_cti(params_parallel, arctic_parallel)

            assert (ci_post_cti_normal == ci_post_cti_fast).all()

        def test__serial_only__compare_to_non_fast_image(self, arctic_serial, params_serial):
            pattern = ci_pattern.CIPatternUniform(normalization=10.0,
                                                  regions=[(1, 5, 2, 3)])
            pattern_fast = ci_pattern.CIPatternUniformFast(
                normalization=10.0, regions=[(1, 5, 2, 3)])

            ci_pre_cti = np.zeros((5, 5))
            ci_pre_cti[:, 2] = 10.0
            ci_pre_cti_normal = ci_data.CIPreCTI(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                                 array=ci_pre_cti, ci_pattern=pattern)
            ci_post_cti_normal = ci_pre_cti_normal.add_cti_to_image(params_serial, arctic_serial)

            ci_pre_cti_fast = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                                   array=ci_pre_cti, ci_pattern=pattern_fast)
            ci_post_cti_fast = ci_pre_cti_fast.create_ci_post_cti(params_serial, arctic_serial)

            assert (ci_post_cti_normal == ci_post_cti_fast).all()

        def test__parallel_and_serial__raises_error(self, arctic_both, params_both):
            pattern_fast = ci_pattern.CIPatternUniformFast(
                normalization=10.0, regions=[(1, 5, 2, 3)])

            ci_pre_cti = np.zeros((5, 5))
            ci_pre_cti[:, 2] = 10.0
            ci_pre_cti_fast = ci_data.CIPreCTIFast(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(),
                                                   array=ci_pre_cti, ci_pattern=pattern_fast)
            with pytest.raises(exc.CIPreCTIException):
                ci_pre_cti_fast.create_ci_post_cti(params_both, arctic_both)

    class TestCompareFastAndNormal:

        def test__parallel__3x4_1_ci_region(self, arctic_parallel, params_parallel):
            ### SETUP FAST PATTERNS, IMAGES AND PRE CTI IMAGES, AND COMPUTE POST CTI IMAGES ###

            pattern_fast = ci_pattern.CIPatternUniformFast(
                normalization=10.0, regions=[(2, 3, 0, 4)])

            data_fast = ci_data.CIImage(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(), ci_pattern=pattern_fast,
                                        array=10.0 * np.ones((3, 4)))

            ci_pre_cti_fast = data_fast.create_ci_pre_cti()
            ci_post_cti_fast = ci_pre_cti_fast.create_ci_post_cti(params_parallel, arctic_parallel)

            ### SETUP NORMAL PATTERNS, IMAGES AND PRE CTI IMAGES, AND COMPUTE POST CTI IMAGES ###

            pattern_normal = ci_pattern.CIPatternUniform(normalization=10.0,
                                                         regions=[(2, 3, 0, 4)])

            data_normal = ci_data.CIImage(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(), ci_pattern=pattern_normal,
                                          array=10.0 * np.ones((3, 4)))

            ci_pre_cti_normal = data_normal.create_ci_pre_cti()
            ci_post_cti_normal = ci_pre_cti_normal.create_ci_post_cti(params_parallel, arctic_parallel)

            ### COMPARE THE TWO ###

            assert (ci_post_cti_fast == ci_post_cti_normal).all()

        def test__parallel__4x3_1_ci_region(self, arctic_parallel, params_parallel):
            ### SETUP FAST PATTERNS, IMAGES AND PRE CTI IMAGES, AND COMPUTE POST CTI IMAGES ###

            pattern_fast = ci_pattern.CIPatternUniformFast(
                normalization=10.0, regions=[(2, 3, 0, 3)])

            data_fast = ci_data.CIImage(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(), ci_pattern=pattern_fast,
                                        array=10.0 * np.ones((4, 3)))

            ci_pre_cti_fast = data_fast.create_ci_pre_cti()
            ci_post_cti_fast = ci_pre_cti_fast.create_ci_post_cti(params_parallel, arctic_parallel)

            ### SETUP NORMAL PATTERNS, IMAGES AND PRE CTI IMAGES, AND COMPUTE POST CTI IMAGES ###

            pattern_normal = ci_pattern.CIPatternUniform(normalization=10.0,
                                                         regions=[(2, 3, 0, 3)])

            data_normal = ci_data.CIImage(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(), ci_pattern=pattern_normal,
                                          array=10.0 * np.ones((4, 3)))

            ci_pre_cti_normal = data_normal.create_ci_pre_cti()
            ci_post_cti_normal = ci_pre_cti_normal.create_ci_post_cti(params_parallel, arctic_parallel)

            ### COMPARE THE TWO ###

            assert (ci_post_cti_fast == ci_post_cti_normal).all()

        def test__serial__3x4_1_ci_region(self, arctic_serial, params_serial):
            ### SETUP FAST PATTERNS, IMAGES AND PRE CTI IMAGES, AND COMPUTE POST CTI IMAGES ###

            pattern_fast = ci_pattern.CIPatternUniformFast(
                normalization=10.0, regions=[(0, 3, 0, 3)])

            data_fast = ci_data.CIImage(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(), ci_pattern=pattern_fast,
                                        array=10.0 * np.ones((3, 4)))

            ci_pre_cti_fast = data_fast.create_ci_pre_cti()
            ci_post_cti_fast = ci_pre_cti_fast.create_ci_post_cti(params_serial, arctic_serial)

            ### SETUP NORMAL PATTERNS, IMAGES AND PRE CTI IMAGES, AND COMPUTE POST CTI IMAGES ###

            pattern_normal = ci_pattern.CIPatternUniform(normalization=10.0,
                                                         regions=[(0, 3, 0, 3)])

            data_normal = ci_data.CIImage(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(), ci_pattern=pattern_normal,
                                          array=10.0 * np.ones((3, 4)))

            ci_pre_cti_normal = data_normal.create_ci_pre_cti()
            ci_post_cti_normal = ci_pre_cti_normal.create_ci_post_cti(params_serial, arctic_serial)

            ### COMPARE THE TWO ###

            assert (ci_post_cti_fast == ci_post_cti_normal).all()

        def test__serial__4x3_1_ci_region(self, arctic_serial, params_serial):
            ### SETUP FAST PATTERNS, IMAGES AND PRE CTI IMAGES, AND COMPUTE POST CTI IMAGES ###

            pattern_fast = ci_pattern.CIPatternUniformFast(
                normalization=10.0, regions=[(0, 4, 0, 2)])

            data_fast = ci_data.CIImage(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(), ci_pattern=pattern_fast,
                                        array=10.0 * np.ones((4, 3)))

            ci_pre_cti_fast = data_fast.create_ci_pre_cti()
            ci_post_cti_fast = ci_pre_cti_fast.create_ci_post_cti(params_serial, arctic_serial)

            ### SETUP NORMAL PATTERNS, IMAGES AND PRE CTI IMAGES, AND COMPUTE POST CTI IMAGES ###

            pattern_normal = ci_pattern.CIPatternUniform(normalization=10.0,
                                                         regions=[(0, 4, 0, 2)])

            data_normal = ci_data.CIImage(frame_geometry=ci_frame.CIQuadGeometryEuclidBL(), ci_pattern=pattern_normal,
                                          array=10.0 * np.ones((4, 3)))

            ci_pre_cti_normal = data_normal.create_ci_pre_cti()
            ci_post_cti_normal = ci_pre_cti_normal.create_ci_post_cti(params_serial, arctic_serial)

            ### COMPARE THE TWO ###

            assert (ci_post_cti_fast == ci_post_cti_normal).all()
