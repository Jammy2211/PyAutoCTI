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
File: tests/python/CIPattern_test.py

Created on: 02/14/18
Author: James Nightingale
"""

import numpy as np
import pytest

from autocti import exc
from autocti.charge_injection import ci_pattern

class TestCIPatternViaList(object):

    def test__2_uniform_patterns__sets_up_collection(self):
        pattern_collection = ci_pattern.uniform_from_lists(normalizations=[1.0, 3.0], regions=[(1, 2, 3, 4)])

        assert type(pattern_collection[0]) == ci_pattern.CIPatternUniform
        assert pattern_collection[0].normalization == 1.0
        assert pattern_collection[0].regions == [(1, 2, 3, 4)]

        assert type(pattern_collection[1]) == ci_pattern.CIPatternUniform
        assert pattern_collection[1].normalization == 3.0
        assert pattern_collection[1].regions == [(1, 2, 3, 4)]

    def test__2_non_uniform_patterns__sets_up_collection(self):
        pattern_collection = ci_pattern.non_uniform_from_lists(normalizations=[1.0, 3.0], regions=[(1, 2, 3, 4)],
                                                               row_slopes=[1.0, 2.0])

        assert type(pattern_collection[0]) == ci_pattern.CIPatternNonUniform
        assert pattern_collection[0].normalization == 1.0
        assert pattern_collection[0].regions == [(1, 2, 3, 4)]
        assert pattern_collection[0].row_slope == 1.0

        assert type(pattern_collection[0]) == ci_pattern.CIPatternNonUniform
        assert pattern_collection[1].normalization == 3.0
        assert pattern_collection[1].regions == [(1, 2, 3, 4)]
        assert pattern_collection[1].row_slope == 2.0


class TestCIPattern(object):
    class TestConstructor:

        def test__setup_all_attributes_correctly(self):
            pattern = ci_pattern.CIPattern(normalization=1.0, regions=[(1, 2, 3, 4)])

            assert pattern.normalization == 1.0
            assert pattern.regions == [(1, 2, 3, 4)]

    class TestCheckImageDimensions:

        def test__pattern_has_more_rows_than_image__1_region(self):
            pattern = ci_pattern.CIPattern(normalization=1.0, regions=([(0, 3, 0, 1)]))

            with pytest.raises(exc.CIPatternException):
                pattern.check_pattern_is_within_image_dimensions(dimensions=(2, 6))

        def test__pattern_has_more_columns_than_image__1_region(self):
            pattern = ci_pattern.CIPattern(normalization=1.0, regions=([(0, 1, 0, 3)]))

            with pytest.raises(exc.CIPatternException):
                pattern.check_pattern_is_within_image_dimensions(dimensions=(6, 2))

        def test__check_rows_and_columns__2_region(self):
            pattern = ci_pattern.CIPattern(normalization=1.0, regions=([(0, 3, 0, 1), (0, 1, 0, 3)]))

            with pytest.raises(exc.CIPatternException):
                pattern.check_pattern_is_within_image_dimensions(dimensions=(2, 6))

            pattern = ci_pattern.CIPattern(normalization=1.0, regions=([(0, 3, 0, 1), (0, 1, 0, 3)]))

            with pytest.raises(exc.CIPatternException):
                pattern.check_pattern_is_within_image_dimensions(dimensions=(6, 2))

        def test__region_cannot_be_negative(self):
            with pytest.raises(exc.RegionException):
                ci_pattern.CIPattern(normalization=1.0, regions=([(-1, 0, 0, 0)]))

            with pytest.raises(exc.RegionException):
                ci_pattern.CIPattern(normalization=1.0, regions=([(0, -1, 0, 0)]))

            with pytest.raises(exc.RegionException):
                ci_pattern.CIPattern(normalization=1.0, regions=([(0, 0, -1, 0)]))

            with pytest.raises(exc.RegionException):
                ci_pattern.CIPattern(normalization=1.0, regions=([(0, 0, 0, -1)]))


class TestCIPatternUniform(object):
    class TestConstructor:

        def test__setup_all_attributes_correctly(self):
            pattern = ci_pattern.CIPatternUniform(normalization=3.0, regions=[(1, 4, 2, 4)])

            assert pattern.normalization == 3.0
            assert pattern.regions == [(1, 4, 2, 4)]

    class TestComputeCIPreCTI:

        def test__uniform_pattern_1_region_normalization_10__correct_ci_pre_cti(self):
            pattern = ci_pattern.CIPatternUniform(normalization=10.0, regions=[(0, 2, 0, 2)])

            ci_pre_cti = pattern.ci_pre_cti_from_shape(shape=(3, 3))

            assert (ci_pre_cti == np.array([[10.0, 10.0, 0.0],
                                            [10.0, 10.0, 0.0],
                                            [0.0, 0.0, 0.0]])).all()

        def test__pattern_bigger_than_image_dimensions__raises_error(self):
            pattern = ci_pattern.CIPatternUniform(normalization=10.0, regions=[(0, 2, 0, 1)])

            with pytest.raises(exc.CIPatternException):
                pattern.ci_pre_cti_from_shape(shape=(1, 1))


class TestCIPatternNonUniform(object):
    class TestConstructor:

        def test__setup_all_attributes_correctly(self):
            pattern = ci_pattern.CIPatternNonUniform(normalization=5.0, regions=[(2, 3, 3, 4)], row_slope=0.1)

            assert pattern.normalization == 5.0
            assert pattern.regions == [(2, 3, 3, 4)]
            assert pattern.row_slope == 0.1

    class TestMeanChargeInColumn:

        def test__column_values_all_10__no_mask__measures_mean_as_10(self):
            pattern = ci_pattern.CIPatternNonUniform(normalization=10.0, regions=[(0, 1, 0, 1)], row_slope=0.0)

            column = np.array([10.0, 10.0, 10.0])
            column_mask = np.array([False, False, False])

            assert pattern.mean_charge_in_column(column, column_mask) == 10.0

        def test__column_values_non_uniform__mask_omits_2_values__measures_mean_correctly(self):
            pattern = ci_pattern.CIPatternNonUniform(normalization=10.0, regions=[(0, 1, 0, 1)], row_slope=0.0)

            column = np.array([11.0, 8.0, 15.0, 10])
            column_mask = np.array([False, True, True, False])

            assert pattern.mean_charge_in_column(column, column_mask) == 10.5

        def test__column_is_fully_masked__mean_is_none(self):
            pattern = ci_pattern.CIPatternNonUniform(normalization=10.0, regions=[(0, 1, 0, 1)], row_slope=0.0)

            column = np.array([11.0, 8.0, 15.0, 10])
            column_mask = np.array([True, True, True, True])

            assert pattern.mean_charge_in_column(column, column_mask) is None

        def test__column_values_all_different__no_row_non_uniformity__measures_mean_correctly(self):
            pattern = ci_pattern.CIPatternNonUniform(normalization=10.0, regions=[(0, 1, 0, 1)], row_slope=0.0)

            column = np.array([11.0, 8.0, 15.0, 10])
            column_mask = np.array([False, False, False, False])

            assert pattern.mean_charge_in_column(column, column_mask) == 11.0

        def test__column_values_identical_after_row_non_uniformity__measures_mean_as_input_normalization(self):
            pattern = ci_pattern.CIPatternNonUniform(normalization=100.0, regions=[(0, 1, 0, 1)], row_slope=-0.01)

            column = np.array([100.0, 99.3, 98.9])
            column_mask = np.array([False, False, False])

            assert pattern.mean_charge_in_column(column, column_mask) == pytest.approx(100., 1e-4)

        def test__column_values_10_above_normliazation_of_100_after_row_non_uniformity__measures_mean_as_110(self):
            pattern = ci_pattern.CIPatternNonUniform(normalization=100.0, regions=[(0, 1, 0, 1)], row_slope=-0.01)

            column = np.array([110.0, 109.3, 108.9])
            column_mask = np.array([False, False, False])

            assert pattern.mean_charge_in_column(column, column_mask) == pytest.approx(110., 1e-4)

        def test__same_as_above_but_normalization_is_500(self):
            pattern_sim = ci_pattern.CIPatternNonUniform(normalization=500.0, regions=[(0, 1, 0, 1)],
                                                         row_slope=1.0)
            column = pattern_sim.generate_column(size=10, normalization=500.0)

            pattern = ci_pattern.CIPatternNonUniform(normalization=500.0, regions=[(0, 1, 0, 1)], row_slope=1.0)

            column_mask = np.array([False] * 10)

            assert pattern.mean_charge_in_column(column, column_mask) == 500.0

        def test__subtract_30_from_the_column_with_normalization_500__mean_is_470_due_to_subtraction(self):
            pattern_sim = ci_pattern.CIPatternNonUniform(normalization=500.0, regions=[(0, 1, 0, 1)],
                                                         row_slope=0.0)

            column = pattern_sim.generate_column(size=10, normalization=500.0)

            pattern = ci_pattern.CIPatternNonUniform(normalization=500.0, regions=[(0, 1, 0, 1)], row_slope=0.0)

            column -= 30.0
            column_mask = np.array([False] * 10)

            assert pattern.mean_charge_in_column(column, column_mask) == 470.0

    class TestMeanChargeInAllColumns:

        def test__uniform_column__ci_region_is_entire_image(self):
            ci_column = np.array([[10.0],
                                  [10.0],
                                  [10.0]])

            column_mask = np.array([[False],
                                    [False],
                                    [False]])

            pattern = ci_pattern.CIPatternNonUniform(normalization=10.0, regions=[(0, 3, 0, 1)], row_slope=0.0)

            mean_of_columns = pattern.mean_charge_in_all_image_columns(column=ci_column, column_mask=column_mask)

            assert mean_of_columns == [10.0]

        def test__non_uniform_column__ci_region_is_entire_image(self):
            ci_column = np.array([[13.0],
                                  [10.0],
                                  [10.0]])

            column_mask = np.array([[False],
                                    [False],
                                    [False]])

            pattern = ci_pattern.CIPatternNonUniform(normalization=10.0, regions=[(0, 3, 0, 1)], row_slope=0.0)

            mean_of_columns = pattern.mean_charge_in_all_image_columns(column=ci_column, column_mask=column_mask)

            assert mean_of_columns == [11.0]

        def test__non_uniform_column__mask_changes_mean_estimate(self):
            ci_column = np.array([[13.0],
                                  [10.0],
                                  [10.0]])

            column_mask = np.array([[False],
                                    [True],
                                    [False]])

            pattern = ci_pattern.CIPatternNonUniform(normalization=10.0, regions=[(0, 3, 0, 1)], row_slope=0.0)

            mean_of_columns = pattern.mean_charge_in_all_image_columns(column=ci_column, column_mask=column_mask)

            assert mean_of_columns == [11.5]

        def test__non_uniform_column__fully_masked_mean_estimate_is_none(self):
            ci_column = np.array([[13.0],
                                  [10.0],
                                  [10.0]])

            column_mask = np.array([[True],
                                    [True],
                                    [True]])

            pattern = ci_pattern.CIPatternNonUniform(normalization=10.0, regions=[(0, 3, 0, 1)], row_slope=0.0)

            mean_of_columns = pattern.mean_charge_in_all_image_columns(column=ci_column, column_mask=column_mask)

            assert mean_of_columns == [None]

        def test__non_uniform_column__ci_region_is_only_two_entries(self):
            ci_column = np.array([[12.0],
                                  [10.0],
                                  [10.0]])

            column_mask = np.array([[False],
                                    [False],
                                    [False]])

            pattern = ci_pattern.CIPatternNonUniform(normalization=10.0, regions=[(0, 2, 0, 1)], row_slope=0.0)

            mean_of_columns = pattern.mean_charge_in_all_image_columns(column=ci_column, column_mask=column_mask)

            assert mean_of_columns == [11.0]

        def test__uniform_column__three_ci_regions(self):
            ci_column = np.array([[10.0],
                                  [10.0],
                                  [10.0],
                                  [0.0],
                                  [10.0],
                                  [10.0],
                                  [10.0],
                                  [0.0],
                                  [10.0],
                                  [10.0],
                                  [10.0]])

            column_mask = np.array([[False],
                                    [False],
                                    [False],
                                    [False],
                                    [False],
                                    [False],
                                    [False],
                                    [False],
                                    [False],
                                    [False],
                                    [False]])

            pattern = ci_pattern.CIPatternNonUniform(normalization=10.0,
                                                     regions=[(0, 3, 0, 1), (4, 7, 0, 1), (8, 11, 0, 1)],
                                                     row_slope=0.0)

            mean_of_columns = pattern.mean_charge_in_all_image_columns(column=ci_column, column_mask=column_mask)

            assert mean_of_columns == [10.0, 10.0, 10.0]

        def test__non_uniform_column__three_ci_regions(self):
            ci_column = np.array([[13.0],
                                  [10.0],
                                  [10.0],
                                  [0.0],
                                  [10.0],
                                  [7.0],
                                  [10.0],
                                  [0.0],
                                  [16.0],
                                  [16.0],
                                  [7.0]])

            column_mask = np.array([[False],
                                    [False],
                                    [False],
                                    [False],
                                    [False],
                                    [False],
                                    [False],
                                    [False],
                                    [False],
                                    [False],
                                    [False]])

            pattern = ci_pattern.CIPatternNonUniform(normalization=10.0,
                                                     regions=[(0, 3, 0, 1), (4, 7, 0, 1), (8, 11, 0, 1)],
                                                     row_slope=0.0)

            mean_of_columns = pattern.mean_charge_in_all_image_columns(column=ci_column, column_mask=column_mask)

            assert mean_of_columns == [11.0, 9.0, 13.0]

        def test__non_uniform_column__three_ci_regions__mask_changes_each_mean(self):
            ci_column = np.array([[13.0],
                                  [10.0],
                                  [10.0],
                                  [0.0],
                                  [10.0],
                                  [7.0],
                                  [10.0],
                                  [0.0],
                                  [16.0],
                                  [16.0],
                                  [7.0]])

            column_mask = np.array([[False],
                                    [True],
                                    [False],
                                    [False],
                                    [True],
                                    [False],
                                    [True],
                                    [False],
                                    [False],
                                    [True],
                                    [False]])

            pattern = ci_pattern.CIPatternNonUniform(normalization=10.0,
                                                     regions=[(0, 3, 0, 1), (4, 7, 0, 1), (8, 11, 0, 1)],
                                                     row_slope=0.0)

            mean_of_columns = pattern.mean_charge_in_all_image_columns(column=ci_column, column_mask=column_mask)

            assert mean_of_columns == [11.5, 7.0, 11.5]

        def test__non_uniform_column__three_ci_regions__fully_masked_includes_a_none(self):
            ci_column = np.array([[13.0],
                                  [10.0],
                                  [10.0],
                                  [0.0],
                                  [10.0],
                                  [7.0],
                                  [10.0],
                                  [0.0],
                                  [16.0],
                                  [16.0],
                                  [7.0]])

            column_mask = np.array([[False],
                                    [True],
                                    [False],
                                    [False],
                                    [True],
                                    [True],
                                    [True],
                                    [False],
                                    [False],
                                    [True],
                                    [False]])

            pattern = ci_pattern.CIPatternNonUniform(normalization=10.0,
                                                     regions=[(0, 3, 0, 1), (4, 7, 0, 1), (8, 11, 0, 1)],
                                                     row_slope=0.0)

            mean_of_columns = pattern.mean_charge_in_all_image_columns(column=ci_column, column_mask=column_mask)

            assert mean_of_columns == [11.5, None, 11.5]

        def test__non_uniform_column__three_ci_regions__include_row_non_uniformity(self):
            pattern_sim = ci_pattern.CIPatternNonUniform(normalization=500.0,
                                                         regions=[(0, 1, 0, 1)],
                                                         row_slope=1.0)

            column = pattern_sim.generate_column(size=3, normalization=500.0)

            ci_column = np.zeros((11,))

            ci_column[0:3] = column
            ci_column[4:7] = column
            ci_column[8:11] = column

            column_mask = np.array([[False],
                                    [False],
                                    [False],
                                    [False],
                                    [False],
                                    [False],
                                    [False],
                                    [False],
                                    [False],
                                    [False],
                                    [False]])

            pattern = ci_pattern.CIPatternNonUniform(normalization=500.0,
                                                     regions=[(0, 3, 0, 1), (4, 7, 0, 1), (8, 11, 0, 1)],
                                                     row_slope=1.0)

            mean_of_columns = pattern.mean_charge_in_all_image_columns(column=ci_column, column_mask=column_mask)

            assert mean_of_columns == [500.0, 500.0, 500.0]

    class TestSetupMockNonUniformImage:

        def test__one_ci_region__all_columns_same__set_up_identical_image(self):
            ci_image = np.array([[10.0, 10.0, 10.0],
                                 [10.0, 10.0, 10.0],
                                 [10.0, 10.0, 10.0]])

            mask = np.array([[False, False, False],
                             [False, False, False],
                             [False, False, False]])

            pattern = ci_pattern.CIPatternNonUniform(normalization=10.0,
                                                     regions=[(0, 3, 0, 3)],
                                                     row_slope=0.0)

            mock_image = pattern.ci_pre_cti_from_ci_image_and_mask(ci_image=ci_image, mask=mask)

            assert (mock_image == ci_image).all()

        def test__one_ci_region__different_columns_but_same_values__set_up_identical_image(self):
            ci_image = np.array([[10.0, 11.0, 18.0],
                                 [10.0, 11.0, 18.0],
                                 [10.0, 11.0, 18.0]])

            mask = np.array([[False, False, False],
                             [False, False, False],
                             [False, False, False]])

            pattern = ci_pattern.CIPatternNonUniform(normalization=10.0,
                                                     regions=[(0, 3, 0, 3)],
                                                     row_slope=0.0)

            mock_image = pattern.ci_pre_cti_from_ci_image_and_mask(ci_image=ci_image, mask=mask)

            assert (mock_image == ci_image).all()

        def test__one_ci_region__different_columns_different_values__set_up_as_means(self):
            ci_image = np.array([[10.0, 11.0, 18.0],
                                 [13.0, 8.0, 18.0],
                                 [16.0, 5.0, 18.0]])

            mask = np.array([[False, False, False],
                             [False, False, False],
                             [False, False, False]])

            pattern = ci_pattern.CIPatternNonUniform(normalization=10.0,
                                                     regions=[(0, 3, 0, 3)],
                                                     row_slope=0.0)

            mock_image = pattern.ci_pre_cti_from_ci_image_and_mask(ci_image=ci_image, mask=mask)

            assert (mock_image == np.array([[13.0, 8.0, 18.0],
                                            [13.0, 8.0, 18.0],
                                            [13.0, 8.0, 18.0]])).all()

        def test__one_ci_region__mask_changes_mean_values(self):
            ci_image = np.array([[10.0, 11.0, 18.0],
                                 [13.0, 8.0, 18.0],
                                 [16.0, 5.0, 18.0]])

            mask = np.array([[True, True, True],
                             [False, False, True],
                             [False, False, False]])

            pattern = ci_pattern.CIPatternNonUniform(normalization=10.0,
                                                     regions=[(0, 3, 0, 3)],
                                                     row_slope=0.0)

            mock_image = pattern.ci_pre_cti_from_ci_image_and_mask(ci_image=ci_image, mask=mask)

            assert (mock_image == np.array([[14.5, 6.5, 18.0],
                                            [14.5, 6.5, 18.0],
                                            [14.5, 6.5, 18.0]])).all()

        def test__one_ci_region__fully_masked_raises_an_error(self):
            ci_image = np.array([[10.0, 11.0, 18.0],
                                 [13.0, 8.0, 18.0],
                                 [16.0, 5.0, 18.0]])

            mask = np.array([[True, True, True],
                             [False, True, True],
                             [False, True, False]])

            pattern = ci_pattern.CIPatternNonUniform(normalization=10.0,
                                                     regions=[(0, 3, 0, 3)],
                                                     row_slope=0.0)

            with pytest.raises(exc.CIPatternException):
                pattern.ci_pre_cti_from_ci_image_and_mask(ci_image=ci_image, mask=mask)

        def test__one_ci_region__includes_row_non_uniformity__set_up_as_means(self):
            pattern_sim = ci_pattern.CIPatternNonUniform(normalization=500.0, regions=[(0, 1, 0, 1)],
                                                         row_slope=1.0)

            column = pattern_sim.generate_column(size=3, normalization=500.0)

            ci_image = np.zeros((3, 3))

            ci_image[:, 0] = column
            ci_image[:, 1] = column
            ci_image[:, 2] = column

            mask = np.array([[False, False, False],
                             [False, False, False],
                             [False, False, False]])

            pattern = ci_pattern.CIPatternNonUniform(normalization=500.0, regions=[(0, 3, 0, 3)],
                                                     row_slope=1.0)

            mock_image = pattern.ci_pre_cti_from_ci_image_and_mask(ci_image=ci_image, mask=mask)

            assert (mock_image == np.array([[500.0, 500.0, 500.0],
                                            [500.0, 500.0, 500.0],
                                            [500.0, 500.0, 500.0]])).all()

        def test__two_ci_regions__takes_mean_of_entire_image_column(self):
            ci_image = np.array([[5.0, 9.0, 8.0],
                                 [5.0, 6.0, 8.0],
                                 [5.0, 3.0, 8.0],
                                 [0.0, 0.0, 0.0],
                                 [5.0, 3.0, 6.0],
                                 [5.0, 6.0, 6.0],
                                 [5.0, 9.0, 6.0]])

            mask = np.array([[False, False, False],
                             [False, False, False],
                             [False, False, False],
                             [False, False, False],
                             [False, False, False],
                             [False, False, False],
                             [False, False, False]])

            pattern = ci_pattern.CIPatternNonUniform(normalization=10.0,
                                                     regions=[(0, 3, 0, 3), (4, 7, 0, 3)], row_slope=0.0)

            mock_image = pattern.ci_pre_cti_from_ci_image_and_mask(ci_image=ci_image, mask=mask)

            assert (mock_image == np.array([[5.0, 6.0, 7.0],
                                            [5.0, 6.0, 7.0],
                                            [5.0, 6.0, 7.0],
                                            [0.0, 0.0, 0.0],
                                            [5.0, 6.0, 7.0],
                                            [5.0, 6.0, 7.0],
                                            [5.0, 6.0, 7.0]])).all()

        def test__two_ci_regions__full_mask_on_one_column__mean_uses_other_column(self):
            ci_image = np.array([[6.0, 9.0, 8.0],
                                 [6.0, 9.0, 8.0],
                                 [6.0, 9.0, 8.0],
                                 [0.0, 0.0, 0.0],
                                 [5.0, 8.0, 6.0],
                                 [5.0, 8.0, 6.0],
                                 [5.0, 8.0, 6.0]])

            mask = np.array([[False, True, False],
                             [False, True, False],
                             [False, True, False],
                             [False, False, False],
                             [True, False, True],
                             [True, False, True],
                             [True, False, True]])

            pattern = ci_pattern.CIPatternNonUniform(normalization=10.0,
                                                     regions=[(0, 3, 0, 3), (4, 7, 0, 3)], row_slope=0.0)

            mock_image = pattern.ci_pre_cti_from_ci_image_and_mask(ci_image=ci_image, mask=mask)

            assert (mock_image == np.array([[6.0, 8.0, 8.0],
                                            [6.0, 8.0, 8.0],
                                            [6.0, 8.0, 8.0],
                                            [0.0, 0.0, 0.0],
                                            [6.0, 8.0, 8.0],
                                            [6.0, 8.0, 8.0],
                                            [6.0, 8.0, 8.0]])).all()

        def test__two_ci_regions__includes_row_non_uniformity__set_up_as_means(self):
            pattern_sim = ci_pattern.CIPatternNonUniform(normalization=500.0,
                                                         regions=[(0, 1, 0, 1)],
                                                         row_slope=1.0)

            column = pattern_sim.generate_column(size=3, normalization=500.0)

            ci_image = np.zeros((7, 3))

            ci_image[0:3, 0] = column
            ci_image[0:3, 1] = column
            ci_image[0:3, 2] = column

            ci_image[4:7, 0] = column
            ci_image[4:7, 1] = column
            ci_image[4:7, 2] = column

            mask = np.array([[False, False, False],
                             [False, False, False],
                             [False, False, False],
                             [False, False, False],
                             [False, False, False],
                             [False, False, False],
                             [False, False, False]])

            pattern = ci_pattern.CIPatternNonUniform(normalization=500.0,
                                                     regions=[(0, 3, 0, 3), (4, 7, 0, 3)], row_slope=1.0)

            mock_image = pattern.ci_pre_cti_from_ci_image_and_mask(ci_image=ci_image, mask=mask)

            assert (mock_image == np.array([[500.0, 500.0, 500.0],
                                            [500.0, 500.0, 500.0],
                                            [500.0, 500.0, 500.0],
                                            [0.0, 0.0, 0.0],
                                            [500.0, 500.0, 500.0],
                                            [500.0, 500.0, 500.0],
                                            [500.0, 500.0, 500.0]])).all()

        def test__pattern_bigger_than_image_dimensions__raises_error(self):
            pattern = ci_pattern.CIPatternNonUniform(normalization=10.0, regions=[(0, 2, 0, 2)], row_slope=0.0)

            with pytest.raises(exc.CIPatternException):
                pattern.ci_pre_cti_from_ci_image_and_mask(ci_image=np.ones((1, 1)), mask=np.ma.ones((1, 1)))


class TestCIPatternSimulateUniform(object):
    class TestSimulateCIPreCTI:

        def test__image_3x3__1_ci_region(self):
            sim_pattern_uni = ci_pattern.CIPatternUniform(normalization=10.0, regions=[(0, 2, 0, 2)])
            image1 = sim_pattern_uni.simulate_ci_pre_cti(shape=(3, 3))

            assert (image1 == np.array([[10.0, 10.0, 0.0],
                                        [10.0, 10.0, 0.0],
                                        [0.0, 0.0, 0.0]])).all()

        def test__image_3x3__2_ci_regions(self):
            sim_pattern_uni = ci_pattern.CIPatternUniform(normalization=10.0,
                                                          regions=[(0, 2, 0, 2), (2, 3, 2, 3)])
            image1 = sim_pattern_uni.simulate_ci_pre_cti(shape=(3, 3))

            assert (image1 == np.array([[10.0, 10.0, 0.0],
                                        [10.0, 10.0, 0.0],
                                        [0.0, 0.0, 10.0]])).all()

        def test__same_as_above__different_normalizaition(self):
            sim_pattern_uni = ci_pattern.CIPatternUniform(normalization=20.0,
                                                          regions=[(0, 2, 0, 2), (2, 3, 2, 3)])
            image1 = sim_pattern_uni.simulate_ci_pre_cti(shape=(3, 3))

            assert (image1 == np.array([[20.0, 20.0, 0.0],
                                        [20.0, 20.0, 0.0],
                                        [0.0, 0.0, 20.0]])).all()

        def test__same_as_above__different_everything(self):
            sim_pattern_uni = ci_pattern.CIPatternUniform(normalization=30.0,
                                                          regions=[(0, 3, 0, 2), (2, 3, 2, 3)])
            image1 = sim_pattern_uni.simulate_ci_pre_cti(shape=(4, 3))

            assert (image1 == np.array([[30.0, 30.0, 0.0],
                                        [30.0, 30.0, 0.0],
                                        [30.0, 30.0, 30.0],
                                        [0.0, 0.0, 0.0]])).all()

        def test__pattern_bigger_than_image_dimensions__raises_error(self):
            pattern = ci_pattern.CIPatternUniform(normalization=10.0, regions=[(0, 2, 0, 2)])

            with pytest.raises(exc.CIPatternException):
                pattern.ci_pre_cti_from_shape(shape=(1, 1))


class TestCIPatternSimulateNonUniform(object):
    class TestSimulateRegion:

        def test__uniform_column_and_uniform_row__returns_uniform_charge_region(self):
            sim_pattern = ci_pattern.CIPatternNonUniform(normalization=100.0, regions=[(0, 1, 0, 1)],
                                                         row_slope=0.0)

            non_uniform_charge_region = sim_pattern.simulate_region(region_dimensions=(3, 3), ci_seed=1)

            assert (non_uniform_charge_region == np.array([[100.0, 100.0, 100.0],
                                                           [100.0, 100.0, 100.0],
                                                           [100.0, 100.0, 100.0]])).all()

        def test__same_as_above_but_different_normalization_and_dimensions__returns_uniform_charge_region(self):
            sim_pattern = ci_pattern.CIPatternNonUniform(normalization=500.0, regions=[(0, 1, 0, 1)],
                                                         row_slope=0.0)

            non_uniform_charge_region = sim_pattern.simulate_region(region_dimensions=(5, 3), ci_seed=1)

            assert (non_uniform_charge_region == np.array([[500.0, 500.0, 500.0],
                                                           [500.0, 500.0, 500.0],
                                                           [500.0, 500.0, 500.0],
                                                           [500.0, 500.0, 500.0],
                                                           [500.0, 500.0, 500.0]])).all()

        def test__non_uniform_column_and_uniform_row__returns_non_uniform_charge_region(self):
            sim_pattern = ci_pattern.CIPatternNonUniform(normalization=100.0, regions=[(0, 1, 0, 1)],
                                                         row_slope=0.0)

            non_uniform_charge_region = sim_pattern.simulate_region(region_dimensions=(3, 3), ci_seed=1,
                                                                    column_deviation=1.0)

            non_uniform_charge_region = np.round(non_uniform_charge_region, 1)

            assert (non_uniform_charge_region == np.array([[101.6, 99.4, 99.5],
                                                           [101.6, 99.4, 99.5],
                                                           [101.6, 99.4, 99.5]])).all()

        def test__same_as_above_but_different_normalization_and_dimensions_2__returns_non_uniform_charge_region(self):
            sim_pattern = ci_pattern.CIPatternNonUniform(normalization=500.0, regions=[(0, 1, 0, 1)],
                                                         row_slope=0.0)

            non_uniform_charge_region = sim_pattern.simulate_region(region_dimensions=(5, 3), ci_seed=1,
                                                                    column_deviation=1.0)

            non_uniform_charge_region = np.round(non_uniform_charge_region, 1)

            assert (non_uniform_charge_region == np.array([[501.6, 499.4, 499.5],
                                                           [501.6, 499.4, 499.5],
                                                           [501.6, 499.4, 499.5],
                                                           [501.6, 499.4, 499.5],
                                                           [501.6, 499.4, 499.5]])).all()

        def test__uniform_column_and_non_uniform_row__returns_non_uniform_charge_region(self):
            sim_pattern = ci_pattern.CIPatternNonUniform(normalization=100.0, regions=[(0, 1, 0, 1)],
                                                         row_slope=-0.01)

            non_uniform_charge_region = sim_pattern.simulate_region(region_dimensions=(3, 3), ci_seed=1)

            non_uniform_charge_region = np.round(non_uniform_charge_region, 1)

            assert (non_uniform_charge_region == np.array([[100.0, 100.0, 100.0],
                                                           [99.3, 99.3, 99.3],
                                                           [98.9, 98.9, 98.9]])).all()

        def test__same_as_above_but_different_normalization_and_dimensions_3__returns_non_uniform_charge(self):
            sim_pattern = ci_pattern.CIPatternNonUniform(normalization=500.0, regions=[(0, 1, 0, 1)],
                                                         row_slope=-0.01)

            non_uniform_charge_region = sim_pattern.simulate_region(region_dimensions=(5, 3), ci_seed=1)

            non_uniform_charge_region = np.round(non_uniform_charge_region, 1)

            assert (non_uniform_charge_region == np.array([[500.0, 500.0, 500.0],
                                                           [496.5, 496.5, 496.5],
                                                           [494.5, 494.5, 494.5],
                                                           [493.1, 493.1, 493.1],
                                                           [492.0, 492.0, 492.0]])).all()

        def test__non_uniform_column_and_non_uniform_row__returns_non_uniform_charge_region(self):
            sim_pattern = ci_pattern.CIPatternNonUniform(normalization=100.0, regions=[(0, 1, 0, 1)],
                                                         row_slope=-0.01)

            non_uniform_charge_region = sim_pattern.simulate_region(region_dimensions=(3, 3), ci_seed=1,
                                                                    column_deviation=1.0)

            non_uniform_charge_region = np.round(non_uniform_charge_region, 1)

            assert (non_uniform_charge_region == np.array([[101.6, 99.4, 99.5],
                                                           [100.9, 98.7, 98.8],
                                                           [100.5, 98.3, 98.4]])).all()

        def test__same_as_above_but_different_normalization_and_dimensions_4__returns_non_uniform_charge_region(self):
            sim_pattern = ci_pattern.CIPatternNonUniform(normalization=500.0, regions=[(0, 1, 0, 1)],
                                                         row_slope=-0.01)

            non_uniform_charge_region = sim_pattern.simulate_region(region_dimensions=(5, 3), ci_seed=1,
                                                                    column_deviation=1.0)

            non_uniform_charge_region = np.round(non_uniform_charge_region, 1)

            assert (non_uniform_charge_region == np.array([[501.6, 499.4, 499.5],
                                                           [498.2, 495.9, 496.0],
                                                           [496.1, 493.9, 494.0],
                                                           [494.7, 492.5, 492.6],
                                                           [493.6, 491.4, 491.5]])).all()

        def test__non_uniform_columns_with_large_deviation_value__no_negative_charge_columns_are_generated(self):
            sim_pattern = ci_pattern.CIPatternNonUniform(normalization=100.0, regions=[(0, 1, 0, 1)],
                                                         row_slope=0.0)

            non_uniform_charge_region = sim_pattern.simulate_region(region_dimensions=(10, 10), ci_seed=1,
                                                                    column_deviation=100.0)

            assert (non_uniform_charge_region > 0).all()

    class TestSimulateImage:

        def test__no_non_uniformity__identical_to_uniform_image__one_ci_region(self):
            sim_pattern_uni = ci_pattern.CIPatternUniform(normalization=10.0, regions=[(2, 4, 0, 5)])
            image1 = sim_pattern_uni.simulate_ci_pre_cti(shape=(5, 5))

            sim_pattern_non_uni = ci_pattern.CIPatternNonUniform(normalization=10.0, regions=[(2, 4, 0, 5)],
                                                                 row_slope=0.0)
            image2 = sim_pattern_non_uni.simulate_ci_pre_cti(shape=(5, 5))

            assert (image1 == image2).all()

        def test__no_non_uniformity__identical_to_uniform_image__rectangular_image(self):
            sim_pattern_uni = ci_pattern.CIPatternUniform(normalization=10.0, regions=[(2, 4, 0, 5)])
            image1 = sim_pattern_uni.simulate_ci_pre_cti(shape=(5, 7))

            sim_pattern_non_uni = ci_pattern.CIPatternNonUniform(
                normalization=10.0,
                regions=[(2, 4, 0, 5)],
                row_slope=0.0)
            image2 = sim_pattern_non_uni.simulate_ci_pre_cti(column_deviation=0.0,
                                                             shape=(5, 7))

            assert (image1 == image2).all()

        def test__no_non_uniformity__identical_to_uniform_image__different_normalization_and_region(self):
            sim_pattern_uni = ci_pattern.CIPatternUniform(normalization=100.0, regions=[(1, 4, 2, 5)])
            image1 = sim_pattern_uni.simulate_ci_pre_cti(shape=(5, 5))

            sim_pattern_non_uni = ci_pattern.CIPatternNonUniform(normalization=100.0, regions=[(1, 4, 2, 5)],
                                                                 row_slope=0.0)
            image2 = sim_pattern_non_uni.simulate_ci_pre_cti(shape=(5, 5))

            assert (image1 == image2).all()

        def test__no_non_uniformity__identical_to_uniform_image__two_ci_regions(self):
            sim_pattern_uni = ci_pattern.CIPatternUniform(normalization=100.0,
                                                          regions=[(0, 2, 0, 2), (2, 3, 0, 5)])
            image1 = sim_pattern_uni.simulate_ci_pre_cti(shape=(5, 5))

            sim_pattern_non_uni = ci_pattern.CIPatternNonUniform(normalization=100.0,
                                                                 regions=[(0, 2, 0, 2), (2, 3, 0, 5)],
                                                                 row_slope=0.0)
            image2 = sim_pattern_non_uni.simulate_ci_pre_cti(shape=(5, 5))

            assert (image1 == image2).all()

        def test__non_uniformity_in_columns_only__one_ci_region__image_is_correct(self):
            sim_pattern_non_uni = ci_pattern.CIPatternNonUniform(normalization=100.0, regions=[(0, 3, 0, 3)],
                                                                 row_slope=0.0)

            image = sim_pattern_non_uni.simulate_ci_pre_cti(shape=(5, 5), ci_seed=1,
                                                            column_deviation=1.0)

            image = np.round(image, 1)

            assert (image == np.array([[101.6, 99.4, 99.5, 0.0, 0.0],
                                       [101.6, 99.4, 99.5, 0.0, 0.0],
                                       [101.6, 99.4, 99.5, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

        def test__non_uniformity_in_columns_only__different_ci_region_to_above__image_is_correct(self):
            sim_pattern_non_uni = ci_pattern.CIPatternNonUniform(normalization=100.0, regions=[(1, 4, 1, 4)],
                                                                 row_slope=0.0)

            image = sim_pattern_non_uni.simulate_ci_pre_cti(shape=(5, 5), ci_seed=1,
                                                            column_deviation=1.0)

            image = np.round(image, 1)

            assert (image == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                       [0.0, 101.6, 99.4, 99.5, 0.0],
                                       [0.0, 101.6, 99.4, 99.5, 0.0],
                                       [0.0, 101.6, 99.4, 99.5, 0.0],
                                       [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

        def test__non_uniformity_in_columns_only__two_ci_regions__image_is_correct(self):
            sim_pattern_non_uni = ci_pattern.CIPatternNonUniform(normalization=100.0,
                                                                 regions=[(1, 4, 1, 3), (1, 4, 4, 5)],
                                                                 row_slope=0.0)

            image = sim_pattern_non_uni.simulate_ci_pre_cti(shape=(5, 5), ci_seed=1,
                                                            column_deviation=1.0)

            image = np.round(image, 1)

            assert (image == np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                       [0.0, 101.6, 99.4, 0.0, 101.6],
                                       [0.0, 101.6, 99.4, 0.0, 101.6],
                                       [0.0, 101.6, 99.4, 0.0, 101.6],
                                       [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

        def test__non_uniformity_in_columns_only__maximum_normalization_input__does_not_simulate_above(self):
            sim_pattern_non_uni = ci_pattern.CIPatternNonUniform(normalization=100.0, regions=[(0, 5, 0, 5)],
                                                                 row_slope=0.0)

            image = sim_pattern_non_uni.simulate_ci_pre_cti(shape=(5, 5), ci_seed=1,
                                                            column_deviation=100.0, maximum_normalization=100.0)

            image = np.round(image, 1)

            # Checked ci_seed to ensure the max is above 100.0 without a maximum_normalization
            assert np.max(image) < 100.0

        def test__non_uniformity_in_rows_only__one_ci_region__image_is_correct(self):
            sim_pattern_non_uni = ci_pattern.CIPatternNonUniform(normalization=100.0,
                                                                 regions=[(0, 3, 0, 3)],
                                                                 row_slope=-0.01)

            image = sim_pattern_non_uni.simulate_ci_pre_cti(column_deviation=0.0,
                                                            shape=(5, 5))

            image = np.round(image, 1)

            assert (image == np.array([[100.0, 100.0, 100.0, 0.0, 0.0],
                                       [99.3, 99.3, 99.3, 0.0, 0.0],
                                       [98.9, 98.9, 98.9, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, 0.0, 0.0]])).all()

        def test__non_uniformity_in_rows_only__two_ci_regions__image_is_correct(self):
            sim_pattern_non_uni = ci_pattern.CIPatternNonUniform(normalization=100.0,
                                                                 regions=[(1, 5, 1, 4), (0, 5, 4, 5)],
                                                                 row_slope=-0.01)

            image = sim_pattern_non_uni.simulate_ci_pre_cti(shape=(5, 5))

            image = np.round(image, 1)

            assert (image == np.array([[0.0, 0.0, 0.0, 0.0, 100.0],
                                       [0.0, 100.0, 100.0, 100.0, 99.3],
                                       [0.0, 99.3, 99.3, 99.3, 98.9],
                                       [0.0, 98.9, 98.9, 98.9, 98.6],
                                       [0.0, 98.6, 98.6, 98.6, 98.4]])).all()

        def test__non_uniformity_in_rows_and_columns__two_ci_regions__image_is_correct(self):
            sim_pattern_non_uni = ci_pattern.CIPatternNonUniform(normalization=100.0,
                                                                 regions=[(1, 5, 1, 4), (0, 5, 4, 5)],
                                                                 row_slope=-0.01)

            image = sim_pattern_non_uni.simulate_ci_pre_cti(shape=(5, 5), ci_seed=1,
                                                            column_deviation=1.0)

            image = np.round(image, 1)

            assert (image == np.array([[0.0, 0.0, 0.0, 0.0, 101.6],
                                       [0.0, 101.6, 99.4, 99.5, 100.9],
                                       [0.0, 100.9, 98.7, 98.8, 100.5],
                                       [0.0, 100.5, 98.3, 98.4, 100.2],
                                       [0.0, 100.2, 98.0, 98.1, 100.0]])).all()

        def test__non_uniformity_in_rows_and_columns__no_random_seed__two_ci_regions_are_identical(self):
            sim_pattern_non_uni = ci_pattern.CIPatternNonUniform(normalization=100.0,
                                                                 regions=[(0, 2, 0, 5), (3, 5, 0, 5)],
                                                                 row_slope=-0.01)

            image = sim_pattern_non_uni.simulate_ci_pre_cti(shape=(5, 5),
                                                            column_deviation=1.0)

            image = np.round(image, 1)

            assert (image[0:2, 0:5] == image[3:5, 0:5]).all()

        def test__pattern_bigger_than_image_dimensions__raises_error(self):
            pattern = ci_pattern.CIPatternNonUniform(normalization=10.0, regions=[(0, 2, 0, 2)],
                                                     row_slope=0.0)

            with pytest.raises(exc.CIPatternException):
                pattern.ci_pre_cti_from_ci_image_and_mask(ci_image=np.ones((1, 1)), mask=np.ma.ones((1, 1)))
