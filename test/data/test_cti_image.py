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
File: tests/python/CTIImage_test.py

Created on: 02/13/18
Author: James Nightingale
"""

import os

import numpy as np
import pytest

from autocti import exc
from autocti.data import cti_image
from autocti.data.charge_injection import ci_frame


@pytest.fixture(scope='function')
def quadrant_data():
    quadrant_data = np.array([[9, 0, 0],
                              [1, 1, 14],
                              [25, -6, 2]])

    return quadrant_data


@pytest.fixture(scope='class')
def image_data():
    image_data = np.array([[1, 0, 0],
                           [1, 1, 1],
                           [0, 1, 0]])

    return image_data


@pytest.fixture(scope='class')
def euclid_data():
    euclid_data = np.zeros((2048, 2066))
    return euclid_data


path = '{}/'.format(os.path.dirname(os.path.realpath(__file__)))


class TestCTIImage:
    class TestConstructor:

        def test__geometry_is_bottom_left__loads_data_and_dimensions(self, euclid_data):
            image = cti_image.CTIImage(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), array=euclid_data)

            assert type(image.frame_geometry) == ci_frame.QuadGeometryEuclid
            assert image.shape == (2048, 2066)
            assert (image == np.zeros((2048, 2066))).all()

        def test__geometry_is_bottom_right__loads_data_and_dimensions(self, euclid_data):
            image = cti_image.CTIImage(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_right(), array=euclid_data)

            assert type(image.frame_geometry) == ci_frame.QuadGeometryEuclid
            assert image.shape == (2048, 2066)
            assert (image == np.zeros((2048, 2066))).all()

        def test__geometry_is_top_left__loads_data_and_dimensions(self, euclid_data):
            image = cti_image.CTIImage(frame_geometry=ci_frame.QuadGeometryEuclid.top_left(), array=euclid_data)

            assert type(image.frame_geometry) == ci_frame.QuadGeometryEuclid
            assert image.shape == (2048, 2066)
            assert (image == np.zeros((2048, 2066))).all()

        def test__geometry_is_top_right__loads_data_and_dimensions(self, euclid_data):
            image = cti_image.CTIImage(frame_geometry=ci_frame.QuadGeometryEuclid.top_right(), array=euclid_data)

            assert type(image.frame_geometry) == ci_frame.QuadGeometryEuclid
            assert image.shape == (2048, 2066)
            assert (image == np.zeros((2048, 2066))).all()


class TestQuadrantEuclidGeometry:
    class TestFromCCDID:
        class TestLeftSide:

            def test__ccd_on_left_side_row_1__quadrant_id_E__chooses_bottom_left_quadrant(self):
                quadrant = ci_frame.QuadGeometryEuclid.from_ccd_and_quadrant_id(ccd_id='text1', quad_id='E')

                assert type(quadrant) == ci_frame.QuadGeometryEuclid

            def test__ccd_on_left_side_row_2__quadrant_id_E__chooses_bottom_left_quadrant(self):
                quadrant = ci_frame.QuadGeometryEuclid.from_ccd_and_quadrant_id(ccd_id='text2', quad_id='E')

                assert type(quadrant) == ci_frame.QuadGeometryEuclid

            def test__ccd_on_left_side_row_3__quadrant_id_E__chooses_bottom_left_quadrant(self):
                quadrant = ci_frame.QuadGeometryEuclid.from_ccd_and_quadrant_id(ccd_id='text3', quad_id='E')

                assert type(quadrant) == ci_frame.QuadGeometryEuclid

            def test__ccd_on_left_side_row_1__quadrant_id_F__chooses_bottom_right_quadrant(self):
                quadrant = ci_frame.QuadGeometryEuclid.from_ccd_and_quadrant_id(ccd_id='text1', quad_id='F')

                assert type(quadrant) == ci_frame.QuadGeometryEuclid

            def test__ccd_on_left_side_row_2__quadrant_id_F__chooses_bottom_right_quadrant(self):
                quadrant = ci_frame.QuadGeometryEuclid.from_ccd_and_quadrant_id(ccd_id='text2', quad_id='F')

                assert type(quadrant) == ci_frame.QuadGeometryEuclid

            def test__ccd_on_left_side_row_3__quadrant_id_F__chooses_bottom_right_quadrant(self):
                quadrant = ci_frame.QuadGeometryEuclid.from_ccd_and_quadrant_id(ccd_id='text3', quad_id='F')

                assert type(quadrant) == ci_frame.QuadGeometryEuclid

            def test__ccd_on_left_side_row_1__quadrant_id_G__chooses_top_right_quadrant(self):
                quadrant = ci_frame.QuadGeometryEuclid.from_ccd_and_quadrant_id(ccd_id='text1', quad_id='G')

                assert type(quadrant) == ci_frame.QuadGeometryEuclid

            def test__ccd_on_left_side_row_2__quadrant_id_G__chooses_top_right_quadrant(self):
                quadrant = ci_frame.QuadGeometryEuclid.from_ccd_and_quadrant_id(ccd_id='text2', quad_id='G')

                assert type(quadrant) == ci_frame.QuadGeometryEuclid

            def test__ccd_on_left_side_row_3__quadrant_id_G__chooses_top_right_quadrant(self):
                quadrant = ci_frame.QuadGeometryEuclid.from_ccd_and_quadrant_id(ccd_id='text3', quad_id='G')

                assert type(quadrant) == ci_frame.QuadGeometryEuclid

            def test__ccd_on_left_side_row_1__quadrant_id_H__chooses_top_left_quadrant(self):
                quadrant = ci_frame.QuadGeometryEuclid.from_ccd_and_quadrant_id(ccd_id='text1', quad_id='H')

                assert type(quadrant) == ci_frame.QuadGeometryEuclid

            def test__ccd_on_left_side_row_2__quadrant_id_H__chooses_top_left_quadrant(self):
                quadrant = ci_frame.QuadGeometryEuclid.from_ccd_and_quadrant_id(ccd_id='text2', quad_id='H')

                assert type(quadrant) == ci_frame.QuadGeometryEuclid

            def test__ccd_on_left_side_row_3__quadrant_id_H__chooses_top_left_quadrant(self):
                quadrant = ci_frame.QuadGeometryEuclid.from_ccd_and_quadrant_id(ccd_id='text3', quad_id='H')

                assert type(quadrant) == ci_frame.QuadGeometryEuclid

        class TestRightSide:

            def test__ccd_on_right_side_row_4__quadrant_id_E__chooses_top_right_quadrant(self):
                quadrant = ci_frame.QuadGeometryEuclid.from_ccd_and_quadrant_id(ccd_id='text4', quad_id='E')

                assert type(quadrant) == ci_frame.QuadGeometryEuclid

            def test__ccd_on_right_side_row_5__quadrant_id_E__chooses_top_right_quadrant(self):
                quadrant = ci_frame.QuadGeometryEuclid.from_ccd_and_quadrant_id(ccd_id='text5', quad_id='E')

                assert type(quadrant) == ci_frame.QuadGeometryEuclid

            def test__ccd_on_right_side_row_6__quadrant_id_E__chooses_top_right_quadrant(self):
                quadrant = ci_frame.QuadGeometryEuclid.from_ccd_and_quadrant_id(ccd_id='text6', quad_id='E')

                assert type(quadrant) == ci_frame.QuadGeometryEuclid

            def test__ccd_on_right_side_row_4__quadrant_id_F__chooses_top_left_quadrant(self):
                quadrant = ci_frame.QuadGeometryEuclid.from_ccd_and_quadrant_id(ccd_id='text4', quad_id='F')

                assert type(quadrant) == ci_frame.QuadGeometryEuclid

            def test__ccd_on_right_side_row_5__quadrant_id_F__chooses_top_left_quadrant(self):
                quadrant = ci_frame.QuadGeometryEuclid.from_ccd_and_quadrant_id(ccd_id='text5', quad_id='F')

                assert type(quadrant) == ci_frame.QuadGeometryEuclid

            def test__ccd_on_right_side_row_6__quadrant_id_F__chooses_top_left_quadrant(self):
                quadrant = ci_frame.QuadGeometryEuclid.from_ccd_and_quadrant_id(ccd_id='text6', quad_id='F')

                assert type(quadrant) == ci_frame.QuadGeometryEuclid

            def test__ccd_on_right_side_row_4__quadrant_id_G__chooses_bottom_left_quadrant(self):
                quadrant = ci_frame.QuadGeometryEuclid.from_ccd_and_quadrant_id(ccd_id='text4', quad_id='G')

                assert type(quadrant) == ci_frame.QuadGeometryEuclid

            def test__ccd_on_right_side_row_5__quadrant_id_G__chooses_bottom_left_quadrant(self):
                quadrant = ci_frame.QuadGeometryEuclid.from_ccd_and_quadrant_id(ccd_id='text5', quad_id='G')

                assert type(quadrant) == ci_frame.QuadGeometryEuclid

            def test__ccd_on_right_side_row_6__quadrant_id_G__chooses_bottom_left_quadrant(self):
                quadrant = ci_frame.QuadGeometryEuclid.from_ccd_and_quadrant_id(ccd_id='text6', quad_id='G')

                assert type(quadrant) == ci_frame.QuadGeometryEuclid

            def test__ccd_on_right_side_row_4__quadrant_id_H__chooses_bottom_right_quadrant(self):
                quadrant = ci_frame.QuadGeometryEuclid.from_ccd_and_quadrant_id(ccd_id='text4', quad_id='H')

                assert type(quadrant) == ci_frame.QuadGeometryEuclid

            def test__ccd_on_right_side_row_5__quadrant_id_H__chooses_bottom_right_quadrant(self):
                quadrant = ci_frame.QuadGeometryEuclid.from_ccd_and_quadrant_id(ccd_id='text5', quad_id='H')

                assert type(quadrant) == ci_frame.QuadGeometryEuclid

            def test__ccd_on_right_side_row_6__quadrant_id_H__chooses_bottom_right_quadrant(self):
                quadrant = ci_frame.QuadGeometryEuclid.from_ccd_and_quadrant_id(ccd_id='text6', quad_id='H')

                assert type(quadrant) == ci_frame.QuadGeometryEuclid


class TestQuadrantGeometryEuclidBottomLeft:
    class TestConstrutor:

        def test__sets_up_quadrant__including_correct_scans(self):
            quadrant = ci_frame.QuadGeometryEuclid.bottom_left()

            assert type(quadrant) == ci_frame.QuadGeometryEuclid

            assert quadrant.parallel_overscan == (2066, 2086, 51, 2099)
            assert quadrant.serial_prescan == (0, 2086, 0, 51)
            assert quadrant.serial_overscan == (0, 2086, 2099, 2119)

    class TestRotations:

        def test__rotate_for_parallel_clocking__oriented_as_described_in_documentation(self, quadrant_data):
            # Quadrant 0 - Bottom left panel of Euclid CCD - input ci_pre_ctis should not be rotated for parallel cti.

            quadrant = ci_frame.QuadGeometryEuclid.bottom_left()

            quadrant_rotated = quadrant.rotate_for_parallel_cti(image=quadrant_data)
            assert (quadrant_rotated == quadrant_data).all()

        def test__rotate_for_parallel_clocking_and_back_again__returns_to_original_orientation(self, quadrant_data):
            # Quadrant 0 - Bottom left panel of Euclid CCD - input ci_pre_ctis should not be rotated for parallel cti.

            quadrant = ci_frame.QuadGeometryEuclid.bottom_left()
            quadrant_rotated = quadrant.rotate_for_parallel_cti(image=quadrant_data)
            quadrant_rotated_back = quadrant.rotate_for_parallel_cti(image=quadrant_rotated)

            assert (quadrant_rotated_back == quadrant_data).all()

        def test__rotate_for_serial_clocking__oriented_as_described_in_documentation(self, quadrant_data):
            # Quadrant 0 - Bottom left panel of Euclid CCD - input ci_pre_ctis should be rotated 90 degrees for
            # serial cti.

            quadrant = ci_frame.QuadGeometryEuclid.bottom_left()

            quadrant_rotated = quadrant.rotate_before_serial_cti(image_pre_clocking=quadrant_data)
            assert (quadrant_rotated == quadrant_data.T).all()

        def test__rotate_for_serial_clocking_and_back_again__returns_to_original_orientation(self, quadrant_data):
            # Quadrant 0 - Bottom left panel of Euclid CCD - input ci_pre_ctis should be rotated 90 degrees for
            # serial cti.

            quadrant = ci_frame.QuadGeometryEuclid.bottom_left()
            quadrant_rotated = quadrant.rotate_before_serial_cti(image_pre_clocking=quadrant_data)
            quadrant_rotated_back = quadrant.rotate_after_serial_cti(image_post_clocking=quadrant_rotated)

            assert (quadrant_rotated_back == quadrant_data).all()


class TestQuadrantGeometryEuclidBottomRight:
    class TestConstrutor:

        def test__sets_up_quadrant__including_correct_overscans(self):
            quadrant = ci_frame.QuadGeometryEuclid.bottom_right()

            assert type(quadrant) == ci_frame.QuadGeometryEuclid
            assert quadrant.parallel_overscan == (2066, 2086, 20, 2068)
            assert quadrant.serial_prescan == (0, 2086, 2068, 2119)
            assert quadrant.serial_overscan == (0, 2086, 0, 20)

    class TestRotations:

        def test__rotate_for_parallel_clocking__oriented_as_described_in_documentation(self, quadrant_data):
            # Quadrant 1 - Bottom right panel of Euclid CCD - input ci_pre_ctis should not be rotateped for parallel cti.

            quadrant = ci_frame.QuadGeometryEuclid.bottom_right()

            quadrant_rotated = quadrant.rotate_for_parallel_cti(image=quadrant_data)
            assert (quadrant_rotated == quadrant_data).all()

        def test__rotate_for_parallel_clocking_and_back_again__returns_to_original_orientation(self, quadrant_data):
            # Quadrant 1 - Bottom right panel of Euclid CCD - input ci_pre_ctis should not be rotateped for parallel cti.

            quadrant = ci_frame.QuadGeometryEuclid.bottom_right()
            quadrant_rotated = quadrant.rotate_for_parallel_cti(image=quadrant_data)
            quadrant_rotated_back = quadrant.rotate_for_parallel_cti(image=quadrant_rotated)

            assert (quadrant_rotated_back == quadrant_data).all()

        def test__rotate_for_serial_clocking__oriented_as_described_in_documentation(self, quadrant_data):
            quadrant = ci_frame.QuadGeometryEuclid.bottom_right()

            quadrant_rotated = quadrant.rotate_before_serial_cti(image_pre_clocking=quadrant_data)
            assert (quadrant_rotated == np.fliplr(quadrant_data).T).all()

        def test__rotate_for_serial_clocking_and_back_again__returns_to_original_orientation(self, quadrant_data):
            quadrant = ci_frame.QuadGeometryEuclid.bottom_right()

            quadrant_rotated = quadrant.rotate_before_serial_cti(image_pre_clocking=quadrant_data)
            quadrant_rotated_back = quadrant.rotate_after_serial_cti(image_post_clocking=quadrant_rotated)
            assert (quadrant_rotated_back == quadrant_data).all()


class TestQuadrantGeometryEuclidTopLeft:
    class TestConstrutor:

        def test__sets_up_quadrant__including_correct_overscans(self):
            quadrant = ci_frame.QuadGeometryEuclid.top_left()

            assert type(quadrant) == ci_frame.QuadGeometryEuclid
            assert quadrant.parallel_overscan == (0, 20, 51, 2099)
            assert quadrant.serial_prescan == (0, 2086, 0, 51)
            assert quadrant.serial_overscan == (0, 2086, 2099, 2119)

    class TestRotations:

        def test__rotate_for_parallel_clocking__oriented_as_described_in_documentation(self, quadrant_data):
            # Quadrant 2 - top left panel of Euclid CCD - input ci_pre_ctis should be rotateped upside-down for parallel cti

            quadrant = ci_frame.QuadGeometryEuclid.top_left()

            quadrant_rotated = quadrant.rotate_for_parallel_cti(image=quadrant_data)
            assert (quadrant_rotated == np.flipud(quadrant_data)).all()

        def test__rotate_for_parallel_clocking_and_back_again__returns_to_original_orientation(self, quadrant_data):
            # Quadrant 2 - top left panel of Euclid CCD - input ci_pre_ctis should be rotateped upside-down for parallel cti

            quadrant = ci_frame.QuadGeometryEuclid.top_left()

            quadrant_rotated = quadrant.rotate_for_parallel_cti(image=quadrant_data)
            quadrant_rotated_back = quadrant.rotate_for_parallel_cti(image=quadrant_rotated)

            assert (quadrant_rotated_back == quadrant_data).all()

        def test__rotate_for_serial_clocking__oriented_as_described_in_documentation(self, quadrant_data):
            # Quadrant 2 - top left panel of Euclid CCD - input ci_pre_ctis should be rotated 90 degrees for serial cti.

            quadrant = ci_frame.QuadGeometryEuclid.top_left()

            quadrant_rotated = quadrant.rotate_before_serial_cti(image_pre_clocking=quadrant_data)
            assert (quadrant_rotated == quadrant_data.T).all()

        def test__rotate_for_serial_clocking_and_back_again__returns_to_original_orientation(self, quadrant_data):
            # Quadrant 2 - top left panel of Euclid CCD - input ci_pre_ctis should be rotated 90 degrees for serial cti.

            quadrant = ci_frame.QuadGeometryEuclid.top_left()

            quadrant_rotated = quadrant.rotate_before_serial_cti(image_pre_clocking=quadrant_data)
            quadrant_rotated_back = quadrant.rotate_after_serial_cti(image_post_clocking=quadrant_rotated)
            assert (quadrant_rotated_back == quadrant_data).all()


class TestQuadrantGeometryEuclidTopRight:
    class TestConstrutor:

        def test__sets_up_quadrant__including_correct_overscans(self):
            quadrant = ci_frame.QuadGeometryEuclid.top_right()

            assert type(quadrant) == ci_frame.QuadGeometryEuclid
            assert quadrant.parallel_overscan == (0, 20, 20, 2068)
            assert quadrant.serial_prescan == (0, 2086, 2068, 2119)
            assert quadrant.serial_overscan == (0, 2086, 0, 20)

    class TestRotations:

        def test__rotate_for_parallel_clocking__oriented_as_described_in_documentation(self, quadrant_data):
            # Quadrant 3 - top right panel of Euclid CCD - input ci_pre_ctis should be rotateped upside-down for parallel cti.

            quadrant = ci_frame.QuadGeometryEuclid.top_right()

            quadrant_rotated = quadrant.rotate_for_parallel_cti(image=quadrant_data)
            assert (quadrant_rotated == np.flipud(quadrant_data)).all()

        def test__rotate_for_parallel_clocking_and_back_again__returns_to_original_orientation(self, quadrant_data):
            # Quadrant 3 - top right panel of Euclid CCD - input ci_pre_ctis should be rotateped upside-down for parallel cti.

            quadrant = ci_frame.QuadGeometryEuclid.top_right()

            quadrant_rotated = quadrant.rotate_for_parallel_cti(image=quadrant_data)
            quadrant_rotated_back = quadrant.rotate_for_parallel_cti(image=quadrant_rotated)
            assert (quadrant_rotated_back == quadrant_data).all()

        def test__rotate_for_serial_clocking__oriented_as_described_in_documentation(self, quadrant_data):
            # Quadrant 3 - top right panel of Euclid CCD - input ci_pre_ctis should be rotated 270 degrees for serial cti.

            quadrant = ci_frame.QuadGeometryEuclid.top_right()

            quadrant_rotated = quadrant.rotate_before_serial_cti(image_pre_clocking=quadrant_data)
            assert (quadrant_rotated == np.fliplr(quadrant_data).T).all()

        def test__rotate_for_serial_clocking_and_back_again__returns_to_original_orientation(self, quadrant_data):
            # Quadrant 3 - top right panel of Euclid CCD - input ci_pre_ctis should be rotated 270 degrees for serial cti.

            quadrant = ci_frame.QuadGeometryEuclid.top_right()

            quadrant_rotated = quadrant.rotate_before_serial_cti(image_pre_clocking=quadrant_data)
            quadrant_rotated_back = quadrant.rotate_after_serial_cti(image_post_clocking=quadrant_rotated)
            assert (quadrant_rotated_back == quadrant_data).all()


class TestRegion(object):
    class TestConstructor:

        def test__constructor__converts_region_to_cartesians(self):
            region = ci_frame.Region(region=(0, 1, 2, 3))

            assert region == (0, 1, 2, 3)

            assert region.y0 == 0
            assert region.y1 == 1
            assert region.x0 == 2
            assert region.x1 == 3
            assert region.total_rows == 1
            assert region.total_columns == 1

        def test__first_row_or_column_equal_too_or_bigger_than_second__raise_errors(self):
            with pytest.raises(exc.RegionException):
                ci_frame.Region(region=(2, 2, 1, 2))

            with pytest.raises(exc.RegionException):
                ci_frame.Region(region=(2, 1, 2, 2))

            with pytest.raises(exc.RegionException):
                ci_frame.Region(region=(2, 1, 1, 2))

            with pytest.raises(exc.RegionException):
                ci_frame.Region(region=(0, 1, 3, 2))

        def test__negative_coordinates_raise_errors(self):
            with pytest.raises(exc.RegionException):
                ci_frame.Region(region=(-1, 0, 1, 2))

            with pytest.raises(exc.RegionException):
                ci_frame.Region(region=(0, -1, 1, 2))

            with pytest.raises(exc.RegionException):
                ci_frame.Region(region=(0, 0, -1, 2))

            with pytest.raises(exc.RegionException):
                ci_frame.Region(region=(0, 1, 2, -1))

    class TestExtractRegionFromArray:

        def test__extracts_2x2_region_of_3x3_array(self):
            array = np.array([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0],
                              [7.0, 8.0, 9.0]])

            region = ci_frame.Region(region=(0, 2, 0, 2))

            new_array = region.extract_region_from_array(array)

            assert (new_array == np.array([[1.0, 2.0],
                                           [4.0, 5.0]])).all()

        def test__extracts_2x3_region_of_3x3_array(self):
            array = np.array([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0],
                              [7.0, 8.0, 9.0]])

            region = ci_frame.Region(region=(1, 3, 0, 3))

            new_array = region.extract_region_from_array(array)

            assert (new_array == np.array([[4.0, 5.0, 6.0],
                                           [7.0, 8.0, 9.0]])).all()

    class TestAddRegionToArrayFromImage:

        def test__array_is_all_zeros__image_goes_into_correct_region_of_array(self):
            array = np.zeros((2, 2))
            image = np.ones((2, 2))
            region = ci_frame.Region(region=(0, 1, 0, 1))

            new_array = region.add_region_from_image_to_array(image=image, array=array)

            assert (new_array == np.array([[1.0, 0.0],
                                           [0.0, 0.0]])).all()

        def test__array_is_all_1s__image_goes_into_correct_region_of_array_and_adds_to_it(self):
            array = np.ones((2, 2))
            image = np.ones((2, 2))
            region = ci_frame.Region(region=(0, 1, 0, 1))

            new_array = region.add_region_from_image_to_array(image=image, array=array)

            assert (new_array == np.array([[2.0, 1.0],
                                           [1.0, 1.0]])).all()

        def test__different_region(self):
            array = np.ones((3, 3))
            image = np.ones((3, 3))
            region = ci_frame.Region(region=(1, 3, 2, 3))

            new_array = region.add_region_from_image_to_array(image=image, array=array)

            assert (new_array == np.array([[1.0, 1.0, 1.0],
                                           [1.0, 1.0, 2.0],
                                           [1.0, 1.0, 2.0]])).all()

    class TestSetRegionToZeros:

        def test__region_is_corner__sets_to_0(self):
            array = np.ones((2, 2))

            region = ci_frame.Region(region=(0, 1, 0, 1))

            new_array = region.set_region_on_array_to_zeros(array=array)

            assert (new_array == np.array([[0.0, 1.0],
                                           [1.0, 1.0]])).all()

        def test__different_region___sets_to_0(self):
            array = np.ones((3, 3))

            region = ci_frame.Region(region=(1, 3, 2, 3))

            new_array = region.set_region_on_array_to_zeros(array=array)

            assert (new_array == np.array([[1.0, 1.0, 1.0],
                                           [1.0, 1.0, 0.0],
                                           [1.0, 1.0, 0.0]])).all()
