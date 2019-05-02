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

import numpy as np

from autocti.charge_injection import ci_frame
from autocti.charge_injection import ci_pattern
from autocti.charge_injection import ci_mask


class MockPattern(object):

    def __init__(self):
        pass


class TestMaskedSerialFrontEdge:

    def test__pattern_left___mask_only_contains_front_edge(self):

        pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        frame = ci_frame.ChInj(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern)

        mask = ci_mask.CIMask.masked_serial_front_edge_from_ci_frame(shape=(3, 10), ci_frame=frame, columns=(0, 2))

        assert type(mask) == ci_mask.CIMask

        assert (mask == np.array([[True, False, False, True, True, True, True, True, True, True],
                                  [True, False, False, True, True, True, True, True, True, True],
                                  [True, False, False, True, True, True, True, True, True, True]])).all()

    def test__pattern_left__2_regions__extracts_columns_correctly(self):

        pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 1, 1, 4), (2, 3, 1, 4)])

        frame = ci_frame.ChInj(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern)

        mask = ci_mask.CIMask.masked_serial_front_edge_from_ci_frame(shape=(3, 10), ci_frame=frame, columns=(0, 3))

        assert type(mask) == ci_mask.CIMask

        assert (mask == np.array([[True, False, False, False, True, True, True, True, True, True],
                                  [True,  True,  True,  True, True, True, True, True, True, True],
                                  [True, False, False, False, True, True, True, True, True, True]])).all()

    def test__pattern_right__mask_only_contains_front_edge(self):

        pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        frame = ci_frame.ChInj(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_right(), ci_pattern=pattern)

        mask = ci_mask.CIMask.masked_serial_front_edge_from_ci_frame(shape=(3, 10), ci_frame=frame, columns=(0, 2))

        assert type(mask) == ci_mask.CIMask

        assert (mask == np.array([[True, True, False, False, True, True, True, True, True, True],
                                  [True, True, False, False, True, True, True, True, True, True],
                                  [True, True, False, False, True, True, True, True, True, True]])).all()

class TestMaskedSerialTrails:

    def test__pattern_left___mask_only_contains_trails(self):

        pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        frame = ci_frame.ChInj(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern)

        mask = ci_mask.CIMask.masked_serial_trails_from_ci_frame(shape=(3, 10), ci_frame=frame, columns=(0, 6))

        assert type(mask) == ci_mask.CIMask

        assert (mask == np.array([[True, True, True, True, False, False, False, False, False, False],
                                  [True, True, True, True, False, False, False, False, False, False],
                                  [True, True, True, True, False, False, False, False, False, False]])).all()

    def test__pattern_left__2_regions__extracts_columns_correctly(self):

        pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 1, 1, 4), (2, 3, 1, 4)])

        frame = ci_frame.ChInj(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern)

        mask = ci_mask.CIMask.masked_serial_trails_from_ci_frame(shape=(3, 10), ci_frame=frame, columns=(0, 6))

        assert type(mask) == ci_mask.CIMask

        assert (mask == np.array([[True, True, True, True, False, False, False, False, False, False],
                                  [True, True, True, True,  True,  True,  True,  True,  True,  True],
                                  [True, True, True, True, False, False, False, False, False, False]])).all()
        
    def test__pattern_right___mask_only_contains_trails(self):

        pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        frame = ci_frame.ChInj(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_right(), ci_pattern=pattern)

        mask = ci_mask.CIMask.masked_serial_trails_from_ci_frame(shape=(3, 10), ci_frame=frame, columns=(0, 1))

        assert type(mask) == ci_mask.CIMask

        assert (mask == np.array([[False, True, True, True, True, True, True, True, True, True],
                                  [False, True, True, True, True, True, True, True, True, True],
                                  [False, True, True, True, True, True, True, True, True, True]])).all()