import numpy as np

from autocti.charge_injection import ci_frame
from autocti.charge_injection import ci_pattern
from autocti.charge_injection import ci_mask


class MockPattern(object):

    def __init__(self):
        pass


class TestMaskedParallelFrontEdge:

    def test__pattern_bottom___mask_only_contains_front_edge(self):

        pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(1, 4, 0, 3)])

        frame = ci_frame.ChInj(frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(), ci_pattern=pattern)

        mask = ci_mask.CIMask.masked_parallel_front_edge_from_ci_frame(shape=(10, 3), ci_frame=frame, rows=(0, 2))

        assert type(mask) == ci_mask.CIMask

        assert (mask == np.array([[False,  False,  False],
                                  [True,  True,  True],  # <- Front edge according to region and this frame_geometry
                                  [True,  True,  True],  # <- Next front edge row.
                                  [False,  False,  False],
                                  [False,  False,  False],
                                  [False,  False,  False],
                                  [False,  False,  False],
                                  [False,  False,  False],
                                  [False,  False,  False],
                                  [False,  False,  False]])).all()


    def test__same_as_above_but_uses_invert(self):

        pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(1, 4, 0, 3)])

        frame = ci_frame.ChInj(frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(), ci_pattern=pattern)

        mask = ci_mask.CIMask.masked_parallel_front_edge_from_ci_frame(shape=(10, 3), ci_frame=frame, rows=(0, 2), 
                                                                       invert=True)

        assert type(mask) == ci_mask.CIMask

        assert (mask == np.array([[True,  True,  True],
                                  [False, False, False],  # <- Front edge according to region and this frame_geometry
                                  [False, False, False],  # <- Next front edge row.
                                  [True,  True,  True],
                                  [True,  True,  True],
                                  [True,  True,  True],
                                  [True,  True,  True],
                                  [True,  True,  True],
                                  [True,  True,  True],
                                  [True,  True,  True]])).all()
        
    def test__pattern_bottom__2_regions__extracts_rows_correctly(self):

        pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(1, 4, 0, 1), (1, 4, 2, 3)])

        frame = ci_frame.ChInj(frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(), ci_pattern=pattern)

        mask = ci_mask.CIMask.masked_parallel_front_edge_from_ci_frame(shape=(10, 3), ci_frame=frame, rows=(0, 2))

        assert type(mask) == ci_mask.CIMask

        assert (mask == np.array([[False,  False,  False],
                                  [True, False, True],  # <- Front edge according to region and this frame_geometry
                                  [True, False, True],  # <- Next front edge row.
                                  [False,  False,  False],
                                  [False,  False,  False],
                                  [False,  False,  False],
                                  [False,  False,  False],
                                  [False,  False,  False],
                                  [False,  False,  False],
                                  [False,  False,  False]])).all()

    def test__pattern_top__mask_only_contains_front_edge(self):

        pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(1, 4, 0, 3)])

        frame = ci_frame.ChInj(frame_geometry=ci_frame.FrameGeometry.euclid_top_left(), ci_pattern=pattern)

        mask = ci_mask.CIMask.masked_parallel_front_edge_from_ci_frame(shape=(10, 3), ci_frame=frame, rows=(0, 2))

        assert type(mask) == ci_mask.CIMask

        assert (mask == np.array([[False,  False,  False],
                                  [False,   False,  False],
                                  [True,  True,  True],  # <- Next front edge row.
                                  [True,  True,  True], # <- Front edge according to region and this frame_geometry
                                  [False,  False,  False],
                                  [False,  False,  False],
                                  [False,  False,  False],
                                  [False,  False,  False],
                                  [False,  False,  False],
                                  [False,  False,  False]])).all()


class TestMaskedParallelTrails:

    def test__pattern_bottom___mask_only_contains_trails(self):
        
        pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(1, 4, 0, 3)])

        frame = ci_frame.ChInj(frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(), ci_pattern=pattern)

        mask = ci_mask.CIMask.masked_parallel_trails_from_ci_frame(shape=(10, 3), ci_frame=frame, rows=(0, 4))

        assert type(mask) == ci_mask.CIMask

        assert (mask == np.array([[False, False, False],
                                  [False, False, False],  
                                  [False, False, False],  
                                  [False, False, False],
                                  [True,  True,  True], # <- Frist Trail according to region and this frame_geometry
                                  [True,  True,  True], # <- Next trail row.
                                  [True,  True,  True],
                                  [True,  True,  True],
                                  [False, False, False],
                                  [False, False, False]])).all()

    def test__same_as_above_but_uses_invert(self):
        
        pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(1, 4, 0, 3)])

        frame = ci_frame.ChInj(frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(), ci_pattern=pattern)

        mask = ci_mask.CIMask.masked_parallel_trails_from_ci_frame(shape=(10, 3), ci_frame=frame, rows=(0, 4), 
                                                                   invert=True)

        assert type(mask) == ci_mask.CIMask

        assert (mask == np.array([[True, True, True],
                                  [True, True, True],
                                  [True, True, True],
                                  [True, True, True],
                                  [False, False, False],  # <- Frist Trail according to region and this frame_geometry
                                  [False, False, False],  # <- Next trail row.
                                  [False, False, False],
                                  [False, False, False],
                                  [True, True, True],
                                  [True, True, True]])).all()

    def test__pattern_bottom__2_regions__extracts_rows_correctly(self):
        
        pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(1, 4, 0, 1), (1, 4, 2, 3)])

        frame = ci_frame.ChInj(frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(), ci_pattern=pattern)

        mask = ci_mask.CIMask.masked_parallel_trails_from_ci_frame(shape=(10, 3), ci_frame=frame, rows=(0, 4))

        assert type(mask) == ci_mask.CIMask
        
        assert (mask == np.array([[False, False, False],
                                  [False, False, False],
                                  [False, False, False],
                                  [False, False, False],
                                  [True,  False,  True], # <- Frist Trail according to region and this frame_geometry
                                  [True,  False,  True], # <- Next trail row.
                                  [True,  False,  True],
                                  [True,  False,  True],
                                  [False, False, False],
                                  [False, False, False]])).all()

    def test__pattern_top__mask_only_contains_trails(self):
        
        pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(1, 4, 0, 3)])

        frame = ci_frame.ChInj(frame_geometry=ci_frame.FrameGeometry.euclid_top_left(), ci_pattern=pattern)

        mask = ci_mask.CIMask.masked_parallel_trails_from_ci_frame(shape=(10, 3), ci_frame=frame, rows=(0, 1))

        assert type(mask) == ci_mask.CIMask

        assert (mask == np.array([[True,  True,  True], # <- Frist Trail according to region and this frame_geometry
                                  [False, False, False],
                                  [False, False, False], 
                                  [False, False, False], 
                                  [False, False, False], 
                                  [False, False, False],
                                  [False, False, False],
                                  [False, False, False],
                                  [False, False, False],
                                  [False, False, False]])).all()


class TestMaskedSerialFrontEdge:

    def test__pattern_left___mask_only_contains_front_edge(self):

        pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        frame = ci_frame.ChInj(frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(), ci_pattern=pattern)

        mask = ci_mask.CIMask.masked_serial_front_edge_from_ci_frame(shape=(3, 10), ci_frame=frame, columns=(0, 2))

        assert type(mask) == ci_mask.CIMask

        assert (mask == np.array([[False, True, True, False, False, False, False, False, False, False],
                                  [False, True, True, False, False, False, False, False, False, False],
                                  [False, True, True, False, False, False, False, False, False, False]])).all()

    def test__same_as_above_but_uses_invert(self):

        pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        frame = ci_frame.ChInj(frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(), ci_pattern=pattern)

        mask = ci_mask.CIMask.masked_serial_front_edge_from_ci_frame(shape=(3, 10), ci_frame=frame, columns=(0, 2),
                                                                     invert=True)

        assert type(mask) == ci_mask.CIMask

        assert (mask == np.array([[True, False, False, True, True, True, True, True, True, True],
                                  [True, False, False, True, True, True, True, True, True, True],
                                  [True, False, False, True, True, True, True, True, True, True]])).all()

    def test__pattern_left__2_regions__extracts_columns_correctly(self):

        pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 1, 1, 4), (2, 3, 1, 4)])

        frame = ci_frame.ChInj(frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(), ci_pattern=pattern)

        mask = ci_mask.CIMask.masked_serial_front_edge_from_ci_frame(shape=(3, 10), ci_frame=frame, columns=(0, 3))

        assert type(mask) == ci_mask.CIMask

        assert (mask == np.array([[False, True, True, True, False, False, False, False, False, False],
                                  [False,  False,  False,  False, False, False, False, False, False, False],
                                  [False, True, True, True, False, False, False, False, False, False]])).all()

    def test__pattern_right__mask_only_contains_front_edge(self):

        pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        frame = ci_frame.ChInj(frame_geometry=ci_frame.FrameGeometry.euclid_bottom_right(), ci_pattern=pattern)

        mask = ci_mask.CIMask.masked_serial_front_edge_from_ci_frame(shape=(3, 10), ci_frame=frame, columns=(0, 2))

        assert type(mask) == ci_mask.CIMask

        assert (mask == np.array([[False, False, True, True, False, False, False, False, False, False],
                                  [False, False, True, True, False, False, False, False, False, False],
                                  [False, False, True, True, False, False, False, False, False, False]])).all()


class TestMaskedSerialTrails:

    def test__pattern_left___mask_only_contains_trails(self):

        pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        frame = ci_frame.ChInj(frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(), ci_pattern=pattern)

        mask = ci_mask.CIMask.masked_serial_trails_from_ci_frame(shape=(3, 10), ci_frame=frame, columns=(0, 6))

        assert type(mask) == ci_mask.CIMask

        assert (mask == np.array([[False, False, False, False, True,  True,  True, True,  True,  True],
                                  [False, False, False, False, True,  True,  True, True,  True,  True],
                                  [False, False, False, False, True,  True,  True, True,  True,  True]])).all()
        
    def test__same_as_above_but_uses_invert(self):

        pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        frame = ci_frame.ChInj(frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(), ci_pattern=pattern)

        mask = ci_mask.CIMask.masked_serial_trails_from_ci_frame(shape=(3, 10), ci_frame=frame, columns=(0, 6), 
                                                                 invert=True)

        assert type(mask) == ci_mask.CIMask

        assert (mask == np.array([[True, True, True, True, False,  False,  False, False,  False,  False],
                                  [True, True, True, True, False,  False,  False, False,  False,  False],
                                  [True, True, True, True, False,  False,  False, False,  False,  False]])).all()

    def test__pattern_left__2_regions__extracts_columns_correctly(self):

        pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 1, 1, 4), (2, 3, 1, 4)])

        frame = ci_frame.ChInj(frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(), ci_pattern=pattern)

        mask = ci_mask.CIMask.masked_serial_trails_from_ci_frame(shape=(3, 10), ci_frame=frame, columns=(0, 6))

        assert type(mask) == ci_mask.CIMask

        assert (mask == np.array([[False, False, False, False, True,  True,  True, True,  True,  True],
                                  [False, False, False, False,  False,  False,  False,  False,  False,  False],
                                  [False, False, False, False, True,  True,  True, True,  True,  True]])).all()
        
    def test__pattern_right___mask_only_contains_trails(self):

        pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        frame = ci_frame.ChInj(frame_geometry=ci_frame.FrameGeometry.euclid_bottom_right(), ci_pattern=pattern)

        mask = ci_mask.CIMask.masked_serial_trails_from_ci_frame(shape=(3, 10), ci_frame=frame, columns=(0, 1))

        assert type(mask) == ci_mask.CIMask

        assert (mask == np.array([[True, False, False, False, False, False, False, False, False, False],
                                  [True, False, False, False, False, False, False, False, False, False],
                                  [True, False, False, False, False, False, False, False, False, False]])).all()