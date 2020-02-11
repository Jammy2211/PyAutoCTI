import numpy as np

import autocti as ac


class MockPattern(object):
    def __init__(self):
        pass


class TestMaskedParallelFrontEdge:
    def test__mask_only_contains_front_edge(self):

        pattern = ac.CIPatternUniform(normalization=1.0, regions=[(1, 4, 0, 3)])

        frame = ac.ci_frame.manual(
            array=np.ones((10, 3)), roe_corner=(1, 0), ci_pattern=pattern
        )

        mask = ac.CIMask.masked_parallel_front_edge_from_ci_frame(
            ci_frame=frame, rows=(0, 2)
        )

        assert type(mask) == ac.CIMask

        assert (
            mask
            == np.array(
                [
                    [False, False, False],
                    [
                        True,
                        True,
                        True,
                    ],  # <- Front edge according to region and this frame_geometry
                    [True, True, True],  # <- Next front edge row.
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ]
            )
        ).all()

    def test__same_as_above_but_uses_invert(self):

        pattern = ac.CIPatternUniform(normalization=1.0, regions=[(1, 4, 0, 3)])

        frame = ac.ci_frame.manual(
            array=np.ones((10, 3)), roe_corner=(1, 0), ci_pattern=pattern
        )

        mask = ac.CIMask.masked_parallel_front_edge_from_ci_frame(
            ci_frame=frame, rows=(0, 2), invert=True
        )

        assert type(mask) == ac.CIMask

        assert (
            mask
            == np.array(
                [
                    [True, True, True],
                    [
                        False,
                        False,
                        False,
                    ],  # <- Front edge according to region and this frame_geometry
                    [False, False, False],  # <- Next front edge row.
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                ]
            )
        ).all()

    def test__2_regions__extracts_rows_correctly(self):

        pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(1, 4, 0, 1), (1, 4, 2, 3)]
        )

        frame = ac.ci_frame.manual(
            array=np.ones((10, 3)), roe_corner=(1, 0), ci_pattern=pattern
        )

        mask = ac.CIMask.masked_parallel_front_edge_from_ci_frame(
            ci_frame=frame, rows=(0, 2)
        )

        assert type(mask) == ac.CIMask

        assert (
            mask
            == np.array(
                [
                    [False, False, False],
                    [
                        True,
                        False,
                        True,
                    ],  # <- Front edge according to region and this frame_geometry
                    [True, False, True],  # <- Next front edge row.
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ]
            )
        ).all()


class TestMaskedParallelTrails:
    def test___mask_only_contains_trails(self):

        pattern = ac.CIPatternUniform(normalization=1.0, regions=[(1, 4, 0, 3)])

        frame = ac.ci_frame.manual(
            array=np.ones((10, 3)), roe_corner=(1, 0), ci_pattern=pattern
        )

        mask = ac.CIMask.masked_parallel_trails_from_ci_frame(
            ci_frame=frame, rows=(0, 4)
        )

        assert type(mask) == ac.CIMask

        assert (
            mask
            == np.array(
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [
                        True,
                        True,
                        True,
                    ],  # <- Frist Trail according to region and this frame_geometry
                    [True, True, True],  # <- Next trail row.
                    [True, True, True],
                    [True, True, True],
                    [False, False, False],
                    [False, False, False],
                ]
            )
        ).all()

    def test__same_as_above_but_uses_invert(self):

        pattern = ac.CIPatternUniform(normalization=1.0, regions=[(1, 4, 0, 3)])

        frame = ac.ci_frame.manual(
            array=np.ones((10, 3)), roe_corner=(1, 0), ci_pattern=pattern
        )

        mask = ac.CIMask.masked_parallel_trails_from_ci_frame(
            ci_frame=frame, rows=(0, 4), invert=True
        )

        assert type(mask) == ac.CIMask

        assert (
            mask
            == np.array(
                [
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [
                        False,
                        False,
                        False,
                    ],  # <- Frist Trail according to region and this frame_geometry
                    [False, False, False],  # <- Next trail row.
                    [False, False, False],
                    [False, False, False],
                    [True, True, True],
                    [True, True, True],
                ]
            )
        ).all()

    def test__pattern_bottom__2_regions__extracts_rows_correctly(self):

        pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(1, 4, 0, 1), (1, 4, 2, 3)]
        )

        frame = ac.ci_frame.manual(
            array=np.ones((10, 3)), roe_corner=(1, 0), ci_pattern=pattern
        )

        mask = ac.CIMask.masked_parallel_trails_from_ci_frame(
            ci_frame=frame, rows=(0, 4)
        )

        assert type(mask) == ac.CIMask

        assert (
            mask
            == np.array(
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [
                        True,
                        False,
                        True,
                    ],  # <- Frist Trail according to region and this frame_geometry
                    [True, False, True],  # <- Next trail row.
                    [True, False, True],
                    [True, False, True],
                    [False, False, False],
                    [False, False, False],
                ]
            )
        ).all()


class TestMaskedSerialFrontEdge:
    def test__mask_only_contains_front_edge(self):

        pattern = ac.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        frame = ac.ci_frame.manual(
            array=np.ones((3, 10)), roe_corner=(1, 0), ci_pattern=pattern
        )

        mask = ac.CIMask.masked_serial_front_edge_from_ci_frame(
            ci_frame=frame, columns=(0, 2)
        )

        assert type(mask) == ac.CIMask

        assert (
            mask
            == np.array(
                [
                    [
                        False,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                    [
                        False,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                    [
                        False,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                ]
            )
        ).all()

    def test__same_as_above_but_uses_invert(self):

        pattern = ac.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        frame = ac.ci_frame.manual(
            array=np.ones((3, 10)), roe_corner=(1, 0), ci_pattern=pattern
        )

        mask = ac.CIMask.masked_serial_front_edge_from_ci_frame(
            ci_frame=frame, columns=(0, 2), invert=True
        )

        assert type(mask) == ac.CIMask

        assert (
            mask
            == np.array(
                [
                    [True, False, False, True, True, True, True, True, True, True],
                    [True, False, False, True, True, True, True, True, True, True],
                    [True, False, False, True, True, True, True, True, True, True],
                ]
            )
        ).all()

    def test__2_regions__extracts_columns_correctly(self):

        pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(0, 1, 1, 4), (2, 3, 1, 4)]
        )

        frame = ac.ci_frame.manual(
            array=np.ones((3, 10)), roe_corner=(1, 0), ci_pattern=pattern
        )

        mask = ac.CIMask.masked_serial_front_edge_from_ci_frame(
            ci_frame=frame, columns=(0, 3)
        )

        assert type(mask) == ac.CIMask

        assert (
            mask
            == np.array(
                [
                    [False, True, True, True, False, False, False, False, False, False],
                    [
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                    [False, True, True, True, False, False, False, False, False, False],
                ]
            )
        ).all()


class TestMaskedSerialTrails:
    def test__mask_only_contains_trails(self):

        pattern = ac.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        frame = ac.ci_frame.manual(
            array=np.ones((3, 10)), roe_corner=(1, 0), ci_pattern=pattern
        )

        mask = ac.CIMask.masked_serial_trails_from_ci_frame(
            ci_frame=frame, columns=(0, 6)
        )

        assert type(mask) == ac.CIMask

        assert (
            mask
            == np.array(
                [
                    [False, False, False, False, True, True, True, True, True, True],
                    [False, False, False, False, True, True, True, True, True, True],
                    [False, False, False, False, True, True, True, True, True, True],
                ]
            )
        ).all()

    def test__same_as_above_but_uses_invert(self):

        pattern = ac.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        frame = ac.ci_frame.manual(
            array=np.ones((3, 10)), roe_corner=(1, 0), ci_pattern=pattern
        )

        mask = ac.CIMask.masked_serial_trails_from_ci_frame(
            ci_frame=frame, columns=(0, 6), invert=True
        )

        assert type(mask) == ac.CIMask

        assert (
            mask
            == np.array(
                [
                    [True, True, True, True, False, False, False, False, False, False],
                    [True, True, True, True, False, False, False, False, False, False],
                    [True, True, True, True, False, False, False, False, False, False],
                ]
            )
        ).all()

    def test__2_regions__extracts_columns_correctly(self):

        pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(0, 1, 1, 4), (2, 3, 1, 4)]
        )

        frame = ac.ci_frame.manual(
            array=np.ones((3, 10)), roe_corner=(1, 0), ci_pattern=pattern
        )

        mask = ac.CIMask.masked_serial_trails_from_ci_frame(
            ci_frame=frame, columns=(0, 6)
        )

        assert type(mask) == ac.CIMask

        assert (
            mask
            == np.array(
                [
                    [False, False, False, False, True, True, True, True, True, True],
                    [
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                    [False, False, False, False, True, True, True, True, True, True],
                ]
            )
        ).all()
