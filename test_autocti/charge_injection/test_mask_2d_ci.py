import numpy as np

import autocti as ac


class MockPattern(object):
    def __init__(self):
        pass


class TestMaskedParallelFrontEdge:
    def test__mask_only_contains_front_edge(self):

        pattern = ac.ci.CIPatternUniform(normalization=1.0, regions=[(1, 4, 0, 3)])

        frame = ac.ci.CIFrame.manual(
            array=np.ones((10, 3)),
            roe_corner=(1, 0),
            pattern_ci=pattern,
            pixel_scales=1.0,
        )

        mask = ac.ci.CIMask2D.masked_parallel_front_edge_from_frame_ci(
            frame_ci=frame,
            settings=ac.ci.SettingsCIMask2D(parallel_front_edge_rows=(0, 2)),
        )

        assert type(mask) == ac.ci.CIMask2D

        assert (
            mask
            == np.array(
                [
                    [False, False, False],
                    [True, True, True],
                    [True, True, True],
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

        pattern = ac.ci.CIPatternUniform(normalization=1.0, regions=[(1, 4, 0, 3)])

        frame = ac.ci.CIFrame.manual(
            array=np.ones((10, 3)),
            roe_corner=(1, 0),
            pattern_ci=pattern,
            pixel_scales=1.0,
        )

        mask = ac.ci.CIMask2D.masked_parallel_front_edge_from_frame_ci(
            frame_ci=frame,
            settings=ac.ci.SettingsCIMask2D(parallel_front_edge_rows=(0, 2)),
            invert=True,
        )

        assert type(mask) == ac.ci.CIMask2D

        assert (
            mask
            == np.array(
                [
                    [True, True, True],
                    [False, False, False],
                    [False, False, False],
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

        pattern = ac.ci.CIPatternUniform(
            normalization=1.0, regions=[(1, 4, 0, 1), (1, 4, 2, 3)]
        )

        frame = ac.ci.CIFrame.manual(
            array=np.ones((10, 3)),
            roe_corner=(1, 0),
            pattern_ci=pattern,
            pixel_scales=1.0,
        )

        mask = ac.ci.CIMask2D.masked_parallel_front_edge_from_frame_ci(
            frame_ci=frame,
            settings=ac.ci.SettingsCIMask2D(parallel_front_edge_rows=(0, 2)),
        )

        assert type(mask) == ac.ci.CIMask2D

        assert (
            mask
            == np.array(
                [
                    [False, False, False],
                    [True, False, True],
                    [True, False, True],
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

        pattern = ac.ci.CIPatternUniform(normalization=1.0, regions=[(1, 4, 0, 3)])

        frame = ac.ci.CIFrame.manual(
            array=np.ones((10, 3)),
            roe_corner=(1, 0),
            pattern_ci=pattern,
            pixel_scales=1.0,
        )

        mask = ac.ci.CIMask2D.masked_parallel_trails_from_frame_ci(
            frame_ci=frame, settings=ac.ci.SettingsCIMask2D(parallel_trails_rows=(0, 4))
        )

        assert type(mask) == ac.ci.CIMask2D

        assert (
            mask
            == np.array(
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [False, False, False],
                    [False, False, False],
                ]
            )
        ).all()

    def test__same_as_above_but_uses_invert(self):

        pattern = ac.ci.CIPatternUniform(normalization=1.0, regions=[(1, 4, 0, 3)])

        frame = ac.ci.CIFrame.manual(
            array=np.ones((10, 3)),
            roe_corner=(1, 0),
            pattern_ci=pattern,
            pixel_scales=1.0,
        )

        mask = ac.ci.CIMask2D.masked_parallel_trails_from_frame_ci(
            frame_ci=frame,
            settings=ac.ci.SettingsCIMask2D(parallel_trails_rows=(0, 4)),
            invert=True,
        )

        assert type(mask) == ac.ci.CIMask2D

        assert (
            mask
            == np.array(
                [
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [True, True, True],
                    [True, True, True],
                ]
            )
        ).all()

    def test__pattern_bottom__2_regions__extracts_rows_correctly(self):

        pattern = ac.ci.CIPatternUniform(
            normalization=1.0, regions=[(1, 4, 0, 1), (1, 4, 2, 3)]
        )

        frame = ac.ci.CIFrame.manual(
            array=np.ones((10, 3)),
            roe_corner=(1, 0),
            pattern_ci=pattern,
            pixel_scales=1.0,
        )

        mask = ac.ci.CIMask2D.masked_parallel_trails_from_frame_ci(
            frame_ci=frame, settings=ac.ci.SettingsCIMask2D(parallel_trails_rows=(0, 4))
        )

        assert type(mask) == ac.ci.CIMask2D

        assert (
            mask
            == np.array(
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [True, False, True],
                    [True, False, True],
                    [True, False, True],
                    [True, False, True],
                    [False, False, False],
                    [False, False, False],
                ]
            )
        ).all()


class TestMaskedSerialFrontEdge:
    def test__mask_only_contains_front_edge(self):

        pattern = ac.ci.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        frame = ac.ci.CIFrame.manual(
            array=np.ones((3, 10)),
            roe_corner=(1, 0),
            pattern_ci=pattern,
            pixel_scales=1.0,
        )

        mask = ac.ci.CIMask2D.masked_serial_front_edge_from_frame_ci(
            frame_ci=frame,
            settings=ac.ci.SettingsCIMask2D(serial_front_edge_columns=(0, 2)),
        )

        assert type(mask) == ac.ci.CIMask2D

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

        pattern = ac.ci.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        frame = ac.ci.CIFrame.manual(
            array=np.ones((3, 10)),
            roe_corner=(1, 0),
            pattern_ci=pattern,
            pixel_scales=1.0,
        )

        mask = ac.ci.CIMask2D.masked_serial_front_edge_from_frame_ci(
            frame_ci=frame,
            settings=ac.ci.SettingsCIMask2D(serial_front_edge_columns=(0, 2)),
            invert=True,
        )

        assert type(mask) == ac.ci.CIMask2D

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

        pattern = ac.ci.CIPatternUniform(
            normalization=1.0, regions=[(0, 1, 1, 4), (2, 3, 1, 4)]
        )

        frame = ac.ci.CIFrame.manual(
            array=np.ones((3, 10)),
            roe_corner=(1, 0),
            pattern_ci=pattern,
            pixel_scales=1.0,
        )

        mask = ac.ci.CIMask2D.masked_serial_front_edge_from_frame_ci(
            frame_ci=frame,
            settings=ac.ci.SettingsCIMask2D(serial_front_edge_columns=(0, 3)),
        )

        assert type(mask) == ac.ci.CIMask2D

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

        pattern = ac.ci.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        frame = ac.ci.CIFrame.manual(
            array=np.ones((3, 10)),
            roe_corner=(1, 0),
            pattern_ci=pattern,
            pixel_scales=1.0,
        )

        mask = ac.ci.CIMask2D.masked_serial_trails_from_frame_ci(
            frame_ci=frame,
            settings=ac.ci.SettingsCIMask2D(serial_trails_columns=(0, 6)),
        )

        assert type(mask) == ac.ci.CIMask2D

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

        pattern = ac.ci.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        frame = ac.ci.CIFrame.manual(
            array=np.ones((3, 10)),
            roe_corner=(1, 0),
            pattern_ci=pattern,
            pixel_scales=1.0,
        )

        mask = ac.ci.CIMask2D.masked_serial_trails_from_frame_ci(
            frame_ci=frame,
            settings=ac.ci.SettingsCIMask2D(serial_trails_columns=(0, 6)),
            invert=True,
        )

        assert type(mask) == ac.ci.CIMask2D

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

        pattern = ac.ci.CIPatternUniform(
            normalization=1.0, regions=[(0, 1, 1, 4), (2, 3, 1, 4)]
        )

        frame = ac.ci.CIFrame.manual(
            array=np.ones((3, 10)),
            roe_corner=(1, 0),
            pattern_ci=pattern,
            pixel_scales=1.0,
        )

        mask = ac.ci.CIMask2D.masked_serial_trails_from_frame_ci(
            frame_ci=frame,
            settings=ac.ci.SettingsCIMask2D(serial_trails_columns=(0, 6)),
        )

        assert type(mask) == ac.ci.CIMask2D

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


class TestMaskedFrontEdgeTrailsAll:
    def test__masks_uses_front_edge_and_trails_parameters(self, imaging_ci_7x7):

        mask = ac.ci.CIMask2D.unmasked(
            shape_native=imaging_ci_7x7.shape_native, pixel_scales=1.0
        )

        ci_mask = ac.ci.CIMask2D.masked_front_edges_and_trails_from_frame_ci(
            frame_ci=imaging_ci_7x7.image,
            mask=mask,
            settings=ac.ci.SettingsCIMask2D(parallel_front_edge_rows=(0, 1)),
        )

        assert (
            ci_mask
            == np.array(
                [
                    [False, False, False, False, False, False, False],
                    [False, True, True, True, True, False, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                ]
            )
        ).all()

        ci_mask = ac.ci.CIMask2D.masked_front_edges_and_trails_from_frame_ci(
            frame_ci=imaging_ci_7x7.image,
            mask=mask,
            settings=ac.ci.SettingsCIMask2D(parallel_trails_rows=(0, 1)),
        )

        assert (
            ci_mask
            == np.array(
                [
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                    [False, True, True, True, True, False, False],
                    [False, False, False, False, False, False, False],
                ]
            )
        ).all()

        ci_mask = ac.ci.CIMask2D.masked_front_edges_and_trails_from_frame_ci(
            frame_ci=imaging_ci_7x7.image,
            mask=mask,
            settings=ac.ci.SettingsCIMask2D(serial_front_edge_columns=(0, 1)),
        )

        assert (
            ci_mask
            == np.array(
                [
                    [False, False, False, False, False, False, False],
                    [False, True, False, False, False, False, False],
                    [False, True, False, False, False, False, False],
                    [False, True, False, False, False, False, False],
                    [False, True, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                ]
            )
        ).all()

        ci_mask = ac.ci.CIMask2D.masked_front_edges_and_trails_from_frame_ci(
            frame_ci=imaging_ci_7x7.image,
            mask=mask,
            settings=ac.ci.SettingsCIMask2D(serial_trails_columns=(0, 1)),
        )

        assert (
            ci_mask
            == np.array(
                [
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, True, False],
                    [False, False, False, False, False, True, False],
                    [False, False, False, False, False, True, False],
                    [False, False, False, False, False, True, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                ]
            )
        ).all()
