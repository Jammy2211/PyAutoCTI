import numpy as np

import autocti as ac


class MockPattern(object):
    def __init__(self):
        pass


class TestSettingsCIMask:
    def test__parallel_front_edge_rows_tag(self):

        settings = ac.ci.SettingsCIMask(parallel_front_edge_rows=None)
        assert settings.parallel_front_edge_rows_tag == ""
        settings = ac.ci.SettingsCIMask(parallel_front_edge_rows=(0, 5))
        assert settings.parallel_front_edge_rows_tag == "__par_front_mask_rows_(0,5)"
        settings = ac.ci.SettingsCIMask(parallel_front_edge_rows=(10, 20))
        assert settings.parallel_front_edge_rows_tag == "__par_front_mask_rows_(10,20)"

    def test__parallel_trails_rows_tag(self):

        settings = ac.ci.SettingsCIMask(parallel_trails_rows=None)
        assert settings.parallel_trails_rows_tag == ""

        settings = ac.ci.SettingsCIMask(parallel_trails_rows=(0, 5))
        assert settings.parallel_trails_rows_tag == "__par_trails_mask_rows_(0,5)"
        settings = ac.ci.SettingsCIMask(parallel_trails_rows=(10, 20))
        assert settings.parallel_trails_rows_tag == "__par_trails_mask_rows_(10,20)"

    def test__serial_front_edge_columns_tag(self):

        settings = ac.ci.SettingsCIMask(serial_front_edge_columns=None)
        assert settings.serial_front_edge_columns_tag == ""

        settings = ac.ci.SettingsCIMask(serial_front_edge_columns=(0, 5))
        assert settings.serial_front_edge_columns_tag == "__ser_front_mask_col_(0,5)"

        settings = ac.ci.SettingsCIMask(serial_front_edge_columns=(10, 20))
        assert settings.serial_front_edge_columns_tag == "__ser_front_mask_col_(10,20)"

    def test__serial_trails_columns_tag(self):

        settings = ac.ci.SettingsCIMask(serial_trails_columns=None)
        assert settings.serial_trails_columns_tag == ""

        settings = ac.ci.SettingsCIMask(serial_trails_columns=(0, 5))
        assert settings.serial_trails_columns_tag == "__ser_trails_mask_col_(0,5)"

        settings = ac.ci.SettingsCIMask(serial_trails_columns=(10, 20))
        assert settings.serial_trails_columns_tag == "__ser_trails_mask_col_(10,20)"

    def test__tag(self):

        settings = ac.ci.SettingsCIMask(
            parallel_front_edge_rows=(10, 20),
            parallel_trails_rows=(10, 20),
            serial_front_edge_columns=(10, 20),
        )

        assert (
            settings.tag
            == "ci_mask[__par_front_mask_rows_(10,20)__par_trails_mask_rows_(10,20)__ser_front_mask_col_(10,20)]"
        )


class TestMaskedParallelFrontEdge:
    def test__mask_only_contains_front_edge(self):

        pattern = ac.ci.CIPatternUniform(normalization=1.0, regions=[(1, 4, 0, 3)])

        frame = ac.ci.CIFrame.manual(
            array=np.ones((10, 3)), roe_corner=(1, 0), ci_pattern=pattern
        )

        mask = ac.ci.CIMask.masked_parallel_front_edge_from_ci_frame(
            ci_frame=frame,
            settings=ac.ci.SettingsCIMask(parallel_front_edge_rows=(0, 2)),
        )

        assert type(mask) == ac.ci.CIMask

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

        pattern = ac.ci.CIPatternUniform(normalization=1.0, regions=[(1, 4, 0, 3)])

        frame = ac.ci.CIFrame.manual(
            array=np.ones((10, 3)), roe_corner=(1, 0), ci_pattern=pattern
        )

        mask = ac.ci.CIMask.masked_parallel_front_edge_from_ci_frame(
            ci_frame=frame,
            settings=ac.ci.SettingsCIMask(parallel_front_edge_rows=(0, 2)),
            invert=True,
        )

        assert type(mask) == ac.ci.CIMask

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

        pattern = ac.ci.CIPatternUniform(
            normalization=1.0, regions=[(1, 4, 0, 1), (1, 4, 2, 3)]
        )

        frame = ac.ci.CIFrame.manual(
            array=np.ones((10, 3)), roe_corner=(1, 0), ci_pattern=pattern
        )

        mask = ac.ci.CIMask.masked_parallel_front_edge_from_ci_frame(
            ci_frame=frame,
            settings=ac.ci.SettingsCIMask(parallel_front_edge_rows=(0, 2)),
        )

        assert type(mask) == ac.ci.CIMask

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

        pattern = ac.ci.CIPatternUniform(normalization=1.0, regions=[(1, 4, 0, 3)])

        frame = ac.ci.CIFrame.manual(
            array=np.ones((10, 3)), roe_corner=(1, 0), ci_pattern=pattern
        )

        mask = ac.ci.CIMask.masked_parallel_trails_from_ci_frame(
            ci_frame=frame, settings=ac.ci.SettingsCIMask(parallel_trails_rows=(0, 4))
        )

        assert type(mask) == ac.ci.CIMask

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

        pattern = ac.ci.CIPatternUniform(normalization=1.0, regions=[(1, 4, 0, 3)])

        frame = ac.ci.CIFrame.manual(
            array=np.ones((10, 3)), roe_corner=(1, 0), ci_pattern=pattern
        )

        mask = ac.ci.CIMask.masked_parallel_trails_from_ci_frame(
            ci_frame=frame,
            settings=ac.ci.SettingsCIMask(parallel_trails_rows=(0, 4)),
            invert=True,
        )

        assert type(mask) == ac.ci.CIMask

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

        pattern = ac.ci.CIPatternUniform(
            normalization=1.0, regions=[(1, 4, 0, 1), (1, 4, 2, 3)]
        )

        frame = ac.ci.CIFrame.manual(
            array=np.ones((10, 3)), roe_corner=(1, 0), ci_pattern=pattern
        )

        mask = ac.ci.CIMask.masked_parallel_trails_from_ci_frame(
            ci_frame=frame, settings=ac.ci.SettingsCIMask(parallel_trails_rows=(0, 4))
        )

        assert type(mask) == ac.ci.CIMask

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

        pattern = ac.ci.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        frame = ac.ci.CIFrame.manual(
            array=np.ones((3, 10)), roe_corner=(1, 0), ci_pattern=pattern
        )

        mask = ac.ci.CIMask.masked_serial_front_edge_from_ci_frame(
            ci_frame=frame,
            settings=ac.ci.SettingsCIMask(serial_front_edge_columns=(0, 2)),
        )

        assert type(mask) == ac.ci.CIMask

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
            array=np.ones((3, 10)), roe_corner=(1, 0), ci_pattern=pattern
        )

        mask = ac.ci.CIMask.masked_serial_front_edge_from_ci_frame(
            ci_frame=frame,
            settings=ac.ci.SettingsCIMask(serial_front_edge_columns=(0, 2)),
            invert=True,
        )

        assert type(mask) == ac.ci.CIMask

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
            array=np.ones((3, 10)), roe_corner=(1, 0), ci_pattern=pattern
        )

        mask = ac.ci.CIMask.masked_serial_front_edge_from_ci_frame(
            ci_frame=frame,
            settings=ac.ci.SettingsCIMask(serial_front_edge_columns=(0, 3)),
        )

        assert type(mask) == ac.ci.CIMask

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
            array=np.ones((3, 10)), roe_corner=(1, 0), ci_pattern=pattern
        )

        mask = ac.ci.CIMask.masked_serial_trails_from_ci_frame(
            ci_frame=frame, settings=ac.ci.SettingsCIMask(serial_trails_columns=(0, 6))
        )

        assert type(mask) == ac.ci.CIMask

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
            array=np.ones((3, 10)), roe_corner=(1, 0), ci_pattern=pattern
        )

        mask = ac.ci.CIMask.masked_serial_trails_from_ci_frame(
            ci_frame=frame,
            settings=ac.ci.SettingsCIMask(serial_trails_columns=(0, 6)),
            invert=True,
        )

        assert type(mask) == ac.ci.CIMask

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
            array=np.ones((3, 10)), roe_corner=(1, 0), ci_pattern=pattern
        )

        mask = ac.ci.CIMask.masked_serial_trails_from_ci_frame(
            ci_frame=frame, settings=ac.ci.SettingsCIMask(serial_trails_columns=(0, 6))
        )

        assert type(mask) == ac.ci.CIMask

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
    def test__masks_uses_front_edge_and_trails_parameters(self, ci_imaging_7x7):

        mask = ac.ci.CIMask.unmasked(shape_2d=ci_imaging_7x7.shape_2d)

        ci_mask = ac.ci.CIMask.masked_front_edges_and_trails_from_ci_frame(
            ci_frame=ci_imaging_7x7.image,
            mask=mask,
            settings=ac.ci.SettingsCIMask(parallel_front_edge_rows=(0, 1)),
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

        ci_mask = ac.ci.CIMask.masked_front_edges_and_trails_from_ci_frame(
            ci_frame=ci_imaging_7x7.image,
            mask=mask,
            settings=ac.ci.SettingsCIMask(parallel_trails_rows=(0, 1)),
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

        ci_mask = ac.ci.CIMask.masked_front_edges_and_trails_from_ci_frame(
            ci_frame=ci_imaging_7x7.image,
            mask=mask,
            settings=ac.ci.SettingsCIMask(serial_front_edge_columns=(0, 1)),
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

        ci_mask = ac.ci.CIMask.masked_front_edges_and_trails_from_ci_frame(
            ci_frame=ci_imaging_7x7.image,
            mask=mask,
            settings=ac.ci.SettingsCIMask(serial_trails_columns=(0, 1)),
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
