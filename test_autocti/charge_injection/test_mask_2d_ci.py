import numpy as np

import autocti as ac


class MockPattern(object):
    def __init__(self):
        pass


class TestMask2DCI:
    def test__masked_parallel_front_edge_from_layout(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(10, 3), normalization=1.0, region_list=[(1, 4, 0, 3)]
        )

        mask = ac.ci.Mask2DCI.masked_parallel_front_edge_from_layout(
            layout=layout,
            settings=ac.ci.SettingsMask2DCI(parallel_front_edge_rows=(0, 2)),
            pixel_scales=0.1,
        )

        assert type(mask) == ac.ci.Mask2DCI

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

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(10, 3), normalization=1.0, region_list=[(1, 4, 0, 3)]
        )

        mask = ac.ci.Mask2DCI.masked_parallel_front_edge_from_layout(
            layout=layout,
            settings=ac.ci.SettingsMask2DCI(parallel_front_edge_rows=(0, 2)),
            pixel_scales=0.1,
            invert=True,
        )

        assert type(mask) == ac.ci.Mask2DCI

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

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(10, 3),
            normalization=1.0,
            region_list=[(1, 4, 0, 1), (1, 4, 2, 3)],
        )

        mask = ac.ci.Mask2DCI.masked_parallel_front_edge_from_layout(
            layout=layout,
            settings=ac.ci.SettingsMask2DCI(parallel_front_edge_rows=(0, 2)),
            pixel_scales=0.1,
        )

        assert type(mask) == ac.ci.Mask2DCI

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

    def test__masked_parallel_trails_from_layout(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(10, 3), normalization=1.0, region_list=[(1, 4, 0, 3)]
        )

        mask = ac.ci.Mask2DCI.masked_parallel_trails_from_layout(
            layout=layout,
            settings=ac.ci.SettingsMask2DCI(parallel_trails_rows=(0, 4)),
            pixel_scales=0.1,
        )

        assert type(mask) == ac.ci.Mask2DCI

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

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(10, 3), normalization=1.0, region_list=[(1, 4, 0, 3)]
        )

        mask = ac.ci.Mask2DCI.masked_parallel_trails_from_layout(
            layout=layout,
            settings=ac.ci.SettingsMask2DCI(parallel_trails_rows=(0, 4)),
            pixel_scales=0.1,
            invert=True,
        )

        assert type(mask) == ac.ci.Mask2DCI

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

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(10, 3),
            normalization=1.0,
            region_list=[(1, 4, 0, 1), (1, 4, 2, 3)],
        )

        mask = ac.ci.Mask2DCI.masked_parallel_trails_from_layout(
            layout=layout,
            settings=ac.ci.SettingsMask2DCI(parallel_trails_rows=(0, 4)),
            pixel_scales=0.1,
        )

        assert type(mask) == ac.ci.Mask2DCI

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

    def test__masked_serial_front_edge_from_layout(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(3, 10), normalization=1.0, region_list=[(0, 3, 1, 4)]
        )

        mask = ac.ci.Mask2DCI.masked_serial_front_edge_from_layout(
            layout=layout,
            settings=ac.ci.SettingsMask2DCI(serial_front_edge_columns=(0, 2)),
            pixel_scales=0.1,
        )

        assert type(mask) == ac.ci.Mask2DCI

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

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(3, 10), normalization=1.0, region_list=[(0, 3, 1, 4)]
        )

        mask = ac.ci.Mask2DCI.masked_serial_front_edge_from_layout(
            layout=layout,
            settings=ac.ci.SettingsMask2DCI(serial_front_edge_columns=(0, 2)),
            pixel_scales=0.1,
            invert=True,
        )

        assert type(mask) == ac.ci.Mask2DCI

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

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(3, 10),
            normalization=1.0,
            region_list=[(0, 1, 1, 4), (2, 3, 1, 4)],
        )

        mask = ac.ci.Mask2DCI.masked_serial_front_edge_from_layout(
            layout=layout,
            settings=ac.ci.SettingsMask2DCI(serial_front_edge_columns=(0, 3)),
            pixel_scales=0.1,
        )

        assert type(mask) == ac.ci.Mask2DCI

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

    def test__masked_serial_trails_from_layout(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(3, 10),
            normalization=1.0,
            region_list=[(0, 3, 1, 4)],
            serial_overscan=(0, 3, 8, 10),
        )

        mask = ac.ci.Mask2DCI.masked_serial_trails_from_layout(
            layout=layout,
            settings=ac.ci.SettingsMask2DCI(serial_trails_columns=(0, 6)),
            pixel_scales=0.1,
        )

        assert type(mask) == ac.ci.Mask2DCI

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

        mask = ac.ci.Mask2DCI.masked_serial_trails_from_layout(
            layout=layout,
            settings=ac.ci.SettingsMask2DCI(serial_trails_columns=(0, 6)),
            pixel_scales=0.1,
            invert=True,
        )

        assert type(mask) == ac.ci.Mask2DCI

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

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(3, 10),
            normalization=1.0,
            region_list=[(0, 1, 1, 4), (2, 3, 1, 4)],
            serial_overscan=(0, 3, 8, 10),
        )

        mask = ac.ci.Mask2DCI.masked_serial_trails_from_layout(
            layout=layout,
            settings=ac.ci.SettingsMask2DCI(serial_trails_columns=(0, 6)),
            pixel_scales=0.1,
        )

        assert type(mask) == ac.ci.Mask2DCI

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

    def test__masked_front_edges_and_trails_from_frame_ci(self, imaging_ci_7x7):

        unmasked = ac.ci.Mask2DCI.unmasked(
            shape_native=imaging_ci_7x7.shape_native, pixel_scales=1.0
        )

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(7, 7),
            normalization=1.0,
            region_list=[(1, 5, 1, 5)],
            serial_overscan=(0, 7, 6, 7),
        )

        mask = ac.ci.Mask2DCI.masked_front_edges_and_trails_from_frame_ci(
            layout=layout,
            mask=unmasked,
            settings=ac.ci.SettingsMask2DCI(parallel_front_edge_rows=(0, 1)),
            pixel_scales=0.1,
        )

        assert (
            mask
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

        mask = ac.ci.Mask2DCI.masked_front_edges_and_trails_from_frame_ci(
            layout=layout,
            mask=unmasked,
            settings=ac.ci.SettingsMask2DCI(parallel_trails_rows=(0, 1)),
            pixel_scales=0.1,
        )

        assert (
            mask
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

        mask = ac.ci.Mask2DCI.masked_front_edges_and_trails_from_frame_ci(
            layout=layout,
            mask=unmasked,
            settings=ac.ci.SettingsMask2DCI(serial_front_edge_columns=(0, 1)),
            pixel_scales=0.1,
        )

        assert (
            mask
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

        mask = ac.ci.Mask2DCI.masked_front_edges_and_trails_from_frame_ci(
            layout=layout,
            mask=unmasked,
            settings=ac.ci.SettingsMask2DCI(serial_trails_columns=(0, 1)),
            pixel_scales=0.1,
        )

        assert (
            mask
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
