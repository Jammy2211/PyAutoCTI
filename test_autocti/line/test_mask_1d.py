import numpy as np

import autocti as ac


class MockPattern(object):
    def __init__(self):
        pass


def test__masked_front_edge_from_layout():

    layout = ac.Layout1DLine(shape_1d=(5,), region_list=[(1, 4)])

    mask = ac.Mask1DLine.masked_front_edge_from_layout(
        layout=layout,
        settings=ac.SettingsMask1DLine(front_edge_pixels=(0, 2)),
        pixel_scales=0.1,
    )

    assert type(mask) == ac.Mask1DLine

    assert (mask == np.array([False, True, True, False, False])).all()

    layout = ac.Layout1DLine(shape_1d=(9,), region_list=[(1, 4), (6, 9)])

    mask = ac.Mask1DLine.masked_front_edge_from_layout(
        layout=layout,
        settings=ac.SettingsMask1DLine(front_edge_pixels=(1, 3)),
        pixel_scales=0.1,
    )

    assert type(mask) == ac.Mask1DLine

    assert (
        mask == np.array([False, False, True, True, False, False, False, True, True])
    ).all()


def test__masked_trails_from_layout():

    layout = ac.Layout1DLine(shape_1d=(5,), region_list=[(1, 3)])

    mask = ac.Mask1DLine.masked_trails_from_layout(
        layout=layout,
        settings=ac.SettingsMask1DLine(trails_pixels=(0, 2)),
        pixel_scales=0.1,
    )

    assert type(mask) == ac.Mask1DLine

    assert (mask == np.array([False, False, False, True, True])).all()

    layout = ac.Layout1DLine(shape_1d=(12,), region_list=[(1, 4), (7, 9)])

    mask = ac.Mask1DLine.masked_trails_from_layout(
        layout=layout,
        settings=ac.SettingsMask1DLine(trails_pixels=(1, 3)),
        pixel_scales=0.1,
    )

    assert type(mask) == ac.Mask1DLine

    assert (
        mask
        == np.array(
            [
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
            ]
        )
    ).all()


def test__masked_front_edges_and_epers_from_layout():

    unmasked = ac.Mask1DLine.unmasked(shape_slim=(5,), pixel_scales=1.0)

    layout = ac.Layout1DLine(shape_1d=(5,), region_list=[(1, 3)])

    mask = ac.Mask1DLine.masked_fprs_and_epers_from(
        mask=unmasked,
        layout=layout,
        settings=ac.SettingsMask1DLine(front_edge_pixels=(1, 2), trails_pixels=(0, 1)),
        pixel_scales=0.1,
    )

    assert (mask == np.array([False, False, True, True, False])).all()
