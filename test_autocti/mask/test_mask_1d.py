import numpy as np

import autocti as ac


def test__masked_fpr_from_layout():

    layout = ac.Layout1D(shape_1d=(5,), region_list=[(1, 4)])

    mask = ac.Mask1D.masked_fpr_from_layout(
        layout=layout, settings=ac.SettingsMask1D(fpr_pixels=(0, 2)), pixel_scales=0.1
    )

    assert type(mask) == ac.Mask1D

    assert (mask == np.array([False, True, True, False, False])).all()

    layout = ac.Layout1D(shape_1d=(9,), region_list=[(1, 4), (6, 9)])

    mask = ac.Mask1D.masked_fpr_from_layout(
        layout=layout, settings=ac.SettingsMask1D(fpr_pixels=(1, 3)), pixel_scales=0.1
    )

    assert type(mask) == ac.Mask1D

    assert (
        mask == np.array([False, False, True, True, False, False, False, True, True])
    ).all()


def test__masked_eper_from_layout():

    layout = ac.Layout1D(shape_1d=(5,), region_list=[(1, 3)])

    mask = ac.Mask1D.masked_eper_from_layout(
        layout=layout, settings=ac.SettingsMask1D(eper_pixels=(0, 2)), pixel_scales=0.1
    )

    assert type(mask) == ac.Mask1D

    assert (mask == np.array([False, False, False, True, True])).all()

    layout = ac.Layout1D(shape_1d=(12,), region_list=[(1, 4), (7, 9)])

    mask = ac.Mask1D.masked_eper_from_layout(
        layout=layout, settings=ac.SettingsMask1D(eper_pixels=(1, 3)), pixel_scales=0.1
    )

    assert type(mask) == ac.Mask1D

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


def test__masked_fpr_and_eper_from_layout():

    unmasked = ac.Mask1D.unmasked(shape_slim=(5,), pixel_scales=1.0)

    layout = ac.Layout1D(shape_1d=(5,), region_list=[(1, 3)])

    mask = ac.Mask1D.masked_fpr_and_eper_from(
        mask=unmasked,
        layout=layout,
        settings=ac.SettingsMask1D(fpr_pixels=(1, 2), eper_pixels=(0, 1)),
        pixel_scales=0.1,
    )

    assert (mask == np.array([False, False, True, True, False])).all()
