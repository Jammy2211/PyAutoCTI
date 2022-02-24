import numpy as np
import pytest
import autocti as ac
from autocti.charge_injection.layout import region_list_ci_from
from autocti import exc


def test__check_layout_dimensions__layout_has_more_rows_than_image__1_region():

    with pytest.raises(exc.LayoutException):
        ac.ci.Layout2DCI(shape_2d=(2, 6), region_list=([(0, 3, 0, 1)]))

    with pytest.raises(exc.LayoutException):
        ac.ci.Layout2DCI(shape_2d=(6, 2), region_list=([(0, 1, 0, 3)]))

    with pytest.raises(exc.LayoutException):
        ac.ci.Layout2DCI(shape_2d=(2, 6), region_list=([(0, 3, 0, 1), (0, 1, 0, 3)]))

    with pytest.raises(exc.LayoutException):
        ac.ci.Layout2DCI(shape_2d=(6, 2), region_list=([(0, 3, 0, 1), (0, 1, 0, 3)]))

    with pytest.raises(exc.RegionException):
        ac.ci.Layout2DCI(shape_2d=(3, 3), region_list=([(-1, 0, 0, 0)]))

    with pytest.raises(exc.RegionException):
        ac.ci.Layout2DCI(shape_2d=(3, 3), region_list=([(0, -1, 0, 0)]))

    with pytest.raises(exc.RegionException):
        ac.ci.Layout2DCI(shape_2d=(3, 3), region_list=([(0, 0, -1, 0)]))

    with pytest.raises(exc.RegionException):
        ac.ci.Layout2DCI(shape_2d=(3, 3), region_list=([(0, 0, 0, -1)]))


def test__rows_between_region_list():

    layout = ac.ci.Layout2DCI(shape_2d=(5, 5), region_list=[(1, 2, 1, 2)])

    assert layout.pixels_between_regions == []

    layout = ac.ci.Layout2DCI(shape_2d=(5, 5), region_list=[(1, 2, 1, 2), (3, 4, 3, 4)])

    assert layout.pixels_between_regions == [1]

    layout = ac.ci.Layout2DCI(shape_2d=(5, 5), region_list=[(1, 2, 1, 2), (4, 5, 3, 4)])

    assert layout.pixels_between_regions == [2]

    layout = ac.ci.Layout2DCI(
        shape_2d=(10, 10), region_list=[(1, 2, 1, 2), (4, 5, 3, 4), (8, 9, 3, 4)]
    )

    assert layout.pixels_between_regions == [2, 3]


def test__serial_trails_pixels(layout_ci_7x7):

    layout = ac.ci.Layout2DCI(
        shape_2d=(10, 10),
        region_list=[(1, 2, 1, 2)],
        serial_overscan=ac.Region2D((0, 1, 0, 10)),
        serial_prescan=ac.Region2D((0, 1, 0, 1)),
        parallel_overscan=ac.Region2D((0, 1, 0, 1)),
    )

    assert layout.serial_eper_pixels == 10

    layout = ac.ci.Layout2DCI(
        shape_2d=(50, 50),
        region_list=[(1, 2, 1, 2)],
        serial_overscan=ac.Region2D((0, 1, 0, 50)),
        serial_prescan=ac.Region2D((0, 1, 0, 1)),
        parallel_overscan=ac.Region2D((0, 1, 0, 1)),
    )

    assert layout.serial_eper_pixels == 50


def test__parallel_eper_size_to_array_edge():

    layout = ac.ci.Layout2DCI(
        shape_2d=(5, 100), region_list=[ac.Region2D(region=(0, 3, 0, 3))]
    )

    assert layout.parallel_rows_to_array_edge == 2

    layout = ac.ci.Layout2DCI(
        shape_2d=(7, 100), region_list=[ac.Region2D(region=(0, 3, 0, 3))]
    )

    assert layout.parallel_rows_to_array_edge == 4

    layout = ac.ci.Layout2DCI(
        shape_2d=(15, 100),
        region_list=[
            ac.Region2D(region=(0, 2, 0, 3)),
            ac.Region2D(region=(5, 8, 0, 3)),
            ac.Region2D(region=(11, 14, 0, 3)),
        ],
    )

    assert layout.parallel_rows_to_array_edge == 1

    layout = ac.ci.Layout2DCI(
        shape_2d=(20, 100),
        region_list=[
            ac.Region2D(region=(0, 2, 0, 3)),
            ac.Region2D(region=(5, 8, 0, 3)),
            ac.Region2D(region=(11, 14, 0, 3)),
        ],
    )

    assert layout.parallel_rows_to_array_edge == 6


def test__with_extracted_regions__region_list_are_extracted_correctly():

    layout = ac.ci.Layout2DCI(shape_2d=(5, 3), region_list=[(0, 2, 0, 2)])

    layout_extracted = layout.with_extracted_regions(
        extraction_region=ac.Region2D((0, 2, 0, 2))
    )

    assert layout_extracted.region_list == [(0, 2, 0, 2)]

    layout_extracted = layout.with_extracted_regions(
        extraction_region=ac.Region2D((0, 1, 0, 1))
    )

    assert layout_extracted.region_list == [(0, 1, 0, 1)]

    layout = ac.ci.Layout2DCI(
        shape_2d=(10, 5), region_list=[(2, 4, 2, 4), (0, 1, 0, 1)]
    )

    layout_extracted = layout.with_extracted_regions(
        extraction_region=ac.Region2D((0, 3, 0, 3))
    )

    assert layout_extracted.region_list == [(2, 3, 2, 3), (0, 1, 0, 1)]

    layout_extracted = layout.with_extracted_regions(
        extraction_region=ac.Region2D((2, 5, 2, 5))
    )

    assert layout_extracted.region_list == [(0, 2, 0, 2)]

    layout_extracted = layout.with_extracted_regions(
        extraction_region=ac.Region2D((8, 9, 8, 9))
    )

    assert layout_extracted.region_list == None


def test__serial_overscan_above_epers_array_2d_from():
    layout = ac.ci.Layout2DCI(
        shape_2d=(5, 4),
        region_list=[(1, 2, 1, 3), (3, 4, 1, 3)],
        serial_prescan=(0, 5, 0, 1),
        serial_overscan=(0, 5, 3, 4),
    )

    array = ac.Array2D.manual(
        array=[
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
            [12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0],
        ],
        pixel_scales=1.0,
    )

    new_array = layout.serial_overscan_above_epers_array_2d_from(array=array)

    assert (
        new_array
        == np.array(
            [
                [0.0, 0.0, 0.0, 3.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 11.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 19.0],
            ]
        )
    ).all()

    layout = ac.ci.Layout2DCI(
        shape_2d=(4, 4),
        region_list=[(0, 1, 0, 2), (2, 3, 0, 2)],
        serial_overscan=(0, 4, 2, 4),
    )

    array = ac.Array2D.manual(
        array=[
            [0.0, 1.0, 2.0, 0.5],
            [3.0, 4.0, 5.0, 0.5],
            [6.0, 7.0, 8.0, 0.5],
            [9.0, 10.0, 11.0, 0.5],
        ],
        pixel_scales=1.0,
    )

    new_array = layout.serial_overscan_above_epers_array_2d_from(array=array)

    assert (
        new_array
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.5],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 11.0, 0.5],
            ]
        )
    ).all()


def test__smallest_parallel_epers_rows_to_frame_edge():

    layout = ac.ci.Layout2DCI(
        shape_2d=(10, 5), region_list=[(0, 3, 0, 3), (5, 7, 0, 3)]
    )

    assert layout.smallest_parallel_rows_between_ci_regions == 2

    layout = ac.ci.Layout2DCI(shape_2d=(8, 5), region_list=[(0, 3, 0, 3), (5, 7, 0, 3)])

    assert layout.smallest_parallel_rows_between_ci_regions == 1


def test__region_list_ci_from():

    region_list_ci = region_list_ci_from(
        injection_on=10,
        injection_off=10,
        injection_total=1,
        parallel_size=10,
        serial_prescan_size=1,
        serial_size=10,
        serial_overscan_size=1,
        roe_corner=(1, 0),
    )

    assert region_list_ci == [(0, 10, 1, 9)]

    region_list_ci = region_list_ci_from(
        injection_on=10,
        injection_off=10,
        injection_total=2,
        parallel_size=30,
        serial_prescan_size=2,
        serial_size=11,
        serial_overscan_size=4,
        roe_corner=(1, 0),
    )

    assert region_list_ci == [(0, 10, 2, 7), (20, 30, 2, 7)]

    region_list_ci = region_list_ci_from(
        injection_on=5,
        injection_off=10,
        injection_total=3,
        parallel_size=35,
        serial_prescan_size=2,
        serial_size=11,
        serial_overscan_size=4,
        roe_corner=(1, 0),
    )

    assert region_list_ci == [(0, 5, 2, 7), (15, 20, 2, 7), (30, 35, 2, 7)]

    region_list_ci = region_list_ci_from(
        injection_on=200,
        injection_off=200,
        injection_total=5,
        parallel_size=2000,
        serial_prescan_size=51,
        serial_size=2128,
        serial_overscan_size=29,
        roe_corner=(1, 0),
    )

    assert region_list_ci == [
        (0, 200, 51, 2099),
        (400, 600, 51, 2099),
        (800, 1000, 51, 2099),
        (1200, 1400, 51, 2099),
        (1600, 1800, 51, 2099),
    ]
