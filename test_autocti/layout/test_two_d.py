import pytest
import autocti as ac
from autocti import exc


def test__check_layout_dimensions__layout_has_more_rows_than_image__1_region():

    with pytest.raises(exc.LayoutException):
        ac.Layout2DCI(shape_2d=(2, 6), region_list=([(0, 3, 0, 1)]))

    with pytest.raises(exc.LayoutException):
        ac.Layout2DCI(shape_2d=(6, 2), region_list=([(0, 1, 0, 3)]))

    with pytest.raises(exc.LayoutException):
        ac.Layout2DCI(shape_2d=(2, 6), region_list=([(0, 3, 0, 1), (0, 1, 0, 3)]))

    with pytest.raises(exc.LayoutException):
        ac.Layout2DCI(shape_2d=(6, 2), region_list=([(0, 3, 0, 1), (0, 1, 0, 3)]))

    with pytest.raises(exc.RegionException):
        ac.Layout2DCI(shape_2d=(3, 3), region_list=([(-1, 0, 0, 0)]))

    with pytest.raises(exc.RegionException):
        ac.Layout2DCI(shape_2d=(3, 3), region_list=([(0, -1, 0, 0)]))

    with pytest.raises(exc.RegionException):
        ac.Layout2DCI(shape_2d=(3, 3), region_list=([(0, 0, -1, 0)]))

    with pytest.raises(exc.RegionException):
        ac.Layout2DCI(shape_2d=(3, 3), region_list=([(0, 0, 0, -1)]))


def test__rows_between_region_list():

    layout = ac.Layout2DCI(shape_2d=(5, 5), region_list=[(1, 2, 1, 2)])

    assert layout.pixels_between_regions == []

    layout = ac.Layout2DCI(shape_2d=(5, 5), region_list=[(1, 2, 1, 2), (3, 4, 3, 4)])

    assert layout.pixels_between_regions == [1]

    layout = ac.Layout2DCI(shape_2d=(5, 5), region_list=[(1, 2, 1, 2), (4, 5, 3, 4)])

    assert layout.pixels_between_regions == [2]

    layout = ac.Layout2DCI(
        shape_2d=(10, 10), region_list=[(1, 2, 1, 2), (4, 5, 3, 4), (8, 9, 3, 4)]
    )

    assert layout.pixels_between_regions == [2, 3]


def test__rows_within_region_list():

    layout = ac.Layout2DCI(shape_2d=(5, 5), region_list=[(1, 2, 1, 2)])

    assert layout.pixels_within_regions == [1]

    layout = ac.Layout2DCI(shape_2d=(5, 5), region_list=[(1, 2, 1, 2), (3, 4, 3, 4)])

    assert layout.pixels_within_regions == [1, 1]

    layout = ac.Layout2DCI(shape_2d=(6, 6), region_list=[(1, 2, 1, 2), (4, 6, 3, 4)])

    assert layout.pixels_within_regions == [1, 2]

    layout = ac.Layout2DCI(
        shape_2d=(10, 10), region_list=[(1, 2, 3, 4), (4, 6, 3, 4), (6, 9, 3, 4)]
    )

    assert layout.pixels_within_regions == [1, 2, 3]


def test__serial_eper_pixels(layout_ci_7x7):

    layout = ac.Layout2DCI(
        shape_2d=(10, 10),
        region_list=[(1, 2, 1, 2)],
        serial_overscan=ac.Region2D((0, 1, 0, 10)),
        serial_prescan=ac.Region2D((0, 1, 0, 1)),
        parallel_overscan=ac.Region2D((0, 1, 0, 1)),
    )

    assert layout.serial_eper_pixels == 10

    layout = ac.Layout2DCI(
        shape_2d=(50, 50),
        region_list=[(1, 2, 1, 2)],
        serial_overscan=ac.Region2D((0, 1, 0, 50)),
        serial_prescan=ac.Region2D((0, 1, 0, 1)),
        parallel_overscan=ac.Region2D((0, 1, 0, 1)),
    )

    assert layout.serial_eper_pixels == 50


def test__parallel_eper_size_to_array_edge():

    layout = ac.Layout2DCI(
        shape_2d=(5, 100), region_list=[ac.Region2D(region=(0, 3, 0, 3))]
    )

    assert layout.parallel_rows_to_array_edge == 2

    layout = ac.Layout2DCI(
        shape_2d=(7, 100), region_list=[ac.Region2D(region=(0, 3, 0, 3))]
    )

    assert layout.parallel_rows_to_array_edge == 4

    layout = ac.Layout2DCI(
        shape_2d=(15, 100),
        region_list=[
            ac.Region2D(region=(0, 2, 0, 3)),
            ac.Region2D(region=(5, 8, 0, 3)),
            ac.Region2D(region=(11, 14, 0, 3)),
        ],
    )

    assert layout.parallel_rows_to_array_edge == 1

    layout = ac.Layout2DCI(
        shape_2d=(20, 100),
        region_list=[
            ac.Region2D(region=(0, 2, 0, 3)),
            ac.Region2D(region=(5, 8, 0, 3)),
            ac.Region2D(region=(11, 14, 0, 3)),
        ],
    )

    assert layout.parallel_rows_to_array_edge == 6


def test__with_extracted_regions__region_list_are_extracted_correctly():

    layout = ac.Layout2DCI(shape_2d=(5, 3), region_list=[(0, 2, 0, 2)])

    layout_extracted = layout.with_extracted_regions(
        extraction_region=ac.Region2D((0, 2, 0, 2))
    )

    assert layout_extracted.region_list == [(0, 2, 0, 2)]

    layout_extracted = layout.with_extracted_regions(
        extraction_region=ac.Region2D((0, 1, 0, 1))
    )

    assert layout_extracted.region_list == [(0, 1, 0, 1)]

    layout = ac.Layout2DCI(shape_2d=(10, 5), region_list=[(2, 4, 2, 4), (0, 1, 0, 1)])

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


def test__smallest_parallel_eper_rows_to_frame_edge():

    layout = ac.Layout2DCI(shape_2d=(10, 5), region_list=[(0, 3, 0, 3), (5, 7, 0, 3)])

    assert layout.smallest_parallel_rows_between_ci_regions == 2

    layout = ac.Layout2DCI(shape_2d=(8, 5), region_list=[(0, 3, 0, 3), (5, 7, 0, 3)])

    assert layout.smallest_parallel_rows_between_ci_regions == 1
