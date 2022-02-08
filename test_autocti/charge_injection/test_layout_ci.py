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


def test__serial_trails_columns(layout_ci_7x7):

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


def test__array_2d_of_regions_from():

    layout = ac.ci.Layout2DCI(shape_2d=(5, 3), region_list=[(0, 3, 0, 3)])

    array = ac.Array2D.manual(
        array=[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        pixel_scales=1.0,
    )

    array_extracted = layout.array_2d_of_regions_from(array=array)

    assert (
        array_extracted
        == np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [0.0, 0.0, 0.0]]
        )
    ).all()

    layout = ac.ci.Layout2DCI(shape_2d=(5, 3), region_list=[(0, 1, 1, 2), (2, 3, 1, 3)])

    array = ac.Array2D.manual(
        array=[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        pixel_scales=1.0,
    )

    array_extracted = layout.array_2d_of_regions_from(array=array)

    assert (
        array_extracted
        == np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 7.0, 8.0], [0.0, 0.0, 0.0]]
        )
    ).all()


def test__array_2d_of_non_regions_from():

    layout = ac.ci.Layout2DCI(shape_2d=(5, 3), region_list=[(0, 3, 0, 3)])

    array = ac.Array2D.manual(
        array=[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        pixel_scales=1.0,
    )

    array_extracted = layout.array_2d_of_non_regions_from(array=array)

    assert (
        array_extracted
        == np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [9.0, 10.0, 11.0]]
        )
    ).all()

    layout = ac.ci.Layout2DCI(shape_2d=(5, 3), region_list=[(0, 1, 0, 3), (3, 4, 0, 3)])

    array = ac.Array2D.manual(
        array=[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0],
            [12.0, 13.0, 14.0],
        ],
        pixel_scales=1.0,
    )

    array_extracted = layout.array_2d_of_non_regions_from(array=array)

    assert (
        array_extracted
        == np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [0.0, 0.0, 0.0],
                [12.0, 13.0, 14.0],
            ]
        )
    ).all()


def test__array_2d_of_parallel_epers_from():

    layout = ac.ci.Layout2DCI(
        shape_2d=(5, 3),
        region_list=[(0, 3, 0, 3)],
        serial_prescan=(3, 5, 2, 3),
        serial_overscan=(3, 5, 0, 1),
    )

    array = ac.Array2D.manual(
        array=[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0],
            [12.0, 13.0, 14.0],
        ],
        pixel_scales=1.0,
    )

    array_extracted = layout.array_2d_of_parallel_epers_from(array=array)

    assert (
        array_extracted
        == np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 13.0, 0.0],
            ]
        )
    ).all()

    layout = ac.ci.Layout2DCI(
        shape_2d=(5, 3),
        region_list=[(0, 1, 0, 3), (3, 4, 0, 3)],
        serial_prescan=(1, 2, 0, 3),
        serial_overscan=(0, 1, 0, 1),
    )

    array_extracted = layout.array_2d_of_parallel_epers_from(array=array)

    assert (
        array_extracted.native
        == np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [6.0, 7.0, 8.0],
                [0.0, 0.0, 0.0],
                [12.0, 13.0, 14.0],
            ]
        )
    ).all()


def test__array_2d_of_parallel_fprs_and_epers_from():

    parallel_array = ac.Array2D.manual(
        array=[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],  # <- Front edge .
            [2.0, 2.0, 2.0],  # <- Next front edge row.
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
            [6.0, 6.0, 6.0],
            [7.0, 7.0, 7.0],
            [8.0, 8.0, 8.0],
            [9.0, 9.0, 9.0],
        ],
        pixel_scales=1.0,
    )

    layout = ac.ci.Layout2DCI(shape_2d=(10, 3), region_list=[(0, 4, 0, 3)])

    new_array = layout.array_2d_of_parallel_fprs_and_epers_from(
        array=parallel_array, fpr_range=(0, 2), trails_rows=(0, 2)
    )

    assert (
        new_array
        == np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [4.0, 4.0, 4.0],
                [5.0, 5.0, 5.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
    ).all()

    layout = ac.ci.Layout2DCI(
        shape_2d=(10, 3), region_list=[(0, 1, 0, 3), (3, 4, 0, 3)]
    )

    new_array = layout.array_2d_of_parallel_fprs_and_epers_from(
        array=parallel_array, fpr_range=(0, 1), trails_rows=(0, 1)
    )

    assert (
        new_array
        == np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
    ).all()


def test__array_2d_for_parallel_calibration_from():

    layout = ac.ci.Layout2DCI(shape_2d=(5, 3), region_list=[(0, 3, 0, 3)])

    array = ac.Array2D.manual(
        array=[
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ],
        pixel_scales=1.0,
    )

    extracted_array = layout.array_2d_for_parallel_calibration_from(
        array=array, columns=(0, 1)
    )

    assert (extracted_array == np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])).all()

    layout = ac.ci.Layout2DCI(shape_2d=(5, 3), region_list=[(0, 5, 0, 3)])

    extracted_array = layout.array_2d_for_parallel_calibration_from(
        array=array, columns=(1, 3)
    )

    assert (
        extracted_array.native
        == np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    ).all()


def test__mask_for_parallel_calibration_from():

    layout = ac.ci.Layout2DCI(shape_2d=(5, 3), region_list=[(0, 5, 0, 3)])

    mask = ac.ci.Mask2DCI.unmasked(shape_native=(5, 3), pixel_scales=1.0)

    mask[0, 1] = True

    extracted_mask = layout.mask_for_parallel_calibration_from(
        mask=mask, columns=(1, 3)
    )

    assert (
        extracted_mask
        == np.array(
            [
                [True, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
            ]
        )
    ).all()


def test__extracted_layout_2d_for_parallel_calibration_from():

    layout = ac.ci.Layout2DCI(shape_2d=(5, 3), region_list=[(0, 3, 0, 3)])

    extracted_layout = layout.extracted_layout_for_parallel_calibration_from(
        columns=(0, 1)
    )

    assert extracted_layout.region_list == [(0, 3, 0, 1)]

    layout = ac.ci.Layout2DCI(shape_2d=(5, 3), region_list=[(0, 5, 0, 3)])

    extracted_layout = layout.extracted_layout_for_parallel_calibration_from(
        columns=(1, 3)
    )

    assert extracted_layout.region_list == [(0, 5, 0, 2)]


def test__array_2d_of_serial_trails_from():

    layout = ac.ci.Layout2DCI(
        shape_2d=(4, 3), region_list=[(0, 4, 0, 2)], serial_overscan=(0, 4, 2, 3)
    )

    array = ac.Array2D.manual(
        array=[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        pixel_scales=1.0,
    )

    new_array = layout.array_2d_of_serial_trails_from(array=array)

    assert (
        new_array
        == np.array(
            [[0.0, 0.0, 2.0], [0.0, 0.0, 5.0], [0.0, 0.0, 8.0], [0.0, 0.0, 11.0]]
        )
    ).all()

    layout = ac.ci.Layout2DCI(
        shape_2d=(4, 4), region_list=[(0, 4, 0, 2)], serial_overscan=(0, 4, 2, 4)
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

    new_array = layout.array_2d_of_serial_trails_from(array=array)

    assert (
        new_array
        == np.array(
            [
                [0.0, 0.0, 2.0, 0.5],
                [0.0, 0.0, 5.0, 0.5],
                [0.0, 0.0, 8.0, 0.5],
                [0.0, 0.0, 11.0, 0.5],
            ]
        )
    ).all()


def test__array_2d_of_serial_overscan_above_trails_from():
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

    new_array = layout.array_2d_of_serial_overscan_above_trails_from(array=array)

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

    new_array = layout.array_2d_of_serial_overscan_above_trails_from(array=array)

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


def test__array_2d_of_serial_edges_and_epers_array():

    layout = ac.ci.Layout2DCI(shape_2d=(3, 4), region_list=[(0, 3, 0, 3)])

    array = ac.Array2D.manual(
        array=[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]],
        pixel_scales=1.0,
    )

    new_array = layout.array_2d_of_serial_edges_and_epers_array(
        array=array, front_edge_columns=(0, 1)
    )

    assert (
        new_array
        == np.array([[0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0]])
    ).all()

    new_array = layout.array_2d_of_serial_edges_and_epers_array(
        array=array, front_edge_columns=(0, 2)
    )

    assert (
        new_array
        == np.array([[0.0, 1.0, 0.0, 0.0], [4.0, 5.0, 0.0, 0.0], [8.0, 9.0, 0.0, 0.0]])
    ).all()

    layout = ac.ci.Layout2DCI(shape_2d=(3, 4), region_list=[(0, 3, 0, 2)])

    new_array = layout.array_2d_of_serial_edges_and_epers_array(
        array=array, trails_columns=(0, 1)
    )

    assert (
        new_array
        == np.array([[0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 6.0, 0.0], [0.0, 0.0, 10.0, 0.0]])
    ).all()

    new_array = layout.array_2d_of_serial_edges_and_epers_array(
        array=array, trails_columns=(0, 2)
    )

    assert (
        new_array
        == np.array(
            [[0.0, 0.0, 2.0, 3.0], [0.0, 0.0, 6.0, 7.0], [0.0, 0.0, 10.0, 11.0]]
        )
    ).all()

    layout = ac.ci.Layout2DCI(shape_2d=(3, 5), region_list=[(0, 3, 0, 1), (0, 3, 3, 4)])

    array = ac.Array2D.manual(
        array=[
            [0.0, 1.0, 1.1, 2.0, 3.0],
            [4.0, 5.0, 1.1, 6.0, 7.0],
            [8.0, 9.0, 1.1, 10.0, 11.0],
        ],
        pixel_scales=1.0,
    )

    new_array = layout.array_2d_of_serial_edges_and_epers_array(
        array=array, front_edge_columns=(0, 1), trails_columns=(0, 1)
    )

    assert (
        new_array
        == np.array(
            [
                [0.0, 1.0, 0.0, 2.0, 3.0],
                [4.0, 5.0, 0.0, 6.0, 7.0],
                [8.0, 9.0, 0.0, 10.0, 11.0],
            ]
        )
    ).all()


def test__array_2d_list_for_serial_calibration():

    layout = ac.ci.Layout2DCI(shape_2d=(3, 5), region_list=[(0, 3, 0, 5)])

    array = ac.Array2D.manual(
        array=[
            [0.0, 1.0, 2.0, 2.0, 2.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 2.0, 4.0, 4.0],
        ],
        pixel_scales=1.0,
    )

    serial_region = layout.array_2d_list_for_serial_calibration(array=array)

    assert (
        serial_region[0]
        == np.array(
            [
                [0.0, 1.0, 2.0, 2.0, 2.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 4.0, 4.0],
            ]
        )
    ).all()

    layout = ac.ci.Layout2DCI(shape_2d=(3, 5), region_list=[(0, 1, 1, 4), (2, 3, 1, 4)])

    serial_region = layout.array_2d_list_for_serial_calibration(array=array)

    assert (serial_region[0] == np.array([[0.0, 1.0, 2.0, 2.0, 2.0]])).all()
    assert (serial_region[1] == np.array([[0.0, 1.0, 2.0, 4.0, 4.0]])).all()


def test__array_2d_for_serial_calibration_from():

    layout = ac.ci.Layout2DCI(
        shape_2d=(3, 5), region_list=[(0, 3, 1, 5)], serial_prescan=(0, 3, 0, 1)
    )

    array = ac.Array2D.manual(
        array=[
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        ],
        pixel_scales=1.0,
    )

    new_array = layout.array_2d_for_serial_calibration_from(array=array, rows=(0, 3))

    assert (
        new_array.native
        == np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ]
        )
    ).all()

    assert new_array.pixel_scales == (1.0, 1.0)

    layout = ac.ci.Layout2DCI(
        shape_2d=(3, 5),
        region_list=[(0, 2, 1, 4)],
        serial_prescan=(0, 3, 0, 1),
        serial_overscan=(0, 3, 3, 4),
    )

    new_array = layout.array_2d_for_serial_calibration_from(array=array, rows=(0, 2))

    assert (
        new_array.native
        == np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0]])
    ).all()

    assert new_array.pixel_scales == (1.0, 1.0)

    array = ac.Array2D.manual(
        array=[
            [0.0, 1.0, 2.0, 2.0, 2.0],
            [0.0, 1.0, 3.0, 3.0, 3.0],
            [0.0, 1.0, 4.0, 4.0, 4.0],
        ],
        pixel_scales=1.0,
    )

    layout = ac.ci.Layout2DCI(
        shape_2d=(3, 5),
        region_list=[(0, 1, 1, 3), (2, 3, 1, 3)],
        serial_prescan=(0, 3, 0, 1),
        serial_overscan=(0, 3, 3, 4),
    )

    new_array = layout.array_2d_for_serial_calibration_from(array=array, rows=(0, 1))

    assert (
        new_array.native
        == np.array([[0.0, 1.0, 2.0, 2.0, 2.0], [0.0, 1.0, 4.0, 4.0, 4.0]])
    ).all()

    assert new_array.pixel_scales == (1.0, 1.0)

    layout = ac.ci.Layout2DCI(
        shape_2d=(5, 5),
        region_list=[(0, 2, 1, 4), (3, 5, 1, 4)],
        serial_prescan=(0, 3, 0, 1),
        serial_overscan=(0, 3, 3, 4),
    )

    array = ac.Array2D.manual(
        array=[
            [0.0, 1.0, 2.0, 2.0, 2.0],
            [0.0, 1.0, 3.0, 3.0, 3.0],
            [0.0, 1.0, 4.0, 4.0, 4.0],
            [0.0, 1.0, 5.0, 5.0, 5.0],
            [0.0, 1.0, 6.0, 6.0, 6.0],
        ],
        pixel_scales=1.0,
    )

    new_array = layout.array_2d_for_serial_calibration_from(array=array, rows=(1, 2))

    assert (
        new_array.native
        == np.array([[0.0, 1.0, 3.0, 3.0, 3.0], [0.0, 1.0, 6.0, 6.0, 6.0]])
    ).all()

    assert new_array.pixel_scales == (1.0, 1.0)


def test__maks_for_serial_calibration_from():

    layout = ac.ci.Layout2DCI(shape_2d=(5, 5), region_list=[(0, 2, 1, 4), (3, 5, 1, 4)])

    mask = ac.ci.Mask2DCI.unmasked(shape_native=(5, 5), pixel_scales=1.0)

    mask[1, 1] = True
    mask[4, 3] = True

    serial_frame = layout.mask_for_serial_calibration_from(mask=mask, rows=(1, 2))

    assert (
        serial_frame
        == np.array(
            [[False, True, False, False, False], [False, False, False, True, False]]
        )
    ).all()


def test__extracted_layout_for_serial_calibration_from():

    layout = ac.ci.Layout2DCI(
        shape_2d=(3, 5), region_list=[(0, 3, 1, 5)], serial_prescan=(0, 3, 0, 1)
    )

    extracted_layout = layout.extracted_layout_for_serial_calibration_from(
        new_shape_2d=(3, 5), rows=(0, 3)
    )

    assert extracted_layout.original_roe_corner == (1, 0)
    assert extracted_layout.region_list == [(0, 3, 1, 5)]
    assert extracted_layout.parallel_overscan == None
    assert extracted_layout.serial_prescan == (0, 3, 0, 1)
    assert extracted_layout.serial_overscan == None

    layout = ac.ci.Layout2DCI(
        shape_2d=(3, 5),
        region_list=[(0, 2, 1, 4)],
        serial_prescan=(0, 3, 0, 1),
        serial_overscan=(0, 3, 3, 4),
    )

    extracted_layout = layout.extracted_layout_for_serial_calibration_from(
        new_shape_2d=(2, 5), rows=(0, 2)
    )

    assert extracted_layout.original_roe_corner == (1, 0)
    assert extracted_layout.region_list == [(0, 2, 1, 4)]
    assert extracted_layout.parallel_overscan == None
    assert extracted_layout.serial_prescan == (0, 2, 0, 1)
    assert extracted_layout.serial_overscan == (0, 2, 3, 4)

    array = ac.Array2D.manual(
        array=[
            [0.0, 1.0, 2.0, 2.0, 2.0],
            [0.0, 1.0, 3.0, 3.0, 3.0],
            [0.0, 1.0, 4.0, 4.0, 4.0],
        ],
        pixel_scales=1.0,
    )

    layout = ac.ci.Layout2DCI(
        shape_2d=(3, 5),
        region_list=[(0, 1, 1, 3), (2, 3, 1, 3)],
        serial_prescan=(0, 3, 0, 1),
        serial_overscan=(0, 3, 3, 4),
    )

    extracted_layout = layout.extracted_layout_for_serial_calibration_from(
        new_shape_2d=(2, 5), rows=(0, 1)
    )

    assert extracted_layout.original_roe_corner == (1, 0)
    assert extracted_layout.region_list == [(0, 1, 1, 3), (1, 2, 1, 3)]
    assert extracted_layout.parallel_overscan == None
    assert extracted_layout.serial_prescan == (0, 2, 0, 1)
    assert extracted_layout.serial_overscan == (0, 2, 3, 4)

    layout = ac.ci.Layout2DCI(
        shape_2d=(5, 5),
        region_list=[(0, 2, 1, 4), (3, 5, 1, 4)],
        serial_prescan=(0, 3, 0, 1),
        serial_overscan=(0, 3, 3, 4),
    )

    array = ac.Array2D.manual(
        array=[
            [0.0, 1.0, 2.0, 2.0, 2.0],
            [0.0, 1.0, 3.0, 3.0, 3.0],
            [0.0, 1.0, 4.0, 4.0, 4.0],
            [0.0, 1.0, 5.0, 5.0, 5.0],
            [0.0, 1.0, 6.0, 6.0, 6.0],
        ],
        pixel_scales=1.0,
    )

    extracted_layout = layout.extracted_layout_for_serial_calibration_from(
        new_shape_2d=(2, 5), rows=(1, 2)
    )

    assert extracted_layout.original_roe_corner == (1, 0)
    assert extracted_layout.region_list == [(0, 1, 1, 4), (1, 2, 1, 4)]
    assert extracted_layout.parallel_overscan == None
    assert extracted_layout.serial_prescan == (0, 2, 0, 1)
    assert extracted_layout.serial_overscan == (0, 2, 3, 4)


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
