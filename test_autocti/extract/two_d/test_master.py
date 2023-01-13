import numpy as np
import autocti as ac


def test__regions_array_2d_from():

    extract = ac.Extract2DMaster(region_list=[(0, 3, 0, 3)])

    array = ac.Array2D.no_mask(
        values=[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        pixel_scales=1.0,
    )

    array_extracted = extract.regions_array_2d_from(array=array)

    assert (
        array_extracted
        == np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [0.0, 0.0, 0.0]]
        )
    ).all()

    extract = ac.Extract2DMaster(region_list=[(0, 1, 1, 2), (2, 3, 1, 3)])

    array = ac.Array2D.no_mask(
        values=[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        pixel_scales=1.0,
    )

    array_extracted = extract.regions_array_2d_from(array=array)

    assert (
        array_extracted
        == np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 7.0, 8.0], [0.0, 0.0, 0.0]]
        )
    ).all()


def test__non_regions_array_2d_from():

    extract = ac.Extract2DMaster(region_list=[(0, 3, 0, 3)])

    array = ac.Array2D.no_mask(
        values=[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        pixel_scales=1.0,
    )

    array_extracted = extract.non_regions_array_2d_from(array=array)

    assert (
        array_extracted
        == np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [9.0, 10.0, 11.0]]
        )
    ).all()

    extract = ac.Extract2DMaster(region_list=[(0, 1, 0, 3), (3, 4, 0, 3)])

    array = ac.Array2D.no_mask(
        values=[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0],
            [12.0, 13.0, 14.0],
        ],
        pixel_scales=1.0,
    )

    array_extracted = extract.non_regions_array_2d_from(array=array)

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


def test__parallel_fpr_and_eper_array_2d_from():

    parallel_array = ac.Array2D.no_mask(
        values=[
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

    extract = ac.Extract2DMaster(region_list=[(0, 4, 0, 3)])

    new_array = extract.parallel_fpr_and_eper_array_2d_from(
        array=parallel_array, fpr_pixels=(0, 2), eper_pixels=(0, 2)
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

    extract = ac.Extract2DMaster(region_list=[(0, 1, 0, 3), (3, 4, 0, 3)])

    new_array = extract.parallel_fpr_and_eper_array_2d_from(
        array=parallel_array, fpr_pixels=(0, 1), eper_pixels=(0, 1)
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


def test__serial_fpr_and_eper_array_2d_from():

    extract = ac.Extract2DMaster(region_list=[(0, 3, 0, 3)])

    array = ac.Array2D.no_mask(
        values=[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]],
        pixel_scales=1.0,
    )

    new_array = extract.serial_fpr_and_eper_array_2d_from(
        array=array, fpr_pixels=(0, 1)
    )

    assert (
        new_array
        == np.array([[0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0]])
    ).all()

    new_array = extract.serial_fpr_and_eper_array_2d_from(
        array=array, fpr_pixels=(0, 2)
    )

    assert (
        new_array
        == np.array([[0.0, 1.0, 0.0, 0.0], [4.0, 5.0, 0.0, 0.0], [8.0, 9.0, 0.0, 0.0]])
    ).all()

    extract = ac.Extract2DMaster(region_list=[(0, 3, 0, 2)])

    new_array = extract.serial_fpr_and_eper_array_2d_from(
        array=array, eper_pixels=(0, 1)
    )

    assert (
        new_array
        == np.array([[0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 6.0, 0.0], [0.0, 0.0, 10.0, 0.0]])
    ).all()

    new_array = extract.serial_fpr_and_eper_array_2d_from(
        array=array, eper_pixels=(0, 2)
    )

    assert (
        new_array
        == np.array(
            [[0.0, 0.0, 2.0, 3.0], [0.0, 0.0, 6.0, 7.0], [0.0, 0.0, 10.0, 11.0]]
        )
    ).all()

    extract = ac.Extract2DMaster(region_list=[(0, 3, 0, 1), (0, 3, 3, 4)])

    array = ac.Array2D.no_mask(
        values=[
            [0.0, 1.0, 1.1, 2.0, 3.0],
            [4.0, 5.0, 1.1, 6.0, 7.0],
            [8.0, 9.0, 1.1, 10.0, 11.0],
        ],
        pixel_scales=1.0,
    )

    new_array = extract.serial_fpr_and_eper_array_2d_from(
        array=array, fpr_pixels=(0, 1), eper_pixels=(0, 1)
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


def test__serial_overscan_above_eper_array_2d_from():
    extract = ac.Extract2DMaster(
        region_list=[(1, 2, 1, 3), (3, 4, 1, 3)],
        serial_prescan=(0, 5, 0, 1),
        serial_overscan=(0, 5, 3, 4),
    )

    array = ac.Array2D.no_mask(
        values=[
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
            [12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0],
        ],
        pixel_scales=1.0,
    )

    new_array = extract.serial_overscan_above_eper_array_2d_from(array=array)

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

    extract = ac.Extract2DMaster(
        region_list=[(0, 1, 0, 2), (2, 3, 0, 2)], serial_overscan=(0, 4, 2, 4)
    )

    array = ac.Array2D.no_mask(
        values=[
            [0.0, 1.0, 2.0, 0.5],
            [3.0, 4.0, 5.0, 0.5],
            [6.0, 7.0, 8.0, 0.5],
            [9.0, 10.0, 11.0, 0.5],
        ],
        pixel_scales=1.0,
    )

    new_array = extract.serial_overscan_above_eper_array_2d_from(array=array)

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
