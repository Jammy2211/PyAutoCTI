import numpy as np
import autocti as ac


def test__regions_array_2d_from():

    extract = ac.Extract2DMisc(region_list=[(0, 3, 0, 3)])

    array = ac.Array2D.manual(
        array=[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        pixel_scales=1.0,
    )

    array_extracted = extract.regions_array_2d_from(array=array)

    assert (
        array_extracted
        == np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [0.0, 0.0, 0.0]]
        )
    ).all()

    extract = ac.Extract2DMisc(region_list=[(0, 1, 1, 2), (2, 3, 1, 3)])

    array = ac.Array2D.manual(
        array=[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
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

    extract = ac.Extract2DMisc(region_list=[(0, 3, 0, 3)])

    array = ac.Array2D.manual(
        array=[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        pixel_scales=1.0,
    )

    array_extracted = extract.non_regions_array_2d_from(array=array)

    assert (
        array_extracted
        == np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [9.0, 10.0, 11.0]]
        )
    ).all()

    extract = ac.Extract2DMisc(region_list=[(0, 1, 0, 3), (3, 4, 0, 3)])

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
