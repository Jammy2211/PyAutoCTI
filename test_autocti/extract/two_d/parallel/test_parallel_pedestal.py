import numpy as np

import autocti as ac


def test__region_list_from__via_array_2d_list_from(
    parallel_array, parallel_masked_array
):

    extract = ac.Extract2DParallelPedestal(
        shape_2d=parallel_array.shape_native,
        parallel_overscan=(8, 10, 0, 2),
        serial_overscan=(0, 8, 2, 3),
    )

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(0, 1))
    )
    assert (array_2d_list[0] == np.array([[8.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(0, 2))
    )
    print(array_2d_list)
    assert (array_2d_list[0] == np.array([[8.0], [9.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(-1, 1))
    )
    assert (array_2d_list[0] == np.array([[7.0], [8.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_masked_array, settings=ac.SettingsExtract(pixels=(-5, -3))
    )

    assert (array_2d_list[0].mask == np.array([[True], [False]])).all()


def test__region_list_from__via_array_2d_list_from__pixels_from_end(
    parallel_array, parallel_masked_array
):

    extract = ac.Extract2DParallelPedestal(
        shape_2d=parallel_array.shape_native,
        parallel_overscan=(8, 10, 0, 2),
        serial_overscan=(0, 8, 2, 3),
    )

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels_from_end=1)
    )
    assert (array_2d_list[0] == np.array([[9.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels_from_end=2)
    )
    assert (array_2d_list[0] == np.array([[8.0], [9.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_masked_array, settings=ac.SettingsExtract(pixels_from_end=2)
    )

    assert (array_2d_list[0].mask == np.array([[False], [False]])).all()


def test__array_2d_from():

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

    extract = ac.Extract2DParallelPedestal(
        shape_2d=array.shape_native,
        parallel_overscan=(3, 5, 0, 1),
        serial_overscan=(0, 3, 1, 3),
    )

    array_2d = extract.array_2d_from(array=array)

    print(array_2d.native)

    assert (
        array_2d.native
        == np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 10.0, 11.0],
                [0.0, 13.0, 14.0],
            ]
        )
    ).all()
