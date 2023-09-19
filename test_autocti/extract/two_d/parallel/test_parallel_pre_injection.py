import numpy as np

import autocti as ac


def test__region_list_from__via_array_2d_list_from(
    parallel_array, parallel_masked_array
):
    extract = ac.Extract2DParallelPreInjection(region_list=[(2, 5, 0, 3)])

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(0, 1))
    )
    assert (array_2d_list[0] == np.array([[0.0, 0.0, 0.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(1, 2))
    )
    assert (array_2d_list[0] == np.array([[1.0, 1.0, 1.0]])).all()

    extract = ac.Extract2DParallelPreInjection(region_list=[(2, 4, 0, 3), (5, 8, 0, 3)])

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels=(0, 1))
    )

    assert (array_2d_list[0] == np.array([[0.0, 0.0, 0.0]])).all()
    assert len(array_2d_list) == 1

    extract = ac.Extract2DParallelPreInjection(region_list=[(3, 4, 0, 3), (5, 8, 0, 3)])

    array_2d_list = extract.array_2d_list_from(
        array=parallel_masked_array, settings=ac.SettingsExtract(pixels=(0, 3))
    )

    assert (
        array_2d_list[0].mask
        == np.array([[False, False, False], [False, False, False], [False, True, False]])
    ).all()


def test__region_list_from__via_array_2d_list_from__pixels_from_end(
    parallel_array, parallel_masked_array
):
    extract = ac.Extract2DParallelPreInjection(region_list=[(2, 4, 0, 3)], shape_2d=(5, 5))

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels_from_end=1)
    )

    assert (array_2d_list[0] == np.array([[1.0, 1.0, 1.0]])).all()

    array_2d_list = extract.array_2d_list_from(
        array=parallel_array, settings=ac.SettingsExtract(pixels_from_end=2)
    )
    assert (array_2d_list[0] == np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])).all()

    extract = ac.Extract2DParallelPreInjection(region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

    array_2d_list = extract.array_2d_list_from(
        array=parallel_masked_array, settings=ac.SettingsExtract(pixels_from_end=1)
    )

    assert (
        array_2d_list[0].mask
        == np.array([[False, False, False]])
    ).all()
