import numpy as np
import pytest
import autocti as ac


def test__pre_cti_data_uniform_from():

    layout = ac.Layout2DCI(shape_2d=(4, 3), region_list=[(0, 1, 0, 2), (2, 3, 0, 2)])

    pre_cti_data = layout.pre_cti_data_uniform_from(norm=30.0, pixel_scales=1.0)

    assert (
        pre_cti_data.native
        == np.array(
            [[30.0, 30.0, 0.0], [0.0, 0.0, 0.0], [30.0, 30.0, 0.0], [0.0, 0.0, 0.0]]
        )
    ).all()


def test__pre_cti_data_non_uniform_from():

    layout = ac.Layout2DCI(shape_2d=(5, 5), region_list=[(0, 3, 0, 3)])

    image = layout.pre_cti_data_non_uniform_from(
        column_norm_list=[100.0, 90.0, 80.0], pixel_scales=1.0, row_slope=0.0
    )

    assert (
        image.native
        == np.array(
            [
                [100.0, 90.0, 80.0, 0.0, 0.0],
                [100.0, 90.0, 80.0, 0.0, 0.0],
                [100.0, 90.0, 80.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    ).all()

    layout = ac.Layout2DCI(shape_2d=(5, 5), region_list=[(0, 2, 1, 3), (3, 5, 1, 3)])

    image = layout.pre_cti_data_non_uniform_from(
        column_norm_list=[10.0, 20.0], pixel_scales=1.0, row_slope=0.0
    )

    assert (
        image.native
        == np.array(
            [
                [0.0, 10.0, 20.0, 0.0, 0.0],
                [0.0, 10.0, 20.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 10.0, 20.0, 0.0, 0.0],
                [0.0, 10.0, 20.0, 0.0, 0.0],
            ]
        )
    ).all()

    image = layout.pre_cti_data_non_uniform_from(
        column_norm_list=[10.0, 20.0], pixel_scales=1.0, row_slope=0.01
    )

    assert image.native == pytest.approx(
        np.array(
            [
                [0.0, 10.0, 20.0, 0.0, 0.0],
                [0.0, 10.0695, 20.13911, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 10.0, 20.0, 0.0, 0.0],
                [0.0, 10.0695, 20.13911, 0.0, 0.0],
            ]
        ),
        1.0e-2,
    )


def test__pre_cti_data_from__compare_uniform_to_non_uniform():

    layout = ac.Layout2DCI(shape_2d=(5, 5), region_list=[(2, 4, 0, 2)])

    pre_cti_data_0 = layout.pre_cti_data_uniform_from(norm=30.0, pixel_scales=1.0)

    pre_cti_data_1 = layout.pre_cti_data_non_uniform_from(
        column_norm_list=[30.0, 30.0], pixel_scales=1.0, row_slope=0.0
    )

    assert (pre_cti_data_0 == pre_cti_data_1).all()
