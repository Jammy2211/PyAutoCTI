from os import path

import autocti as ac


def test__acs_layout_for_left_and_right_quandrants__loads_data_and_dimensions(
    acs_quadrant,
):

    layout = ac.acs.Layout2DACS.from_sizes(
        roe_corner=(1, 0), serial_prescan_size=24, parallel_overscan_size=20
    )

    assert layout.original_roe_corner == (1, 0)
    assert layout.shape_2d == (2068, 2072)
    assert layout.parallel_overscan == (2048, 2068, 24, 2072)
    assert layout.serial_prescan == (0, 2068, 0, 24)
