import numpy as np

import autocti as ac


def test__region_ci_from():

    region = ac.util.ci.region_ci_from(
        region_dimensions=(3, 3), column_norm_list=[90.0, 100.0, 110.0]
    )

    assert (
        region
        == np.array([[90.0, 100.0, 110.0], [90.0, 100.0, 110.0], [90.0, 100.0, 110.0]])
    ).all()

    region = ac.util.ci.region_ci_from(
        region_dimensions=(10, 10),
        column_norm_list=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0],
    )

    assert (region > 0).all()


def test__region_list_ci_via_electronics_from():

    region_list_ci = ac.util.ci.region_list_ci_via_electronics_from(
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

    region_list_ci = ac.util.ci.region_list_ci_via_electronics_from(
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

    region_list_ci = ac.util.ci.region_list_ci_via_electronics_from(
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

    region_list_ci = ac.util.ci.region_list_ci_via_electronics_from(
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
