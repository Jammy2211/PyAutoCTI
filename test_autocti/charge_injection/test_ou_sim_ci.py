import numpy as np

import autocti as ac

from autocti.charge_injection import ou_sim_ci


def test__non_uniform_array_is_correct_with_rotation():

    # bottom left

    array = ou_sim_ci.charge_injection_array_from(
        ccd_id="123",
        quadrant_id="E",
        injection_start=0,
        injection_end=2000,
        injection_on=200,
        injection_off=200,
        injection_norm=50000.0,
    )

    assert array.shape_native == (2086, 2128)
    assert array.native[0, 50] == 0
    assert array.native[0, 2099] == 0
    assert (array.native[0:200, 51:2099] > 0).all()
    assert 49000.0 < np.mean(array.native[0:200, 51:2099]) < 51000.0

    # top left

    array = ou_sim_ci.charge_injection_array_from(
        ccd_id="123",
        quadrant_id="H",
        injection_start=0,
        injection_end=2000,
        injection_on=200,
        injection_off=200,
        injection_norm=50000.0,
    )

    assert array.native[1938, 50] == 0
    assert array.native[1938, 2099] == 0
    assert (array.native[1928:2128, 51:2099] > 0).all()
    assert 49000.0 < np.mean(array.native[1928:2128, 51:2099]) < 51000.0

    # bottom right

    array = ou_sim_ci.charge_injection_array_from(
        ccd_id="123",
        quadrant_id="F",
        injection_start=0,
        injection_end=2000,
        injection_on=200,
        injection_off=200,
        injection_norm=50000.0,
    )

    assert array.shape_native == (2086, 2128)
    assert array.native[0, 28] == 0
    assert array.native[0, 2077] == 0
    assert (array.native[0:200, 29:2077] > 0).all()
    assert 49000.0 < np.mean(array.native[0:200, 51:2099]) < 51000.0

    # top right

    array = ou_sim_ci.charge_injection_array_from(
        ccd_id="123",
        quadrant_id="G",
        injection_start=0,
        injection_end=2000,
        injection_on=200,
        injection_off=200,
        injection_norm=50000.0,
    )

    assert array.shape_native == (2086, 2128)
    assert array.native[1938, 28] == 0
    assert array.native[1938, 2077] == 0
    assert (array.native[1928:2128, 29:2077] > 0).all()
    assert 49000.0 < np.mean(array.native[1928:2128, 29:2077]) < 51000.0


def test__add_cti_to_pre_cti_data():

    clocker = ac.Clocker2D(parallel_express=2, serial_express=2)

    parallel_trap_list = [ac.TrapInstantCapture(density=0.13, release_timescale=1.25)]
    parallel_ccd = ac.CCDPhase(
        well_fill_power=0.8, well_notch_depth=0.0, full_well_depth=84700.0
    )
    serial_trap_list = [ac.TrapInstantCapture(density=0.0442, release_timescale=0.8)]
    serial_ccd = ac.CCDPhase(
        well_fill_power=0.8, well_notch_depth=0.0, full_well_depth=84700.0
    )

    # bottom left

    array = ou_sim_ci.charge_injection_array_from(
        ccd_id="123",
        quadrant_id="E",
        injection_start=0,
        injection_end=2000,
        injection_on=200,
        injection_off=200,
        injection_norm=50000.0,
    )

    assert array.native[199, 100] > 0.0
    assert array.native[200, 100] == 0.0

    pre_cti_data = array.native[:, 100:101]
    pre_cti_data.mask = pre_cti_data.mask[:, 100:101]

    post_cti_data = ou_sim_ci.add_cti_to_pre_cti_data(
        pre_cti_data=pre_cti_data,
        ccd_id="123",
        quadrant_id="E",
        clocker=clocker,
        parallel_trap_list=parallel_trap_list,
        parallel_ccd=parallel_ccd,
        serial_trap_list=serial_trap_list,
        serial_ccd=serial_ccd,
    )

    assert post_cti_data[200, 0] > 0.0

    # top left

    array = ou_sim_ci.charge_injection_array_from(
        ccd_id="123",
        quadrant_id="H",
        injection_start=0,
        injection_end=2000,
        injection_on=200,
        injection_off=200,
        injection_norm=50000.0,
    )

    assert array.native[1886, 100] > 0.0
    assert array.native[1885, 100] == 0.0

    pre_cti_data = array.native[:, 100:101]
    pre_cti_data.mask = pre_cti_data.mask[:, 100:101]

    post_cti_data = ou_sim_ci.add_cti_to_pre_cti_data(
        pre_cti_data=pre_cti_data,
        ccd_id="123",
        quadrant_id="H",
        clocker=clocker,
        parallel_trap_list=parallel_trap_list,
        parallel_ccd=parallel_ccd,
        serial_trap_list=serial_trap_list,
        serial_ccd=serial_ccd,
    )

    assert post_cti_data[1885, 0] > 0.0

    # bottom right

    array = ou_sim_ci.charge_injection_array_from(
        ccd_id="123",
        quadrant_id="F",
        injection_start=0,
        injection_end=2000,
        injection_on=200,
        injection_off=200,
        injection_norm=50000.0,
    )

    assert array.native[199, 100] > 0.0
    assert array.native[200, 100] == 0.0

    pre_cti_data = array.native[:, 100:101]
    pre_cti_data.mask = pre_cti_data.mask[:, 100:101]

    post_cti_data = ou_sim_ci.add_cti_to_pre_cti_data(
        pre_cti_data=pre_cti_data,
        ccd_id="123",
        quadrant_id="F",
        clocker=clocker,
        parallel_trap_list=parallel_trap_list,
        parallel_ccd=parallel_ccd,
        serial_trap_list=serial_trap_list,
        serial_ccd=serial_ccd,
    )

    assert post_cti_data[200, 0] > 0.0

    # top right

    array = ou_sim_ci.charge_injection_array_from(
        ccd_id="123",
        quadrant_id="G",
        injection_start=0,
        injection_end=2000,
        injection_on=200,
        injection_off=200,
        injection_norm=50000.0,
    )

    assert array.native[1886, 100] > 0.0
    assert array.native[1885, 100] == 0.0

    pre_cti_data = array.native[:, 100:101]
    pre_cti_data.mask = pre_cti_data.mask[:, 100:101]

    post_cti_data = ou_sim_ci.add_cti_to_pre_cti_data(
        pre_cti_data=pre_cti_data,
        ccd_id="123",
        quadrant_id="G",
        clocker=clocker,
        parallel_trap_list=parallel_trap_list,
        parallel_ccd=parallel_ccd,
        serial_trap_list=serial_trap_list,
        serial_ccd=serial_ccd,
    )

    assert post_cti_data[1885, 0] > 0.0


def test__tvac_values():

    array = ou_sim_ci.charge_injection_array_from(
        #   iquad=0,
        ccd_id="123",
        quadrant_id="E",
        injection_start=16,
        injection_end=2086,
        injection_on=420,
        injection_off=100,
        injection_norm=50000.0,
    )

    tvac_region_1 = ac.Region2D(region=[16, 436, 51, 2099])

    assert (array[tvac_region_1.slice] > 10000).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x0 - 1] == 0).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y0 - 1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()

    tvac_region_1 = ac.Region2D(region=[536, 956, 51, 2099])

    assert (array[tvac_region_1.slice] > 10000).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x0 - 1] == 0).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y0 - 1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()

    array = ou_sim_ci.charge_injection_array_from(
        #   iquad=1,
        ccd_id="123",
        quadrant_id="F",
        injection_start=16,
        injection_end=2086,
        injection_on=420,
        injection_off=100,
        injection_norm=50000.0,
    )

    tvac_region_1 = ac.Region2D(region=[16, 436, 29, 2077])

    assert (array[tvac_region_1.slice] > 10000).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x0 - 1] == 0).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y0 - 1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()

    tvac_region_1 = ac.Region2D(region=[536, 956, 29, 2077])

    assert (array[tvac_region_1.slice] > 10000).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x0 - 1] == 0).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y0 - 1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()

    array = ou_sim_ci.charge_injection_array_from(
        #  iquad=2,
        ccd_id="123",
        quadrant_id="H",
        injection_start=16,
        injection_end=2086,
        injection_on=420,
        injection_off=100,
        injection_norm=50000.0,
    )

    tvac_region_1 = ac.Region2D(region=[1650, 2070, 51, 2099])

    assert (array[tvac_region_1.slice] > 10000).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x0 - 1] == 0).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y0 - 1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()

    tvac_region_1 = ac.Region2D(region=[1130, 1550, 51, 2099])

    assert (array[tvac_region_1.slice] > 10000).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x0 - 1] == 0).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y0 - 1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()

    array = ou_sim_ci.charge_injection_array_from(
        #   iquad=3,
        ccd_id="123",
        quadrant_id="G",
        injection_start=16,
        injection_end=2086,
        injection_on=420,
        injection_off=100,
        injection_norm=50000.0,
    )

    tvac_region_1 = ac.Region2D(region=[1650, 2070, 29, 2077])

    assert (array[tvac_region_1.slice] > 10000).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x0 - 1] == 0).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y0 - 1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()

    tvac_region_1 = ac.Region2D(region=[1130, 1550, 29, 2077])

    assert (array[tvac_region_1.slice] > 10000).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x0 - 1] == 0).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y0 - 1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()

    array = ou_sim_ci.charge_injection_array_from(
        ccd_id="456",
        quadrant_id="E",
        injection_start=16,
        injection_end=2086,
        injection_on=420,
        injection_off=100,
        injection_norm=50000.0,
    )

    tvac_region_1 = ac.Region2D(region=[1650, 2070, 29, 2077])

    assert (array[tvac_region_1.slice] > 10000).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x0 - 1] == 0).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y0 - 1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()

    tvac_region_1 = ac.Region2D(region=[1130, 1550, 29, 2077])

    assert (array[tvac_region_1.slice] > 10000).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x0 - 1] == 0).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y0 - 1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()

    array = ou_sim_ci.charge_injection_array_from(
        ccd_id="456",
        quadrant_id="F",
        injection_start=16,
        injection_end=2086,
        injection_on=420,
        injection_off=100,
        injection_norm=50000.0,
    )

    tvac_region_1 = ac.Region2D(region=[1650, 2070, 51, 2099])

    assert (array[tvac_region_1.slice] > 10000).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x0 - 1] == 0).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y0 - 1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()

    tvac_region_1 = ac.Region2D(region=[1130, 1550, 51, 2099])

    assert (array[tvac_region_1.slice] > 10000).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x0 - 1] == 0).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y0 - 1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()

    array = ou_sim_ci.charge_injection_array_from(
        ccd_id="456",
        quadrant_id="G",
        injection_start=16,
        injection_end=2086,
        injection_on=420,
        injection_off=100,
        injection_norm=50000.0,
    )

    tvac_region_1 = ac.Region2D(region=[16, 436, 51, 2099])

    assert (array[tvac_region_1.slice] > 10000).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x0 - 1] == 0).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y0 - 1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()

    tvac_region_1 = ac.Region2D(region=[536, 956, 51, 2099])

    assert (array[tvac_region_1.slice] > 10000).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x0 - 1] == 0).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y0 - 1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()

    array = ou_sim_ci.charge_injection_array_from(
        ccd_id="456",
        quadrant_id="H",
        injection_start=16,
        injection_end=2086,
        injection_on=420,
        injection_off=100,
        injection_norm=50000.0,
    )

    tvac_region_1 = ac.Region2D(region=[16, 436, 29, 2077])

    assert (array[tvac_region_1.slice] > 10000).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x0 - 1] == 0).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y0 - 1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()

    tvac_region_1 = ac.Region2D(region=[536, 956, 29, 2077])

    assert (array[tvac_region_1.slice] > 10000).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x0 - 1] == 0).all()
    assert (array[tvac_region_1.y0 : tvac_region_1.y1, tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y0 - 1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()
    assert (array[tvac_region_1.y1, tvac_region_1.x0 : tvac_region_1.x1] == 0).all()
