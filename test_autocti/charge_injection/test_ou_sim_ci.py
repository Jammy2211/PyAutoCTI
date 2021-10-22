import numpy as np
from autocti.charge_injection import ou_sim_ci

from autocti.util.clocker import Clocker2D
from arcticpy.src import ccd
from arcticpy.src import traps


def test__non_uniform_array_is_correct_with_rotation():

    # bottom left

    array = ou_sim_ci.charge_injection_array_from(
        iquad=0,
        injection_total=5,
        injection_on=200,
        injection_off=200,
        injection_normalization=50000.0,
    )

    assert array.shape == (2086, 2128)
    assert array[0, 50] == 0
    assert array[0, 2099] == 0
    assert (array[0:200, 51:2099] > 0).all()
    assert 49000.0 < np.mean(array[0:200, 51:2099]) < 51000.0

    # top left

    array = ou_sim_ci.charge_injection_array_from(
        iquad=2,
        injection_total=5,
        injection_on=200,
        injection_off=200,
        injection_normalization=50000.0,
    )

    assert array[1938, 50] == 0
    assert array[1938, 2099] == 0
    assert (array[1928:2128, 51:2099] > 0).all()
    assert 49000.0 < np.mean(array[1928:2128, 51:2099]) < 51000.0

    # bottom right

    array = ou_sim_ci.charge_injection_array_from(
        iquad=1,
        injection_total=5,
        injection_on=200,
        injection_off=200,
        injection_normalization=50000.0,
    )

    assert array.shape == (2086, 2128)
    assert array[0, 28] == 0
    assert array[0, 2077] == 0
    assert (array[0:200, 29:2077] > 0).all()
    assert 49000.0 < np.mean(array[0:200, 51:2099]) < 51000.0

    # top right

    array = ou_sim_ci.charge_injection_array_from(
        iquad=3,
        injection_total=5,
        injection_on=200,
        injection_off=200,
        injection_normalization=50000.0,
    )

    assert array.shape == (2086, 2128)
    assert array[1938, 28] == 0
    assert array[1938, 2077] == 0
    assert (array[1928:2128, 29:2077] > 0).all()
    assert 49000.0 < np.mean(array[1928:2128, 29:2077]) < 51000.0


def test__add_cti_to_pre_cti_data():

    clocker = Clocker2D(
        parallel_express=2, parallel_charge_injection_mode=True, serial_express=2
    )

    parallel_trap_list = [
        traps.TrapInstantCapture(density=0.13, release_timescale=1.25)
    ]
    parallel_ccd = ccd.CCDPhase(
        well_fill_power=0.8, well_notch_depth=0.0, full_well_depth=84700.0
    )
    serial_trap_list = [traps.TrapInstantCapture(density=0.0442, release_timescale=0.8)]
    serial_ccd = ccd.CCDPhase(
        well_fill_power=0.8, well_notch_depth=0.0, full_well_depth=84700.0
    )

    # bottom left

    array = ou_sim_ci.charge_injection_array_from(
        iquad=0,
        injection_total=5,
        injection_on=200,
        injection_off=200,
        injection_normalization=50000.0,
    )

    assert array[199, 100] > 0.0
    assert array[200, 100] == 0.0

    pre_cti_data = array[:, 100:101]
    pre_cti_data.mask = pre_cti_data.mask[:, 100:101]

    post_cti_data = ou_sim_ci.add_cti_to_pre_cti_data(
        pre_cti_data=pre_cti_data,
        iquad=0,
        clocker=clocker,
        parallel_trap_list=parallel_trap_list,
        parallel_ccd=parallel_ccd,
        serial_trap_list=serial_trap_list,
        serial_ccd=serial_ccd,
    )

    assert post_cti_data[200, 0] > 0.0

    # top left

    array = ou_sim_ci.charge_injection_array_from(
        iquad=2,
        injection_total=5,
        injection_on=200,
        injection_off=200,
        injection_normalization=50000.0,
    )

    assert array[1886, 100] > 0.0
    assert array[1885, 100] == 0.0

    pre_cti_data = array[:, 100:101]
    pre_cti_data.mask = pre_cti_data.mask[:, 100:101]

    post_cti_data = ou_sim_ci.add_cti_to_pre_cti_data(
        pre_cti_data=pre_cti_data,
        iquad=2,
        clocker=clocker,
        parallel_trap_list=parallel_trap_list,
        parallel_ccd=parallel_ccd,
        serial_trap_list=serial_trap_list,
        serial_ccd=serial_ccd,
    )

    assert post_cti_data[1885, 0] > 0.0

    # bottom right

    array = ou_sim_ci.charge_injection_array_from(
        iquad=1,
        injection_total=5,
        injection_on=200,
        injection_off=200,
        injection_normalization=50000.0,
    )

    assert array[199, 100] > 0.0
    assert array[200, 100] == 0.0

    pre_cti_data = array[:, 100:101]
    pre_cti_data.mask = pre_cti_data.mask[:, 100:101]

    post_cti_data = ou_sim_ci.add_cti_to_pre_cti_data(
        pre_cti_data=pre_cti_data,
        iquad=1,
        clocker=clocker,
        parallel_trap_list=parallel_trap_list,
        parallel_ccd=parallel_ccd,
        serial_trap_list=serial_trap_list,
        serial_ccd=serial_ccd,
    )

    assert post_cti_data[200, 0] > 0.0

    # top right

    array = ou_sim_ci.charge_injection_array_from(
        iquad=3,
        injection_total=5,
        injection_on=200,
        injection_off=200,
        injection_normalization=50000.0,
    )

    assert array[1886, 100] > 0.0
    assert array[1885, 100] == 0.0

    pre_cti_data = array[:, 100:101]
    pre_cti_data.mask = pre_cti_data.mask[:, 100:101]

    post_cti_data = ou_sim_ci.add_cti_to_pre_cti_data(
        pre_cti_data=pre_cti_data,
        iquad=3,
        clocker=clocker,
        parallel_trap_list=parallel_trap_list,
        parallel_ccd=parallel_ccd,
        serial_trap_list=serial_trap_list,
        serial_ccd=serial_ccd,
    )

    assert post_cti_data[1885, 0] > 0.0
