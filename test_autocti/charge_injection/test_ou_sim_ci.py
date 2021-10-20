import numpy as np
from autocti.charge_injection import ou_sim_ci


def test__non_uniform_array_is_correct_with_rotation():

    # bottom left

    array = ou_sim_ci.charge_injection_array_from(iquad=0, ci_normalization=50000.0)

    assert array.shape == (2086, 2128)
    assert array[0, 50] == 0
    assert array[0, 2099] == 0
    assert (array[0:200, 51:2099] > 0).all()
    assert 49000.0 < np.mean(array[0:200, 51:2099]) < 51000.0

    # top left

    array = ou_sim_ci.charge_injection_array_from(iquad=2, ci_normalization=50000.0)

    assert array[1938, 50] == 0
    assert array[1938, 2099] == 0
    assert (array[1928:2128, 51:2099] > 0).all()
    assert 49000.0 < np.mean(array[1928:2128, 51:2099]) < 51000.0

    # bottom right

    array = ou_sim_ci.charge_injection_array_from(iquad=1, ci_normalization=50000.0)

    assert array.shape == (2086, 2128)
    assert array[0, 28] == 0
    assert array[0, 2077] == 0
    assert (array[0:200, 29:2077] > 0).all()
    assert 49000.0 < np.mean(array[0:200, 51:2099]) < 51000.0

    # top right

    array = ou_sim_ci.charge_injection_array_from(iquad=3, ci_normalization=50000.0)

    assert array.shape == (2086, 2128)
    assert array[1938, 28] == 0
    assert array[1938, 2077] == 0
    assert (array[1928:2128, 29:2077] > 0).all()
    assert 49000.0 < np.mean(array[1928:2128, 29:2077]) < 51000.0


def test__add_cti_to_pre_cti_data():

    # bottom left

    array = ou_sim_ci.charge_injection_array_from(iquad=0, ci_normalization=50000.0)

    assert array[199, 100] > 0.0
    assert array[200, 100] == 0.0

    pre_cti_data = array[:, 100:101]

    post_cti_data = ou_sim_ci.add_cti_to_pre_cti_data(
        pre_cti_data=pre_cti_data, iquad=0
    )

    assert post_cti_data[200, 0] > 0.0

    # top left

    array = ou_sim_ci.charge_injection_array_from(iquad=2, ci_normalization=50000.0)

    assert array[1886, 100] > 0.0
    assert array[1885, 100] == 0.0

    pre_cti_data = array[:, 100:101]

    post_cti_data = ou_sim_ci.add_cti_to_pre_cti_data(
        pre_cti_data=pre_cti_data, iquad=2
    )

    assert post_cti_data[1885, 0] > 0.0

    # bottom right

    array = ou_sim_ci.charge_injection_array_from(iquad=1, ci_normalization=50000.0)

    assert array[199, 100] > 0.0
    assert array[200, 100] == 0.0

    pre_cti_data = array[:, 100:101]

    post_cti_data = ou_sim_ci.add_cti_to_pre_cti_data(
        pre_cti_data=pre_cti_data, iquad=1
    )

    assert post_cti_data[200, 0] > 0.0

    # top right

    array = ou_sim_ci.charge_injection_array_from(iquad=3, ci_normalization=50000.0)

    assert array[1886, 100] > 0.0
    assert array[1885, 100] == 0.0

    pre_cti_data = array[:, 100:101]

    post_cti_data = ou_sim_ci.add_cti_to_pre_cti_data(
        pre_cti_data=pre_cti_data, iquad=3
    )

    assert post_cti_data[1885, 0] > 0.0
