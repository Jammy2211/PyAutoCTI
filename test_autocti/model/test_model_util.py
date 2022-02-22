import pytest

import autocti as ac


def test__trap_list():

    parallel_trap_list = [
        ac.TrapInstantCapture(density=1.0),
        ac.TrapInstantCapture(density=2.0),
    ]

    cti = ac.CTI2D(parallel_traps=parallel_trap_list)

    assert cti.trap_list[0].density == 1.0
    assert cti.trap_list[1].density == 2.0

    serial_trap_list = [
        ac.TrapInstantCapture(density=3.0),
        ac.TrapInstantCapture(density=4.0),
    ]

    cti = ac.CTI2D(serial_traps=serial_trap_list)

    assert cti.trap_list[0].density == 3.0
    assert cti.trap_list[1].density == 4.0

    cti = ac.CTI2D(parallel_traps=parallel_trap_list, serial_traps=serial_trap_list)

    assert cti.trap_list[0].density == 1.0
    assert cti.trap_list[1].density == 2.0
    assert cti.trap_list[2].density == 3.0
    assert cti.trap_list[3].density == 4.0


def test__delta_ellipticity():

    parallel_trap_list = [
        ac.TrapInstantCapture(density=1.0, release_timescale=2.0),
        ac.TrapInstantCapture(density=2.0, release_timescale=4.0),
    ]

    cti = ac.CTI2D(parallel_traps=parallel_trap_list)

    assert cti.delta_ellipticity == pytest.approx(4.0 * 0.57029, 1.0e-4)

    serial_trap_list = [
        ac.TrapInstantCapture(density=5.0, release_timescale=6.0),
        ac.TrapInstantCapture(density=7.0, release_timescale=8.0),
    ]

    cti = ac.CTI2D(serial_traps=serial_trap_list)

    assert cti.delta_ellipticity == pytest.approx(4.0 * 1.875875, 1.0e-4)

    cti = ac.CTI2D(parallel_traps=parallel_trap_list, serial_traps=serial_trap_list)

    assert cti.delta_ellipticity == pytest.approx(4.0 * 2.446169, 1.0e-4)
