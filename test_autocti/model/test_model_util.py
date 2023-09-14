import pytest
from os import path

from autoconf.dictable import to_dict, output_to_json, from_json

import autocti as ac


def test__trap_list():
    parallel_trap_list = [
        ac.TrapInstantCapture(density=1.0),
        ac.TrapInstantCapture(density=2.0),
    ]

    cti = ac.CTI2D(parallel_trap_list=parallel_trap_list)

    assert cti.trap_list[0].density == 1.0
    assert cti.trap_list[1].density == 2.0

    serial_trap_list = [
        ac.TrapInstantCapture(density=3.0),
        ac.TrapInstantCapture(density=4.0),
    ]

    cti = ac.CTI2D(serial_trap_list=serial_trap_list)

    assert cti.trap_list[0].density == 3.0
    assert cti.trap_list[1].density == 4.0

    cti = ac.CTI2D(
        parallel_trap_list=parallel_trap_list, serial_trap_list=serial_trap_list
    )

    assert cti.trap_list[0].density == 1.0
    assert cti.trap_list[1].density == 2.0
    assert cti.trap_list[2].density == 3.0
    assert cti.trap_list[3].density == 4.0


def test__delta_ellipticity():
    trap_list = [
        ac.TrapInstantCapture(density=1.0, release_timescale=2.0),
        ac.TrapInstantCapture(density=2.0, release_timescale=4.0),
    ]

    cti = ac.CTI1D(trap_list=trap_list)

    assert cti.delta_ellipticity == pytest.approx(4.0 * 0.57029, 1.0e-4)

    parallel_trap_list = [
        ac.TrapInstantCapture(density=1.0, release_timescale=2.0),
        ac.TrapInstantCapture(density=2.0, release_timescale=4.0),
    ]

    cti = ac.CTI2D(parallel_trap_list=parallel_trap_list)

    assert cti.delta_ellipticity == pytest.approx(4.0 * 0.57029, 1.0e-4)

    serial_trap_list = [
        ac.TrapInstantCapture(density=5.0, release_timescale=6.0),
        ac.TrapInstantCapture(density=7.0, release_timescale=8.0),
    ]

    cti = ac.CTI2D(serial_trap_list=serial_trap_list)

    assert cti.delta_ellipticity == pytest.approx(4.0 * 1.875875, 1.0e-4)

    cti = ac.CTI2D(
        parallel_trap_list=parallel_trap_list, serial_trap_list=serial_trap_list
    )

    assert cti.delta_ellipticity == pytest.approx(4.0 * 2.446169, 1.0e-4)


def test__dictable():
    json_file = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "tracer.json"
    )

    parallel_trap_list = [
        ac.TrapInstantCapture(density=1.0),
        ac.TrapInstantCapture(density=2.0),
    ]
    parallel_ccd = ac.CCDPhase(
        full_well_depth=1.0, well_notch_depth=2.0, well_fill_power=3.0
    )
    serial_trap_list = [
        ac.TrapInstantCapture(density=5.0, release_timescale=6.0),
        ac.TrapInstantCapture(density=7.0, release_timescale=8.0),
    ]
    serial_ccd = ac.CCDPhase(
        full_well_depth=4.0, well_notch_depth=5.0, well_fill_power=6.0
    )

    cti = ac.CTI2D(
        parallel_trap_list=parallel_trap_list,
        parallel_ccd=parallel_ccd,
        serial_trap_list=serial_trap_list,
        serial_ccd=serial_ccd,
    )

    output_to_json(obj=cti, file_path=json_file)

    cti_from_json = ac.CTI2D.from_json(file_path=json_file)

    assert cti_from_json.parallel_trap_list[0].density == 1.0
    assert cti_from_json.parallel_trap_list[1].density == 2.0
    assert cti_from_json.parallel_ccd.full_well_depth == 1.0
    assert cti_from_json.parallel_ccd.well_notch_depth == 2.0
    assert cti_from_json.parallel_ccd.well_fill_power == 3.0

    assert cti_from_json.serial_trap_list[0].density == 5.0
    assert cti_from_json.serial_trap_list[0].release_timescale == 6.0
    assert cti_from_json.serial_trap_list[1].density == 7.0
    assert cti_from_json.serial_trap_list[1].release_timescale == 8.0
    assert cti_from_json.serial_ccd.full_well_depth == 4.0
    assert cti_from_json.serial_ccd.well_notch_depth == 5.0
    assert cti_from_json.serial_ccd.well_fill_power == 6.0
