import autofit as af
import autocti as ac


def test__cti_model__creates_correct_model_from_inputs():
    model = af.Model(
        ac.CTI2D,
        parallel_trap_list=[ac.TrapInstantCapture],
        parallel_ccd=ac.CCDPhase,
        serial_trap_list=[ac.TrapInstantCapture],
        serial_ccd=ac.CCDPhase,
    )

    model.parallel_trap_list[0].fractional_volume_none_exposed = 0.0
    model.parallel_trap_list[0].fractional_volume_full_exposed = 0.0
    model.serial_trap_list[0].fractional_volume_none_exposed = 0.0
    model.serial_trap_list[0].fractional_volume_full_exposed = 0.0

    parallel_trap_list = model.parallel_trap_list[0]
    parallel_ccd = model.parallel_ccd
    serial_trap_list = model.serial_trap_list[0]
    serial_ccd = model.serial_ccd

    arguments = {
        parallel_trap_list.density: 0.1,
        parallel_trap_list.release_timescale: 0.2,
        parallel_ccd.full_well_depth: 0.3,
        parallel_ccd.well_notch_depth: 0.4,
        parallel_ccd.well_fill_power: 0.5,
        serial_trap_list.density: 0.6,
        serial_trap_list.release_timescale: 0.7,
        serial_ccd.full_well_depth: 0.8,
        serial_ccd.well_notch_depth: 0.9,
        serial_ccd.well_fill_power: 1.0,
    }

    instance = model.instance_for_arguments(arguments=arguments)

    assert instance.parallel_trap_list[0].density == 0.1
    assert instance.parallel_trap_list[0].release_timescale == 0.2
    assert instance.parallel_ccd.full_well_depth == 0.3
    assert instance.parallel_ccd.well_notch_depth == 0.4
    assert instance.parallel_ccd.well_fill_power == 0.5
    assert instance.serial_trap_list[0].density == 0.6
    assert instance.serial_trap_list[0].release_timescale == 0.7
    assert instance.serial_ccd.full_well_depth == 0.8
    assert instance.serial_ccd.well_notch_depth == 0.9
    assert instance.serial_ccd.well_fill_power == 1.0
