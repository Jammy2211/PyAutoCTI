import autofit as af
import autocti as ac


def test__cti_model__creates_correct_model_from_inputs():

    model = af.Model(
        ac.CTI2D,
        parallel_traps=[ac.TrapInstantCapture],
        parallel_ccd=ac.CCDPhase,
        serial_traps=[ac.TrapInstantCapture],
        serial_ccd=ac.CCDPhase,
    )

    parallel_traps = model.parallel_traps[0]
    parallel_ccd = model.parallel_ccd
    serial_traps = model.serial_traps[0]
    serial_ccd = model.serial_ccd

    arguments = {
        parallel_traps.density: 0.1,
        parallel_traps.release_timescale: 0.2,
        parallel_ccd.full_well_depth: 0.3,
        parallel_ccd.well_notch_depth: 0.4,
        parallel_ccd.well_fill_power: 0.5,
        serial_traps.density: 0.6,
        serial_traps.release_timescale: 0.7,
        serial_ccd.full_well_depth: 0.8,
        serial_ccd.well_notch_depth: 0.9,
        serial_ccd.well_fill_power: 1.0,
    }

    instance = model.instance_for_arguments(arguments=arguments)

    assert instance.parallel_traps[0].density == 0.1
    assert instance.parallel_traps[0].release_timescale == 0.2
    assert instance.parallel_ccd.full_well_depth == 0.3
    assert instance.parallel_ccd.well_notch_depth == 0.4
    assert instance.parallel_ccd.well_fill_power == 0.5
    assert instance.serial_traps[0].density == 0.6
    assert instance.serial_traps[0].release_timescale == 0.7
    assert instance.serial_ccd.full_well_depth == 0.8
    assert instance.serial_ccd.well_notch_depth == 0.9
    assert instance.serial_ccd.well_fill_power == 1.0


def test__recognises_type_of_fit_from_model():

    model = af.Model(
        ac.CTI2D, parallel_traps=[ac.TrapInstantCapture], parallel_ccd=ac.CCDPhase
    )

    assert ac.util.model.is_parallel_fit(model=model) is True
    assert ac.util.model.is_serial_fit(model=model) is False
    assert ac.util.model.is_parallel_and_serial_fit(model=model) is False

    model = af.Model(
        ac.CTI2D, serial_traps=[ac.TrapInstantCapture], serial_ccd=ac.CCDPhase
    )

    assert ac.util.model.is_parallel_fit(model=model) is False
    assert ac.util.model.is_serial_fit(model=model) is True
    assert ac.util.model.is_parallel_and_serial_fit(model=model) is False

    model = af.Model(
        ac.CTI2D,
        parallel_traps=[ac.TrapInstantCapture],
        parallel_ccd=ac.CCDPhase,
        serial_traps=[ac.TrapInstantCapture],
        serial_ccd=ac.CCDPhase,
    )

    assert ac.util.model.is_parallel_fit(model=model) is False
    assert ac.util.model.is_serial_fit(model=model) is False
    assert ac.util.model.is_parallel_and_serial_fit(model=model) is True
