import autocti as ac
import autofit as af
from test_autocti.integration.tests import runner

test_type = "serial"
test_name = "x3_species__x1_image"
ci_data_type = "ci_uniform"
ci_data_model = "serial_x3"
resolution = "patch"
ci_normalizations = [10000.0, 84700.0]


clocker = ac.Clocker(serial_express=2)


def make_pipeline(name, folders, search=af.DynestyStatic()):

    serial_ccd_volume = af.PriorModel(ac.CCDVolume)

    serial_ccd_volume.well_max_height = 8.47e4
    serial_ccd_volume.well_notch_depth = 1e-7

    phase1 = ac.PhaseCIImaging(
        phase_name="phase_1",
        folders=folders,
        search=search,
        serial_traps=3 * [af.PriorModel(ac.Trap)],
        serial_ccd_volume=serial_ccd_volume,
        columns=3,
    )

    phase1 = phase1.extend_with_hyper_noise_phases()

    phase2 = ac.PhaseCIImaging(
        phase_name="phase_2",
        folders=folders,
        serial_traps=phase1.result.model.serial_traps,
        serial_ccd_volume=phase1.result.model.serial_ccd_volume,
        hyper_noise_scalar_of_ci_regions=phase1.result.hyper_combined.instance.hyper_noise_scalar_of_ci_regions,
        hyper_noise_scalar_of_serial_trails=phase1.result.hyper_combined.instance.hyper_noise_scalar_of_serial_trails,
        search=search,
    )

    return ac.Pipeline(name, phase1)


if __name__ == "__main__":

    import sys

    runner.run(sys.modules[__name__], clocker=clocker)
