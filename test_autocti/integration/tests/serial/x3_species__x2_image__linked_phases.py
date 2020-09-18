import autocti as ac
import autofit as af
from test_autocti.integration.tests import runner

test_type = "serial"
test_name = "x3_species__x2_image__linked_phases"
ci_data_type = "ci_uniform"
ci_data_model = "serial_x3"
resolution = "patch"
ci_normalizations = [84700.0]


clocker = ac.ClockerWrap(serial_express=2)


def make_pipeline(name, folders, search=af.DynestyStatic()):

    serial_ccd = af.PriorModel(ac.CCDWrap)

    serial_ccd.full_well_depth = 8.47e4
    serial_ccd.well_notch_depth = 1e-7

    phase1 = ac.PhaseCIImaging(
        phase_name="phase_1",
        folders=folders,
        search=search,
        serial_traps=[af.PriorModel(ac.TrapInstantCaptureWrap)],
        serial_ccd=serial_ccd,
        columns=None,
    )

    previous_total_density = phase1.result.instance.serial_traps[0].trap_density

    trap = af.PriorModel(ac.TrapInstantCaptureWrap)
    trap.density = af.UniformPrior(lower_limit=0.0, upper_limit=previous_total_density)

    phase2 = ac.PhaseCIImaging(
        phase_name="phase_2",
        folders=folders,
        search=search,
        serial_traps=3 * [trap],
        serial_ccd=phase1.result.model.serial_ccd,
    )

    phase2 = phase2.extend_with_hyper_noise_phases()

    phase3 = ac.PhaseCIImaging(
        phase_name="phase_3",
        folders=folders,
        serial_traps=phase1.result.model.serial_traps,
        serial_ccd=phase1.result.model.serial_ccd,
        hyper_noise_scalar_of_ci_regions=phase1.result.hyper_combined.instance.hyper_noise_scalar_of_ci_regions,
        hyper_noise_scalar_of_serial_trails=phase1.result.hyper_combined.instance.hyper_noise_scalar_of_serial_trails,
        search=search,
    )

    return ac.Pipeline(name, phase1, phase2, phase3)


if __name__ == "__main__":

    import sys

    runner.run(sys.modules[__name__], clocker=clocker)
