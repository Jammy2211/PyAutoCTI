import autocti as ac
import autofit as af
from test_autocti.integration.tests import runner

test_type = "parallel"
test_name = "x3_species__x2_image__linked_phases"
ci_data_type = "ci_uniform"
ci_data_model = "parallel_x3"
resolution = "patch"
ci_normalizations = [84700.0]


clocker = ac.Clocker(parallel_express=2)


def make_pipeline(name, folders, search=af.DynestyStatic()):

    parallel_ccd = af.PriorModel(ac.CCD)

    parallel_ccd.full_well_depth = 8.47e4
    parallel_ccd.well_notch_depth = 1e-7

    phase1 = ac.PhaseCIImaging(
        name="phase_1",
        folders=folders,
        search=search,
        parallel_traps=[af.PriorModel(ac.TrapInstantCapture)],
        parallel_ccd=parallel_ccd,
    )

    previous_total_density = phase1.result.instance.parallel_trap[0].trap_density

    trap = af.PriorModel(ac.TrapInstantCapture)
    trap.density = af.UniformPrior(lower_limit=0.0, upper_limit=previous_total_density)

    phase2 = ac.PhaseCIImaging(
        name="phase_2",
        folders=folders,
        search=search,
        parallel_traps=3 * [trap],
        parallel_ccd=phase1.result.model.parallel_ccd,
    )

    phase2 = phase2.extend_with_hyper_noise_phases()

    phase3 = ac.PhaseCIImaging(
        name="phase_3",
        folders=folders,
        parallel_traps=phase1.result.model.parallel_trap,
        parallel_ccd=phase1.result.model.parallel_ccd,
        hyper_noise_scalar_of_ci_regions=phase1.result.hyper.instance.hyper_noise_scalar_of_ci_regions,
        hyper_noise_scalar_of_parallel_trails=phase1.result.hyper.instance.hyper_noise_scalar_of_parallel_trails,
        search=search,
    )

    return ac.Pipeline(name, phase1, phase2, phase3)


if __name__ == "__main__":

    import sys

    runner.run(sys.modules[__name__], clocker=clocker)
