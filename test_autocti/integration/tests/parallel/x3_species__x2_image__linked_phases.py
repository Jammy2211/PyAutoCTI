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

    parallel_ccd_volume = af.PriorModel(ac.CCDVolume)

    parallel_ccd_volume.well_max_height = 8.47e4
    parallel_ccd_volume.well_notch_depth = 1e-7

    phase1 = ac.PhaseCIImaging(
        phase_name="phase_1",
        folders=folders,
        search=search,
        parallel_traps=[af.PriorModel(ac.Trap)],
        parallel_ccd_volume=parallel_ccd_volume,
    )

    previous_total_density = phase1.result.instance.parallel_traps[0].trap_density

    trap = af.PriorModel(ac.Trap)
    trap.density = af.UniformPrior(lower_limit=0.0, upper_limit=previous_total_density)

    phase2 = ac.PhaseCIImaging(
        phase_name="phase_2",
        folders=folders,
        search=search,
        parallel_traps=3 * [trap],
        parallel_ccd_volume=phase1.result.model.parallel_ccd_volume,
    )

    phase2 = phase2.extend_with_hyper_noise_phases()

    phase3 = ac.PhaseCIImaging(
        phase_name="phase_3",
        folders=folders,
        parallel_traps=phase1.result.model.parallel_traps,
        parallel_ccd_volume=phase1.result.model.parallel_ccd_volume,
        hyper_noise_scalar_of_ci_regions=phase1.result.hyper_combined.instance.hyper_noise_scalar_of_ci_regions,
        hyper_noise_scalar_of_parallel_trails=phase1.result.hyper_combined.instance.hyper_noise_scalar_of_parallel_trails,
        search=search,
    )

    return ac.Pipeline(name, phase1, phase2, phase3)


if __name__ == "__main__":

    import sys

    runner.run(sys.modules[__name__], clocker=clocker)
