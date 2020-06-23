import autocti as ac
import autofit as af
from test_autocti.integration.tests import runner

test_type = "parallel"
test_name = "x3_species__x1_image"
ci_data_type = "ci_uniform"
ci_data_model = "parallel_x3"
resolution = "patch"
ci_normalizations = [10000.0, 84700.0]


clocker = ac.Clocker(parallel_express=2)


def make_pipeline(name, folders, search=af.DynestyStatic()):

    parallel_ccd_volume = af.PriorModel(ac.CCDVolume)

    parallel_ccd_volume.well_max_height = 8.47e4
    parallel_ccd_volume.well_notch_depth = 1e-7

    phase1 = ac.PhaseCIImaging(
        phase_name="phase_1",
        folders=folders,
        search=search,
        parallel_traps=[
            af.PriorModel(ac.Trap),
            af.PriorModel(ac.Trap),
            af.PriorModel(ac.Trap),
        ],
        parallel_ccd_volume=parallel_ccd_volume,
        columns=3,
    )

    phase1.search.n_live_points = 60
    phase1.search.const_efficiency_mode = True
    phase1.search.facc = 0.2

    phase1 = phase1.extend_with_hyper_noise_phases()

    phase2 = ac.PhaseCIImaging(
        phase_name="phase_2",
        folders=folders,
        parallel_traps=phase1.result.model.parallel_traps,
        parallel_ccd_volume=phase1.result.model.parallel_ccd_volume,
        hyper_noise_scalar_of_ci_regions=phase1.result.hyper_combined.instance.hyper_noise_scalar_of_ci_regions,
        hyper_noise_scalar_of_parallel_trails=phase1.result.hyper_combined.instance.hyper_noise_scalar_of_parallel_trails,
        search=search,
        columns=None,
    )

    # For the final CTI model, constant efficiency mode has a tendancy to sample parameter space too fast and infer an
    # inaccurate model. Thus, we turn it off for phase 2.

    phase2.search.const_efficiency_mode = False
    phase2.search.n_live_points = 50
    phase2.search.facc = 0.3

    return ac.Pipeline(name, phase1)


if __name__ == "__main__":

    import sys

    runner.run(sys.modules[__name__], clocker=clocker)
