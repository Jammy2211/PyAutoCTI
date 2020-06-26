import autocti as ac
import autofit as af
from test_autocti.integration.tests import runner

test_type = "parallel"
test_name = "x1_species__x2_image__cosmic_rays"
ci_data_type = "ci_uniform_cosmic_rays"
ci_data_model = "parallel_x1"
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
        parallel_traps=[af.PriorModel(ac.Trap)],
        parallel_ccd_volume=parallel_ccd_volume,
        settings=ac.PhaseSettingsCIImaging(columns=40),
    )

    phase1 = phase1.extend_with_hyper_noise_phases()

    phase2 = ac.PhaseCIImaging(
        phase_name="phase_2",
        folders=folders,
        parallel_traps=phase1.result.model.parallel_traps,
        parallel_ccd_volume=phase1.result.model.parallel_ccd_volume,
        hyper_noise_scalar_of_ci_regions=phase1.result.hyper_combined.instance.hyper_noise_scalar_of_ci_regions,
        hyper_noise_scalar_of_parallel_trails=phase1.result.hyper_combined.instance.hyper_noise_scalar_of_parallel_trails,
        search=search,
    )

    return ac.Pipeline(name, phase1, phase2)


if __name__ == "__main__":

    import sys

    runner.run(sys.modules[__name__], clocker=clocker, load_cosmic_ray_map=True)
