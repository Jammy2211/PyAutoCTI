import autocti as ac
import autofit as af
from test_autocti.integration.tests import runner

test_type = "parallel_x1__serial_x1"
test_name = "x1_species__x1_image__hyper"
ci_data_type = "ci_uniform"
ci_data_model = "parallel_x1__serial_x1"
resolution = "patch"
ci_normalizations = [10000.0, 84700.0]


clocker = ac.Clocker(parallel_express=2, serial_express=2)


def make_pipeline(name, folders, search=af.DynestyStatic()):

    parallel_ccd_volume = af.PriorModel(ac.CCDVolume)

    parallel_ccd_volume.well_max_height = 8.47e4
    parallel_ccd_volume.well_notch_depth = 1e-7

    serial_ccd_volume = af.PriorModel(ac.CCDVolume)

    serial_ccd_volume.well_max_height = 8.47e4
    serial_ccd_volume.well_notch_depth = 1e-7

    phase1 = ac.PhaseCIImaging(
        phase_name="phase_1",
        folders=folders,
        search=search,
        parallel_traps=[af.PriorModel(ac.Trap)],
        parallel_ccd_volume=parallel_ccd_volume,
        serial_traps=[af.PriorModel(ac.Trap)],
        serial_ccd_volume=serial_ccd_volume,
    )

    phase2 = ac.PhaseCIImaging(
        phase_name="phase_2",
        folders=folders,
        parallel_traps=phase1.result.instance.parallel_traps,
        parallel_ccd_volume=phase1.result.instance.parallel_ccd_volume,
        serial_traps=phase1.result.instance.serial_traps,
        serial_ccd_volume=phase1.result.instance.serial_ccd_volume,
        hyper_noise_scalar_of_ci_regions=ac.ci.CIHyperNoiseScalar,
        hyper_noise_scalar_of_parallel_trails=ac.ci.CIHyperNoiseScalar,
        hyper_noise_scalar_of_serial_trails=ac.ci.CIHyperNoiseScalar,
        hyper_noise_scalar_of_serial_overscan_no_trails=ac.ci.CIHyperNoiseScalar,
        search=search,
    )

    phase3 = ac.PhaseCIImaging(
        phase_name="phase_3",
        folders=folders,
        parallel_traps=phase1.result.model.parallel_traps,
        parallel_ccd_volume=phase1.result.model.parallel_ccd_volume,
        serial_traps=phase1.result.model.serial_traps,
        serial_ccd_volume=phase1.result.model.serial_ccd_volume,
        hyper_noise_scalar_of_ci_regions=phase2.result.instance.hyper_noise_scalar_of_ci_regions,
        hyper_noise_scalar_of_parallel_trails=phase2.result.instance.hyper_noise_scalar_of_parallel_trails,
        hyper_noise_scalar_of_serial_trails=phase2.result.instance.hyper_noise_scalar_of_serial_trails,
        hyper_noise_scalar_of_serial_overscan_no_trails=phase2.result.instance.hyper_noise_scalar_of_serial_overscan_no_trails,
        search=search,
    )

    return ac.Pipeline(name, phase1, phase2, phase3)


if __name__ == "__main__":

    import sys

    runner.run(sys.modules[__name__], clocker=clocker)
