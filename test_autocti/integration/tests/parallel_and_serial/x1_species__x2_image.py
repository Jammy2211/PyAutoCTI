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


def make_pipeline(name, phase_folders, non_linear_class=af.MultiNest):

    parallel_ccd_volume = af.PriorModel(ac.CCDVolume)

    parallel_ccd_volume.well_max_height = 8.47e4
    parallel_ccd_volume.well_notch_depth = 1e-7

    serial_ccd_volume = af.PriorModel(ac.CCDVolume)

    serial_ccd_volume.well_max_height = 8.47e4
    serial_ccd_volume.well_notch_depth = 1e-7

    phase1 = ac.PhaseCIImaging(
        phase_name="phase_1",
        phase_folders=phase_folders,
        non_linear_class=non_linear_class,
        parallel_traps=[af.PriorModel(ac.Trap)],
        parallel_ccd_volume=parallel_ccd_volume,
        serial_traps=[af.PriorModel(ac.Trap)],
        serial_ccd_volume=serial_ccd_volume,
    )

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.sampling_efficiency = 0.2

    class HyperModelFixedPhaseCI(ac.PhaseCIImaging):
        def customize_priors(self, results):

            self.parallel_traps = results.from_phase("phase_1").instance.parallel_traps
            self.parallel_ccd_volume = results.from_phase(
                "phase_1"
            ).instance.parallel_ccd_volume
            self.serial_traps = results.from_phase("phase_1").instance.serial_traps
            self.serial_ccd_volume = results.from_phase(
                "phase_1"
            ).instance.serial_ccd_volume

    phase2 = HyperModelFixedPhaseCI(
        phase_name="phase_2",
        phase_folders=phase_folders,
        parallel_traps=[af.PriorModel(ac.Trap)],
        parallel_ccd_volume=parallel_ccd_volume,
        serial_traps=[af.PriorModel(ac.Trap)],
        serial_ccd_volume=serial_ccd_volume,
        hyper_noise_scalar_of_ci_regions=ac.CIHyperNoiseScalar,
        hyper_noise_scalar_of_parallel_trails=ac.CIHyperNoiseScalar,
        hyper_noise_scalar_of_serial_trails=ac.CIHyperNoiseScalar,
        hyper_noise_scalar_of_serial_overscan_no_trails=ac.CIHyperNoiseScalar,
        non_linear_class=non_linear_class,
    )

    class FixedPhaseCI(ac.PhaseCIImaging):
        def customize_priors(self, results):

            self.hyper_noise_scalar_of_ci_regions = results.from_phase(
                "phase_2"
            ).instance.hyper_noise_scalar_of_ci_regions

            self.hyper_noise_scalar_of_parallel_trails = results.from_phase(
                "phase_2"
            ).instance.hyper_noise_scalar_of_parallel_trails

            self.hyper_noise_scalar_of_serial_trails = results.from_phase(
                "phase_2"
            ).instance.hyper_noise_scalar_of_serial_trails

            self.hyper_noise_scalar_of_serial_overscan_no_trails = results.from_phase(
                "phase_2"
            ).instance.hyper_noise_scalar_of_serial_overscan_no_trails

    phase3 = FixedPhaseCI(
        phase_name="phase_3",
        phase_folders=phase_folders,
        parallel_traps=phase1.result.model.parallel_traps,
        parallel_ccd_volume=phase1.result.model.parallel_ccd_volume,
        serial_traps=phase1.result.model.serial_traps,
        serial_ccd_volume=phase1.result.model.serial_ccd_volume,
        hyper_noise_scalar_of_ci_regions=ac.CIHyperNoiseScalar,
        hyper_noise_scalar_of_parallel_trails=ac.CIHyperNoiseScalar,
        hyper_noise_scalar_of_serial_trails=ac.CIHyperNoiseScalar,
        hyper_noise_scalar_of_serial_overscan_no_trails=ac.CIHyperNoiseScalar,
        non_linear_class=non_linear_class,
        rows=None,
    )

    # For the final CTI model, constant efficiency mode has a tendancy to sample parameter space too fast and infer an
    # inaccurate model. Thus, we turn it off for phase 2.

    phase3.optimizer.const_efficiency_mode = False
    phase3.optimizer.n_live_points = 50
    phase3.optimizer.sampling_efficiency = 0.3

    return ac.Pipeline(name, phase1, phase2, phase3)


if __name__ == "__main__":

    import sys

    runner.run(sys.modules[__name__], clocker=clocker)
