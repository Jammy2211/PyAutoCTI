import autofit as af
import autocti as ac
from test.integration.tests import runner

test_type = "parallel"
test_name = "x1_species__x4_images__hyper"
ci_data_type = "ci_uniform"
ci_data_model = "parallel_x1"
ci_data_resolution = "patch"
ci_normalizations = [1000.0, 10000.0, 25000.0, 84700.0]


parallel_settings = ac.Settings(
    well_depth=84700,
    niter=1,
    express=2,
    n_levels=2000,
    charge_injection_mode=False,
    readout_offset=0,
)
cti_settings = ac.ArcticSettings(parallel=parallel_settings)


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    class ParallelPhase(ac.ParallelPhase):
        def customize_priors(self, results):
            self.parallel_ccd_volume.well_fill_alpha = 1.0
            self.parallel_ccd_volume.well_fill_gamma = 0.0

    phase1 = ParallelPhase(
        phase_name="phase_1",
        phase_folders=phase_folders,
        optimizer_class=optimizer_class,
        parallel_species=[af.PriorModel(ac.Species)],
        parallel_ccd_volume=ac.CCDVolume,
        columns=40,
    )

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.sampling_efficiency = 0.2

    class ParallelModelFixedPhase(ac.ParallelPhase):
        def customize_priors(self, results):

            self.parallel_species = results.from_phase(
                "phase_1"
            ).constant.parallel_species
            self.parallel_ccd_volume = results.from_phase(
                "phase_1"
            ).constant.parallel_ccd_volume

    phase2 = ParallelModelFixedPhase(
        phase_name="phase_2",
        phase_folders=phase_folders,
        parallel_species=[af.PriorModel(ac.Species)],
        parallel_ccd_volume=ac.CCDVolume,
        optimizer_class=optimizer_class,
        columns=None,
    )

    class ParallelFixedPhase(ac.ParallelPhase):
        def customize_priors(self, results):

            self.hyper_noise_scalars = results.from_phase(
                "phase_2"
            ).constant.hyper_noise_scalars
            self.parallel_species = results.from_phase(
                "phase_1"
            ).variable.parallel_species
            self.parallel_ccd_volume = results.from_phase(
                "phase_1"
            ).variable.parallel_ccd_volume
            self.parallel_ccd_volume.well_fill_alpha = 1.0
            self.parallel_ccd_volume.well_fill_gamma = 0.0

    phase3 = ParallelFixedPhase(
        phase_name="phase_3",
        phase_folders=phase_folders,
        optimizer_class=optimizer_class,
        columns=None,
    )

    # For the final CTI model, constant efficiency mode has a tendancy to sample parameter space too fast and infer an
    # inaccurate model. Thus, we turn it off for phase 2.

    phase3.optimizer.const_efficiency_mode = False
    phase3.optimizer.n_live_points = 50
    phase3.optimizer.sampling_efficiency = 0.3

    return ac.Pipeline(name, phase1, phase2, phase3)


if __name__ == "__main__":

    import sys

    runner.run(sys.modules[__name__], cti_settings=cti_settings)
