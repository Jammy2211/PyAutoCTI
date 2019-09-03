import autofit as af
import autocti as ac
from test.integration.tests import runner

test_type = "parallel"
test_name = "x3_species__x2_image__linked_phases"
ci_data_type = "ci_uniform"
ci_data_model = "parallel_x3"
ci_data_resolution = "patch"
ci_normalizations = [84700.0]


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
        columns=None,
    )

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.sampling_efficiency = 0.2

    class ParallelPhase(ac.ParallelPhase):
        def customize_priors(self, results):

            previous_total_density = (
                results[-1].constant.parallel_species[0].trap_density
            )

            self.parallel_species[0].trap_density = af.UniformPrior(
                lower_limit=0.0, upper_limit=previous_total_density
            )
            self.parallel_species[1].trap_density = af.UniformPrior(
                lower_limit=0.0, upper_limit=previous_total_density
            )
            self.parallel_species[2].trap_density = af.UniformPrior(
                lower_limit=0.0, upper_limit=previous_total_density
            )
            self.parallel_species[0].trap_lifetime = af.UniformPrior(
                lower_limit=0.0, upper_limit=30.0
            )
            self.parallel_species[1].trap_lifetime = af.UniformPrior(
                lower_limit=0.0, upper_limit=30.0
            )
            self.parallel_species[2].trap_lifetime = af.UniformPrior(
                lower_limit=0.0, upper_limit=30.0
            )

            self.parallel_ccd_volume.well_notch_depth = results.from_phase(
                "phase_1"
            ).variable.parallel_ccd_volume.well_notch_depth
            self.parallel_ccd_volume.well_fill_beta = results.from_phase(
                "phase_1"
            ).variable.parallel_ccd_volume.well_fill_beta

    phase2 = ParallelPhase(
        phase_name="phase_2",
        phase_folders=phase_folders,
        optimizer_class=optimizer_class,
        parallel_species=[
            af.PriorModel(ac.Species),
            af.PriorModel(ac.Species),
            af.PriorModel(ac.Species),
        ],
        parallel_ccd_volume=ac.CCDVolume,
        columns=3,
    )

    phase2.optimizer.n_live_points = 60
    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.sampling_efficiency = 0.2

    return ac.Pipeline(name, phase1, phase2)


if __name__ == "__main__":

    import sys

    runner.run(sys.modules[__name__], cti_settings=cti_settings)
