import autofit as af
import autocti as ac
from test.integration import integration_util
from test.integration.tests import runner
from test.simulation import simulation_util

test_type = "parallel"
test_name = "x3_species__x1_image"
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
        parallel_species=[
            af.PriorModel(ac.Species),
            af.PriorModel(ac.Species),
            af.PriorModel(ac.Species),
        ],
        parallel_ccd_volume=ac.CCDVolume,
        columns=3,
    )

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.sampling_efficiency = 0.2

    return ac.Pipeline(name, phase1)


if __name__ == "__main__":

    import sys

    runner.run(sys.modules[__name__], cti_settings=cti_settings)
