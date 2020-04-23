import autofit as af
import autocti as ac
from test import runner

from multiprocessing import Pool

test_type = "parallel"
test_name = "x1_species__x2_images__pool"
ci_data_type = "ci__uniform"
ci_data_model = "parallel_x1"
ci_data_resolution = "patch"
ci_normalizations = [10000.0, 84700.0]


parallel_settings = ac.Settings(
    well_depth=84700,
    niter=1,
    express=2,
    n_levels=2000,
    charge_injection_mode=False,
    readout_offset=0,
)
clocker = ac.ArcticSettings(parallel=parallel_settings)


def make_pipeline(name, phase_folders, non_linear_class=af.MultiNest):
    class PhaseCI(ac.PhaseCI):
        def customize_priors(self, results):

            self.parallel_ccd_volume.well_fill_alpha = 1.0
            self.parallel_ccd_volume.well_fill_gamma = 0.0

    phase1 = PhaseCI(
        phase_name="phase_1",
        phase_folders=phase_folders,
        non_linear_class=non_linear_class,
        parallel_traps=[af.PriorModel(ac.Trap)],
        parallel_ccd_volume=ac.CCDVolume,
        columns=40,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 60
    phase1.optimizer.sampling_efficiency = 0.2

    return ac.Pipeline(name, phase1)


if __name__ == "__main__":

    import sys

    runner.run(sys.modules[__name__], clocker=clocker, pool=Pool(processes=2))
