import autocti as ac
import autofit as af
from test_autocti.integration.tests import runner

test_type = "features/total_density_range"
test_name = "parallel"

test_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + "output/"
config_path = test_path + "config"
conf.instance = conf.Config(config_path=config_path, output_path=output_path)


parallel_settings = ac.Settings(
    well_depth=84700,
    niter=1,
    express=2,
    n_levels=2000,
    charge_injection_mode=False,
    offset=0,
)
clocker = ac.ArcticSettings(parallel=parallel_settings)


def make_pipeline(name, folders, search):
    class PhaseCIImaging(ac.PhaseCIImaging):
        def customize_priors(self, results):
            self.parallel_ccd.well_fill_alpha = 1.0
            self.parallel_ccd.well_fill_gamma = 0.0

    phase1 = ac.PhaseCIImaging(
        phase_name="phase_1",
        folders=folders,
        search=search,
        parallel_traps=[af.PriorModel(ac.TrapInstantCaptureWrap)],
        parallel_ccd=parallel_ccd,
        columns=None,
        parallel_total_density_range=(0.1, 0.3),
    )

    phase1.search.n_live_points = 60
    phase1.search.const_efficiency_mode = True
    phase1.search.facc = 0.2

    return ac.Pipeline(name, phase1)


if __name__ == "__main__":

    import sys

    runner.run(sys.modules[__name__], clocker=clocker)
