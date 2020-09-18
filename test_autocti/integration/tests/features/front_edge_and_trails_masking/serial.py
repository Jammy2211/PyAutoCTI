import autocti as ac
import autofit as af
from test_autocti.integration.tests import runner

test_type = "features/front_edge_and_trails_masking"
test_name = "serial"

test_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + "output/"
config_path = test_path + "config"
conf.instance = conf.Config(config_path=config_path, output_path=output_path)


serial_settings = ac.Settings(
    well_depth=84700,
    niter=1,
    express=2,
    n_levels=2000,
    charge_injection_mode=False,
    offset=0,
)
clocker = ac.ArcticSettings(serial=serial_settings)


def make_pipeline(name, folders, search=):
    class PhaseCIImaging(ac.PhaseCIImaging):
        def customize_priors(self, results):

            self.serial_ccd.well_fill_alpha = 1.0
            self.serial_ccd.well_fill_gamma = 0.0

    phase1 = ac.PhaseCIImaging(
        phase_name="phase_1",
        folders=folders,
        search=search,
        serial_traps=[af.PriorModel(ac.TrapInstantCaptureWrap)],
        serial_ccd=serial_ccd,
        serial_front_edge_columns=(0, 1),
        serial_trails_columns=(0, 1),
    )

    phase1.search.n_live_points = 60
    phase1.search.const_efficiency_mode = True
    phase1.search.facc = 0.2

    return ac.Pipeline(name, phase1)


if __name__ == "__main__":

    import sys

    runner.run(sys.modules[__name__], clocker=clocker)
