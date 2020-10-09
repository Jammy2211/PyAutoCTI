import autocti as ac
import autofit as af
from test_autocti.integration.tests import runner

test_type = "features/front_edge_and_trails_masking"
test_name = "parallel_x1__serial_x1"

test_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + "output/"
config_path = test_path + "config"
conf.instance = conf.Config(config_path=config_path, output_path=output_path)


clocker = ac.Clocker(parallel_express=2, serial_express=2)


def make_pipeline(name, folders, search):
    class PhaseCIImaging(ac.PhaseCIImaging):
        def customize_priors(self, results):

            self.parallel_ccd.well_fill_alpha = 1.0
            self.parallel_ccd.well_fill_gamma = 0.0
            self.serial_ccd.well_fill_alpha = 1.0
            self.serial_ccd.well_fill_gamma = 0.0

    phase1 = ac.PhaseCIImaging(
        name="phase_1",
        folders=folders,
        search=search,
        parallel_traps=[af.PriorModel(ac.TrapInstantCapture)],
        parallel_ccd=parallel_ccd,
        serial_traps=[af.PriorModel(ac.TrapInstantCapture)],
        serial_ccd=serial_ccd,
        parallel_front_edge_rows=(0, 1),
        parallel_trails_rows=(0, 1),
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
