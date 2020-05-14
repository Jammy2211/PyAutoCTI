import autocti as ac
import autofit as af
from test_autocti.integration.tests import runner

test_type = "features/front_edge_and_trails_masking"
test_name = "parallel_x1__serial_x1"

test_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + "output/"
config_path = test_path + "config"
af.conf.instance = af.conf.Config(config_path=config_path, output_path=output_path)


clocker = ac.Clocker(parallel_express=2, serial_express=2)


def make_pipeline(name, phase_folders, non_linear_class=af.MultiNest):
    class PhaseCIImaging(ac.PhaseCIImaging):
        def customize_priors(self, results):

            self.parallel_ccd_volume.well_fill_alpha = 1.0
            self.parallel_ccd_volume.well_fill_gamma = 0.0
            self.serial_ccd_volume.well_fill_alpha = 1.0
            self.serial_ccd_volume.well_fill_gamma = 0.0

    phase1 = ac.PhaseCIImaging(
        phase_name="phase_1",
        phase_folders=phase_folders,
        non_linear_class=non_linear_class,
        parallel_traps=[af.PriorModel(ac.Trap)],
        parallel_ccd_volume=parallel_ccd_volume,
        serial_traps=[af.PriorModel(ac.Trap)],
        serial_ccd_volume=serial_ccd_volume,
        parallel_front_edge_mask_rows=(0, 1),
        parallel_trails_mask_rows=(0, 1),
        serial_front_edge_mask_columns=(0, 1),
        serial_trails_mask_columns=(0, 1),
    )

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.sampling_efficiency = 0.2

    return ac.Pipeline(name, phase1)


if __name__ == "__main__":

    import sys

    runner.run(sys.modules[__name__], clocker=clocker)
