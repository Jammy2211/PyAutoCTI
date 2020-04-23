import autofit as af
import autocti as ac
from test import runner

test_type = "features/total_density_range"
test_name = "serial"

test_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + "output/"
config_path = test_path + "config"
af.conf.instance = af.conf.Config(config_path=config_path, output_path=output_path)


serial_settings = ac.Settings(
    well_depth=84700,
    niter=1,
    express=2,
    n_levels=2000,
    charge_injection_mode=False,
    readout_offset=0,
)
clocker = ac.ArcticSettings(serial=serial_settings)


def make_pipeline(name, phase_folders, non_linear_class=af.MultiNest):
    class PhaseCI(ac.PhaseCI):
        def customize_priors(self, results):

            self.serial_ccd_volume.well_fill_alpha = 1.0
            self.serial_ccd_volume.well_fill_gamma = 0.0

    phase1 = PhaseCI(
        phase_name="phase_1",
        phase_folders=phase_folders,
        non_linear_class=non_linear_class,
        serial_traps=[af.PriorModel(ac.Trap)],
        serial_ccd_volume=ac.CCDVolume,
        serial_total_density_range=(0.1, 0.3),
    )

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.sampling_efficiency = 0.2

    return ac.Pipeline(name, phase1)


if __name__ == "__main__":

    import sys

    runner.run(sys.modules[__name__], clocker=clocker)
