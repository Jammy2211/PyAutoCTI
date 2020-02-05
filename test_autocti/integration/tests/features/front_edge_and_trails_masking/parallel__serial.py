import autofit as af
import autocti as ac
from test import runner

test_type = "features/front_edge_and_trails_masking"
test_name = "parallel_x1__serial_x1"

test_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + "output/"
config_path = test_path + "config"
af.conf.instance = af.conf.Config(config_path=config_path, output_path=output_path)


parallel_settings = ac.Settings(
    well_depth=84700,
    niter=1,
    express=2,
    n_levels=2000,
    charge_injection_mode=False,
    readout_offset=0,
)
serial_settings = ac.Settings(
    well_depth=84700,
    niter=1,
    express=2,
    n_levels=2000,
    charge_injection_mode=False,
    readout_offset=0,
)
cti_settings = ac.ArcticSettings(parallel=parallel_settings, serial=serial_settings)


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    class PhaseCI(ac.PhaseCI):
        def customize_priors(self, results):

            self.parallel_ccd_volume.well_fill_alpha = 1.0
            self.parallel_ccd_volume.well_fill_gamma = 0.0
            self.serial_ccd_volume.well_fill_alpha = 1.0
            self.serial_ccd_volume.well_fill_gamma = 0.0

    phase1 = PhaseCI(
        phase_name="phase_1",
        phase_folders=phase_folders,
        optimizer_class=optimizer_class,
        parallel_traps=[af.PriorModel(ac.Trap)],
        parallel_ccd_volume=ac.CCDVolume,
        serial_traps=[af.PriorModel(ac.Trap)],
        serial_ccd_volume=ac.CCDVolume,
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

    runner.run(sys.modules[__name__], cti_settings=cti_settings)
