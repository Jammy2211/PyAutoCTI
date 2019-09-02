import autofit as af
import autocti as ac
from test.integration.tests import runner

from multiprocessing import Pool

test_type = "serial"
test_name = "x1_species__x2_images__pool"
ci_data_type = "ci_uniform"
ci_data_model = "serial_x1"
ci_data_resolution = "patch"
ci_normalizations = [84700.0]


serial_settings = ac.Settings(
    well_depth=84700,
    niter=1,
    express=2,
    n_levels=2000,
    charge_injection_mode=False,
    readout_offset=0,
)
cti_settings = ac.ArcticSettings(serial=serial_settings)


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    class SerialPhase(ac.SerialPhase):
        def customize_priors(self, results):

            self.serial_ccd_volume.well_fill_alpha = 1.0
            self.serial_ccd_volume.well_fill_gamma = 0.0

    phase1 = SerialPhase(
        phase_name="phase_1",
        phase_folders=phase_folders,
        optimizer_class=optimizer_class,
        serial_species=[af.PriorModel(ac.Species)],
        serial_ccd_volume=ac.CCDVolume,
        rows=None,
    )

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.sampling_efficiency = 0.2

    return ac.Pipeline(name, phase1)


if __name__ == "__main__":

    import sys

    runner.run(sys.modules[__name__], cti_settings=cti_settings, pool=Pool(processes=2))
