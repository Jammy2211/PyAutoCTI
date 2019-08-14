import os

import autofit as af
from autocti.model import arctic_params
from autocti.model import arctic_settings

from autocti.pipeline import phase as ph
from autocti.pipeline import pipeline as pl
from test.simulation import simulation_util
from test.integration import integration_util
from test.integration.tests import runner

from multiprocessing import Pool

test_type = "parallel_and_serial"
test_name = "x1_species_x2_images_use_pool"
ci_data_type = "ci_uniform"
ci_data_model = "parallel_and_serial_x1"
ci_data_resolution = "patch"
ci_normalizations = [84700.0]

test_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + "output/"
config_path = test_path + "config"
af.conf.instance = af.conf.Config(config_path=config_path, output_path=output_path)

parallel_settings = arctic_settings.Settings(
    well_depth=84700,
    niter=1,
    express=2,
    n_levels=2000,
    charge_injection_mode=False,
    readout_offset=0,
)
serial_settings = arctic_settings.Settings(
    well_depth=84700,
    niter=1,
    express=2,
    n_levels=2000,
    charge_injection_mode=False,
    readout_offset=0,
)
cti_settings = arctic_settings.ArcticSettings(
    parallel=parallel_settings, serial=serial_settings
)


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    class ParallelSerialPhase(ph.ParallelSerialPhase):
        def pass_priors(self, results):

            self.parallel_ccd.well_fill_alpha = 1.0
            self.parallel_ccd.well_fill_gamma = 0.0
            self.serial_ccd.well_fill_alpha = 1.0
            self.serial_ccd.well_fill_gamma = 0.0

    phase1 = ParallelSerialPhase(
        phase_name="phase_1",
        phase_folders=phase_folders,
        optimizer_class=optimizer_class,
        parallel_species=[af.PriorModel(arctic_params.Species)],
        parallel_ccd=arctic_params.CCD,
        serial_species=[af.PriorModel(arctic_params.Species)],
        serial_ccd=arctic_params.CCD,
    )
    phase1.optimizer.n_live_points = 60
    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.sampling_efficiency = 0.2

    return pl.Pipeline(name, phase1)


if __name__ == "__main__":

    import sys

    runner.run(sys.modules[__name__], cti_settings=cti_settings)
