import os

import autofit as af
from autocti.model import arctic_params
from autocti.model import arctic_settings
from autocti.pipeline import phase as ph
from autocti.pipeline import pipeline as pl

from test.simulation import simulation_util
from test.integration import integration_util

test_type = 'parallel'
test_name = 'x1_species_x1_image_cosmic_rays'

test_path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + 'output/'
config_path = test_path + 'config'
af.conf.instance = af.conf.Config(config_path=config_path, output_path=output_path)


def pipeline():

    integration_util.reset_paths(test_name=test_name, output_path=output_path)

    parallel_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=2, n_levels=2000,
                                                 charge_injection_mode=False, readout_offset=0)
    cti_settings = arctic_settings.ArcticSettings(parallel=parallel_settings)
    data = simulation_util.load_test_ci_data(ci_data_type='ci_uniform_cosmic_rays', ci_data_model='parallel_x1',
                                             ci_data_resolution='patch', normalization=84700.0,
                                             load_cosmic_ray_image=True)
    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(ci_datas=[data], cti_settings=cti_settings)


def make_pipeline(test_name):

    class ParallelPhase(ph.ParallelPhase):

        def pass_priors(self, results):
            self.parallel_ccd.well_fill_alpha = 1.0
            self.parallel_ccd.well_fill_gamma = 0.0

    phase1 = ParallelPhase(phase_name='phase_1', phase_folders=[test_type, test_name],
                           optimizer_class=af.MultiNest,
                           parallel_species=[af.PriorModel(arctic_params.Species)],
                           parallel_ccd=arctic_params.CCD, columns=None)

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.sampling_efficiency = 0.2

    return pl.Pipeline(test_type, phase1)


if __name__ == "__main__":
    pipeline()
