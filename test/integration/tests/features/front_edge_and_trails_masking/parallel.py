import os

from autofit import conf
from autofit.mapper import prior_model
from autofit.optimize import non_linear as nl

from autocti.model import arctic_params
from autocti.model import arctic_settings
from autocti.pipeline import phase as ph
from autocti.pipeline import pipeline as pl

from test.simulation import simulation_util
from test.integration import integration_util

test_type = 'features/front_edge_and_trails_masking'
test_name = 'parallel'

test_path = '{}/../../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + 'output/'
config_path = test_path + 'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)


def pipeline():

    integration_util.reset_paths(test_name=test_name, output_path=output_path)

    parallel_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=2, n_levels=2000,
                                                 charge_injection_mode=False, readout_offset=0)
    cti_settings = arctic_settings.ArcticSettings(parallel=parallel_settings)
    data = simulation_util.load_test_ci_data(ci_data_type='ci_uniform', ci_data_model='parallel_x1',
                                             ci_data_resolution='patch',normalization=84700.0)
    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(ci_datas=[data], cti_settings=cti_settings)


def make_pipeline(test_name):

    class ParallelPhase(ph.ParallelPhase):

        def pass_priors(self, results):
            self.parallel_ccd.well_fill_alpha = 1.0
            self.parallel_ccd.well_fill_gamma = 0.0

    phase1 = ParallelPhase(phase_name='phase_1', phase_folders=[test_type, test_name],
                           optimizer_class=nl.MultiNest,
                           parallel_species=[prior_model.PriorModel(arctic_params.Species)],
                           parallel_ccd=arctic_params.CCD, columns=None,
                           parallel_front_edge_mask_rows=(0, 1), parallel_trails_mask_rows=(0,1))

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.sampling_efficiency = 0.2

    return pl.Pipeline(test_type, phase1)


if __name__ == "__main__":
    pipeline()
