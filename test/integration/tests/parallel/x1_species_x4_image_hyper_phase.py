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

test_type = 'parallel'
test_name = 'x1_species_x4_image_hyper_phase'

test_path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + 'output/'
config_path = test_path + 'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)

def pipeline():

    integration_util.reset_paths(test_name=test_name, output_path=output_path)

    parallel_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=2, n_levels=2000,
                                                 charge_injection_mode=False, readout_offset=0)
    cti_settings = arctic_settings.ArcticSettings(parallel=parallel_settings)
    data = list(map(lambda normalization :
                    simulation_util.load_test_ci_data(ci_data_type='ci_uniform', ci_data_model='parallel_x1',
                                                      ci_data_resolution='patch', normalization=normalization),
                    [1000.0, 10000.0, 25000.0, 84700.0]))
    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(ci_datas=data, cti_settings=cti_settings)


def make_pipeline(test_name):

    class ParallelPhase(ph.ParallelPhase):

        def pass_priors(self, results):
            self.parallel_ccd.well_fill_alpha = 1.0
            self.parallel_ccd.well_fill_gamma = 0.0

    phase1 = ParallelPhase(phase_name='phase_1', phase_folders=[test_type, test_name],
                           optimizer_class=nl.MultiNest,
                           parallel_species=[prior_model.PriorModel(arctic_params.Species)],
                           parallel_ccd=arctic_params.CCD, columns=40)

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.sampling_efficiency = 0.2

    class ParallelHyperModelFixedPhase(ph.ParallelHyperPhase):

        def pass_priors(self, results):

            self.parallel_species = results.from_phase('phase_1').constant.parallel_species
            self.parallel_ccd = results.from_phase('phase_1').constant.parallel_ccd

    phase2 = ParallelHyperModelFixedPhase(phase_name='phase_2', phase_folders=[test_type, test_name],
                                          parallel_species=[prior_model.PriorModel(arctic_params.Species)],
                                          parallel_ccd=arctic_params.CCD,
                                          optimizer_class=nl.MultiNest, columns=None)

    class ParallelHyperFixedPhase(ph.ParallelHyperPhase):

        def pass_priors(self, results):

            self.hyper_noise_scalars = results.from_phase('phase_2').constant.hyper_noise_scalars
            self.parallel_species = results.from_phase('phase_1').variable.parallel_species
            self.parallel_ccd = results.from_phase('phase_1').variable.parallel_ccd
            self.parallel_ccd.well_fill_alpha = 1.0
            self.parallel_ccd.well_fill_gamma = 0.0

    phase3 = ParallelHyperFixedPhase(phase_name='phase_3', phase_folders=[test_type, test_name],
                                     optimizer_class=nl.MultiNest, columns=None)

    # For the final CTI model, constant efficiency mode has a tendancy to sample parameter space too fast and infer an
    # inaccurate model. Thus, we turn it off for phase 2.

    phase3.optimizer.const_efficiency_mode = False
    phase3.optimizer.n_live_points = 50
    phase3.optimizer.sampling_efficiency = 0.3

    return pl.Pipeline(test_type, phase1, phase2, phase3)


if __name__ == "__main__":
    pipeline()