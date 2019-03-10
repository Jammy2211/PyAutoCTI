import os
from multiprocessing import Pool

from autofit import conf
from autofit.mapper import prior_model
from autofit.optimize import non_linear as nl

from autocti.model import arctic_params
from autocti.model import arctic_settings
from autocti.pipeline import phase as ph
from autocti.pipeline import pipeline as pl
from test.simulation import simulation_util
from test.integration import integration_util

test_type = 'parallel_and_serial'
test_name = 'x3_species_x2_images_use_pool'

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = path + 'output/' + test_type
config_path = path + 'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)


def pipeline():

    integration_util.reset_paths(test_name=test_name, output_path=output_path)
    parallel_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=2, n_levels=2000,
                                                 charge_injection_mode=False, readout_offset=0)
    serial_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=2, n_levels=2000,
                                                 charge_injection_mode=False, readout_offset=0)
    cti_settings = arctic_settings.ArcticSettings(parallel=parallel_settings, serial=serial_settings)
    data = list(map(lambda normalization :
                    simulation_util.load_test_ci_data(data_name='ci_uniform_parallel_and_serial_x3_species',
                                                      data_resolution='patch', normalization=normalization),
                    [1000.0, 84700.0]))
    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(ci_datas=data, cti_settings=cti_settings, pool=Pool(processes=2))


def make_pipeline(test_name):
    class ParallelSerialPhase(ph.ParallelSerialPhase):

        def pass_priors(self, previous_results):
            self.parallel_ccd.well_fill_alpha = 1.0
            self.parallel_ccd.well_fill_gamma = 0.0
            self.serial_ccd.well_fill_alpha = 1.0
            self.serial_ccd.well_fill_gamma = 0.0

    phase1 = ParallelSerialPhase(optimizer_class=nl.MultiNest,
                                 parallel_species=[prior_model.PriorModel(arctic_params.Species),
                                                   prior_model.PriorModel(arctic_params.Species),
                                                   prior_model.PriorModel(arctic_params.Species)],
                                 parallel_ccd=arctic_params.CCD,
                                 serial_species=[prior_model.PriorModel(arctic_params.Species),
                                                 prior_model.PriorModel(arctic_params.Species),
                                                 prior_model.PriorModel(arctic_params.Species)],
                                 serial_ccd=arctic_params.CCD, phase_name="{}/phase1".format(test_name))

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.sampling_efficiency = 0.2

    return pl.Pipeline(phase1)


if __name__ == "__main__":
    pipeline()
