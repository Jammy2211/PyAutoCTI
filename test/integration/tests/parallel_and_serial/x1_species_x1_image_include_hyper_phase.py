import os

from autofit import conf
from autofit.mapper import prior_model
from autofit.optimize import non_linear as nl

from autocti.charge_injection import ci_data
from autocti.charge_injection import ci_pattern
from autocti.charge_injection import ci_hyper
from autocti.model import arctic_params
from autocti.model import arctic_settings
from autocti.pipeline import phase as ph
from autocti.pipeline import pipeline as pl
from test.simulation import simulation_util
from test.integration import integration_util

test_type = 'parallel_and_serial'
test_name = 'x1_species_x1_image_no_pool'

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = path + 'output/' + test_type
config_path = path + 'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)


def pipeline():


    integration_util.reset_paths(test_name=test_name, output_path=output_path)

    parallel_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=2, n_levels=2000,
                                                 charge_injection_mode=False, readout_offset=0)
    serial_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=1, n_levels=2000,
                                                 charge_injection_mode=False, readout_offset=0)
    cti_settings = arctic_settings.ArcticSettings(parallel=parallel_settings, serial=serial_settings)
    data = simulation_util.load_test_ci_data(data_name='ci_uniform_parallel_and_serial_x1_species',
                                             data_resolution='patch', normalization=84700.0)
    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(ci_datas=[data], cti_settings=cti_settings)

def make_pipeline(test_name):

    class ParallelSerialPhase(ph.ParallelSerialPhase):

        def pass_priors(self, previous_results):

            self.parallel_ccd.well_fill_alpha = 1.0
            self.parallel_ccd.well_fill_gamma = 0.0
            self.serial_ccd.well_fill_alpha = 1.0
            self.serial_ccd.well_fill_gamma = 0.0

    phase1 = ParallelSerialPhase(optimizer_class=nl.MultiNest,
                                 parallel_species=[prior_model.PriorModel(arctic_params.Species)],
                                 parallel_ccd=arctic_params.CCD,
                                 serial_species=[prior_model.PriorModel(arctic_params.Species)],
                                 serial_ccd=arctic_params.CCD, phase_name="{}/phase1".format(test_name))

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.sampling_efficiency = 0.2

    class ParallelSerialHyperModelFixedPhase(ph.ParallelSerialHyperPhase):

        def pass_priors(self, previous_results):

            self.parallel_species = previous_results[0].constant.parallel_species
            self.parallel_ccd = previous_results[0].constant.parallel_ccd
            self.serial_species = previous_results[0].constant.serial_species
            self.serial_ccd = previous_results[0].constant.serial_ccd

    phase2 = ParallelSerialHyperModelFixedPhase(parallel_species=[prior_model.PriorModel(arctic_params.Species)],
                                                parallel_ccd=arctic_params.CCD,
                                                serial_species=[prior_model.PriorModel(arctic_params.Species)],
                                                serial_ccd=arctic_params.CCD,
                                                optimizer_class=nl.MultiNest,
                                                phase_name="{}/phase2".format(test_name))

    class SerialHyperFixedPhase(ph.SerialHyperPhase):

        def pass_priors(self, previous_results):

            self.hyper_noise_scalars = previous_results[1].constant.hyper_noise_scalars
            self.parallel_species = previous_results[0].variable.parallel_species
            self.parallel_ccd = previous_results[0].variable.parallel_ccd
            self.parallel_ccd.well_fill_alpha = 1.0
            self.parallel_ccd.well_fill_gamma = 0.0
            self.serial_species = previous_results[0].variable.serial_species
            self.serial_ccd = previous_results[0].variable.serial_ccd
            self.serial_ccd.well_fill_alpha = 1.0
            self.serial_ccd.well_fill_gamma = 0.0

    phase3 = SerialHyperFixedPhase(optimizer_class=nl.MultiNest, rows=None, phase_name="{}/phase3".format(test_name))

    # For the final CTI model, constant efficiency mode has a tendancy to sample parameter space too fast and infer an
    # inaccurate model. Thus, we turn it off for phase 2.

    phase3.optimizer.const_efficiency_mode = False
    phase3.optimizer.n_live_points = 50
    phase3.optimizer.sampling_efficiency = 0.3

    return pl.Pipeline(phase1, phase2, phase3)


if __name__ == "__main__":
    pipeline()
