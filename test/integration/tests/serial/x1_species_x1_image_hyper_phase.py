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

test_type = 'serial'
test_name = 'x1_species_x1_image_no_pool'

test_path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + 'output/'
config_path = test_path + 'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)


def pipeline():

    integration_util.reset_paths(test_name=test_name, output_path=output_path)

    serial_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=2, n_levels=2000,
                                                 charge_injection_mode=False, readout_offset=0)
    cti_settings = arctic_settings.ArcticSettings(serial=serial_settings)
    data = simulation_util.load_test_ci_data(ci_data_type='ci_uniform', ci_data_model='serial_x1_species',
                                             ci_data_resolution='patch',normalization=84700.0)
    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(ci_datas=[data], cti_settings=cti_settings)


def make_pipeline(test_name):

    class SerialPhase(ph.SerialPhase):

        def pass_priors(self, results):

            self.serial_ccd.well_fill_alpha = 1.0
            self.serial_ccd.well_fill_gamma = 0.0

    phase1 = SerialPhase(phase_name='phase_1', phase_folders=[test_type, test_name],
                         optimizer_class=nl.MultiNest,
                         serial_species=[prior_model.PriorModel(arctic_params.Species)], rows=(0,4),
                         serial_ccd=arctic_params.CCD)

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.sampling_efficiency = 0.2

    class SerialHyperModelFixedPhase(ph.SerialHyperPhase):

        def pass_priors(self, results):

            self.serial_species = results.from_phase('phase_1').constant.serial_species
            self.serial_ccd = results.from_phase('phase_1').constant.serial_ccd

    phase2 = SerialHyperModelFixedPhase(phase_name='phase_2', phase_folders=[test_type, test_name],
                                        serial_species=[prior_model.PriorModel(arctic_params.Species)],
                                        serial_ccd=arctic_params.CCD,
                                        optimizer_class=nl.MultiNest, rows=None)

    class SerialHyperFixedPhase(ph.SerialHyperPhase):

        def pass_priors(self, results):

            self.hyper_noise_scalars = results.from_phase('phase_2').constant.hyper_noise_scalars
            self.serial_species = results.from_phase('phase_1').variable.serial_species
            self.serial_ccd = results.from_phase('phase_1').variable.serial_ccd
            self.serial_ccd.well_fill_alpha = 1.0
            self.serial_ccd.well_fill_gamma = 0.0

    phase3 = SerialHyperFixedPhase(phase_name='phase_3', phase_folders=[test_type, test_name],
                                   optimizer_class=nl.MultiNest, rows=None)

    # For the final CTI model, constant efficiency mode has a tendancy to sample parameter space too fast and infer an
    # inaccurate model. Thus, we turn it off for phase 2.

    phase3.optimizer.const_efficiency_mode = False
    phase3.optimizer.n_live_points = 50
    phase3.optimizer.sampling_efficiency = 0.3

    return pl.Pipeline(test_type, phase1, phase2, phase3)


if __name__ == "__main__":
    pipeline()
