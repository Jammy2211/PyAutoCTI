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
output_path = test_path + 'output/' + test_type
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

        def pass_priors(self, previous_results):

            self.serial_ccd.well_fill_alpha = 1.0
            self.serial_ccd.well_fill_gamma = 0.0

    phase1 = SerialPhase(phase_name='phase1', phase_folders=[test_name],
                         optimizer_class=nl.MultiNest,
                         serial_species=[prior_model.PriorModel(arctic_params.Species)],
                         serial_ccd=arctic_params.CCD)

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.sampling_efficiency = 0.2

    return pl.Pipeline(phase1)


if __name__ == "__main__":
    pipeline()
