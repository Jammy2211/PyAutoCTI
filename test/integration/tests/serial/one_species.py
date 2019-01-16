import os

from autofit import conf
from autofit.core import non_linear as nl
from autofit.core import model_mapper as mm
from autocti.model import arctic_params
from autocti.model import arctic_settings

from autocti.pipeline import phase as ph
from autocti.pipeline import pipeline as pl
from test.integration import tools

shape = (36, 36)
ci_regions = [(1, 7, 1, 30), (17, 23, 1, 30)]
normalizations = [84700.0]
frame_geometry = tools.CIQuadGeometryIntegration()

test_type = 'serial'
test_name = 'one_species'

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = path+'output/'+test_type
config_path = path+'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)

def pipeline():

    serial_species = arctic_params.Species(trap_density=0.1, trap_lifetime=1.5)
    serial_ccd = arctic_params.CCD(well_notch_depth=0.01, well_fill_alpha=0.2,
                                     well_fill_beta=0.8, well_fill_gamma=2.0)
    cti_params = arctic_params.ArcticParams(serial_ccd=serial_ccd, serial_species=[serial_species])

    serial_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=1, n_levels=2000,
                                                 charge_injection_mode=False, readout_offset=0)
    cti_settings = arctic_settings.ArcticSettings(serial=serial_settings)

    tools.reset_paths(test_name=test_name, output_path=output_path)
    tools.simulate_integration_quadrant(test_name=test_name, cti_params=cti_params, cti_settings=cti_settings)
    ci_datas = tools.load_ci_datas(test_name=test_name)

    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(ci_datas=[ci_datas], cti_settings=cti_settings)


def make_pipeline(test_name):

    class SerialPhase(ph.SerialPhase):

        def pass_priors(self, previous_results):

            self.serial_ccd.well_fill_alpha = 1.0
            self.serial_ccd.well_fill_gamma = 0.0

    phase1 = SerialPhase(optimizer_class=nl.MultiNest, serial_species=[mm.PriorModel(arctic_params.Species)],
                           serial_ccd=arctic_params.CCD,
                           columns=3, phase_name="{}/phase1".format(test_name))

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.sampling_efficiency = 0.2

    return pl.Pipeline(phase1)


if __name__ == "__main__":
    pipeline()
