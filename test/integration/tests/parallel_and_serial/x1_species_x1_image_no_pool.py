import os

from autofit import conf
from autofit.mapper import prior_model
from autofit.optimize import non_linear as nl

from autocti.charge_injection import ci_data, ci_pattern
from autocti.model import arctic_params
from autocti.model import arctic_settings
from autocti.pipeline import phase as ph
from autocti.pipeline import pipeline as pl
from test.integration import tools

shape = (36, 36)
ci_regions = [(1, 7, 1, 30), (17, 23, 1, 30)]
normalizations = [84700.0]
frame_geometry = tools.CIQuadGeometryIntegration()

test_type = 'parallel_and_serial'
test_name = 'one_species_x1_image_no_pool'

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = path + 'output/' + test_type
config_path = path + 'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)


def pipeline():

    tools.reset_paths(test_name=test_name, output_path=output_path)

    parallel_species = arctic_params.Species(trap_density=0.1, trap_lifetime=1.5)
    parallel_ccd = arctic_params.CCD(well_notch_depth=0.01, well_fill_alpha=0.2,
                                     well_fill_beta=0.8, well_fill_gamma=2.0)

    serial_species = arctic_params.Species(trap_density=0.1, trap_lifetime=1.5)
    serial_ccd = arctic_params.CCD(well_notch_depth=0.01, well_fill_alpha=0.2,
                                     well_fill_beta=0.8, well_fill_gamma=2.0)

    cti_params = arctic_params.ArcticParams(parallel_ccd=parallel_ccd, parallel_species=[parallel_species],
                                            serial_ccd=serial_ccd, serial_species=[serial_species])

    parallel_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=1, n_levels=2000,
                                                 charge_injection_mode=True, readout_offset=0)
    serial_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=1, n_levels=2000,
                                                 charge_injection_mode=False, readout_offset=0)
    cti_settings = arctic_settings.ArcticSettings(parallel=parallel_settings, serial=serial_settings)

    tools.simulate_integration_quadrant(test_name=test_name, normalizations=normalizations, cti_params=cti_params,
                                        cti_settings=cti_settings)

    pattern = ci_pattern.CIPatternUniform(normalization=normalizations[0], regions=ci_regions)

    data = ci_data.load_ci_data_from_fits(frame_geometry=frame_geometry, ci_pattern=pattern,
                                          ci_image_path=path + '/data/' + test_name + '/ci_image_0.fits',
                                          ci_noise_map_from_single_value=1.0)

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

    return pl.Pipeline(phase1)


if __name__ == "__main__":
    pipeline()
