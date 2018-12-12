import os
import shutil
from os import path

from autofit import conf
from autofit.core import non_linear as nl

from autocti.model import arctic_params
from autocti.model import arctic_settings
from autocti.pipeline import phase as ph
from autocti.pipeline import pipeline as pl
from test.integration import tools

directory = path.dirname(path.realpath(__file__))

output_path = '{}/output/'.format(directory)

shape = (36, 36)
ci_regions = [(1, 7, 1, 30), (17, 23, 1, 30)]
normalizations = [84700.0]
cti_geometry = tools.QuadGeometryIntegration()


def test_pipeline_parallel_serial_1_species():
    pipeline_name = 'Par_Ser_x1s'
    data_name = '/int_x1_ps1_e1'

    if os.path.exists("{}/data/{}".format(output_path, data_name)):
        shutil.rmtree("{}/data/{}".format(output_path, data_name))

    parallel_species = arctic_params.Species(trap_density=1.0, trap_lifetime=1.5)
    parallel_ccd = arctic_params.CCD(well_notch_depth=1e-4, well_fill_alpha=1.0, well_fill_beta=0.5,
                                     well_fill_gamma=0.0)

    serial_species = arctic_params.Species(trap_density=1.0, trap_lifetime=1.5)
    serial_ccd = arctic_params.CCD(well_notch_depth=1e-4, well_fill_alpha=1.0, well_fill_beta=0.5, well_fill_gamma=0.0)

    cti_params = arctic_params.ArcticParams(parallel_species=[parallel_species], serial_species=[serial_species],
                                            parallel_ccd=parallel_ccd, serial_ccd=serial_ccd)

    parallel_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=1,
                                                 n_levels=2000,
                                                 charge_injection_mode=True, readout_offset=0)
    serial_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=1,
                                               n_levels=2000,
                                               charge_injection_mode=False, readout_offset=0)

    cti_settings = arctic_settings.ArcticSettings(parallel=parallel_settings, serial=serial_settings)

    tools.simulate_integration_quadrant(data_name, cti_params, cti_settings)
    ci_datas = tools.load_ci_datas(data_name)

    conf.instance.output_path = output_path

    if os.path.exists(output_path + pipeline_name):
        shutil.rmtree(output_path + pipeline_name)

    pipeline = make_parallel_serial_x1s_pipeline(pipeline_name=pipeline_name)
    results = pipeline.run(ci_datas=ci_datas, cti_settings=cti_settings)

    for result in results:
        print(result)


def make_parallel_serial_x1s_pipeline(pipeline_name):
    class ParallelPhase(ph.ParallelPhase):
        def pass_priors(self, previous_results):
            self.parallel_species.well_fill_alpha = 1.0
            self.parallel_species.well_fill_gamma = 0.0

    phase1 = ParallelPhase(optimizer_class=nl.MultiNest, parallel_species=arctic_params.Species,
                           columns=3, phase_name="{}/phase1".format(pipeline_name))

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.sampling_efficiency = 0.2

    class SerialParallelFixedPhase(ph.ParallelSerialPhase):
        def pass_priors(self, previous_results):
            self.parallel = previous_results[-1].constant.parallel
            self.serial.well_fill_alpha = 1.0
            self.serial.well_fill_gamma = 0.0

    phase2 = SerialParallelFixedPhase(optimizer_class=nl.MultiNest, serial=arctic_params.Species,
                                      phase_name="{}/phase2".format(pipeline_name))

    phase2.optimizer.n_live_points = 60
    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.sampling_efficiency = 0.2

    class ParallelSerialPhase(ph.ParallelSerialPhase):
        def pass_priors(self, previous_results):
            self.parallel = previous_results[0].variable.parallel
            self.serial = previous_results[1].variable.serial
            self.parallel.well_fill_alpha = 1.0
            self.parallel.well_fill_gamma = 0.0
            self.serial.well_fill_alpha = 1.0
            self.serial.well_fill_gamma = 0.0

    phase3 = ParallelSerialPhase(optimizer_class=nl.MultiNest, phase_name="{}/phase3".format(pipeline_name))

    phase3.optimizer.n_live_points = 80
    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.sampling_efficiency = 0.2

    phase3h = ph.ParallelSerialHyperOnlyPhase(optimizer_class=nl.MultiNest,
                                              phase_name="{}/phase3h".format(pipeline_name))

    class ParallelSerialHyperFixedPhase(ph.ParallelSerialHyperPhase):
        def pass_priors(self, previous_results):
            self.parallel = previous_results[-1].variable.parallel
            self.serial = previous_results[-1].variable.serial
            self.hyp_ci_regions = previous_results[-1].hyper.constant.hyp_ci_regions
            self.hyp_parallel_trails = previous_results[-1].hyper.constant.hyp_parallel_trails
            self.hyp_serial_trails = previous_results[-1].hyper.constant.hyp_serial_trails
            self.hyp_parallel_serial_trails = previous_results[-1].hyper.constant.hyp_parallel_serial_trails
            self.parallel.well_fill_alpha = 1.0
            self.parallel.well_fill_gamma = 0.0
            self.serial.well_fill_alpha = 1.0
            self.serial.well_fill_gamma = 0.0

    phase4 = ParallelSerialHyperFixedPhase(optimizer_class=nl.MultiNest, phase_name="{}/phase4".format(pipeline_name))

    phase4.optimizer.n_live_points = 80
    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.sampling_efficiency = 0.2

    return pl.Pipeline(phase1, phase2, phase3, phase3h, phase4)


if __name__ == "__main__":
    test_pipeline_parallel_serial_1_species()
