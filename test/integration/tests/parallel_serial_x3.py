from autofit import conf
from autofit.core import non_linear as nl
from autofit.core import model_mapper as mm
from autocti.pipeline import pipeline as pl
from autocti.pipeline import phase as ph
from autocti.pyarctic import arctic_params
from autocti.pyarctic import arctic_settings
from test.integration import tools

import shutil
import numpy as np
import os

output_path = '/gpfs/data/pdtw24/CTI/integration/'

shape = (36, 36)
ci_regions = [(1, 7, 1, 30), (17, 23, 1, 30)]
normalizations = [84700.0]
cti_geometry = tools.CIQuadGeometryIntegration()

def test_pipeline_parallel_serial_1_species():

    pipeline_name = 'Par_Ser_x3s'
    data_name = '/int_x1_ps3_e1'

    if os.path.exists("{}/data/{}".format(output_path, data_name)):
        shutil.rmtree("{}/data/{}".format(output_path, data_name))

    parallel_params = arctic_params.ParallelOneSpecies(trap_densities=(1.0,), trap_lifetimes=(1.5,), well_notch_depth=1e-4,
                                                 well_fill_alpha=1.0, well_fill_beta=0.5, well_fill_gamma=0.0)

    serial_params = arctic_params.SerialOneSpecies(trap_densities=(1.0,), trap_lifetimes=(1.5,), well_notch_depth=1e-4,
                                                 well_fill_alpha=1.0, well_fill_beta=0.5, well_fill_gamma=0.0)

    cti_params = arctic_params.ArcticParams(parallel=parallel_params, serial=serial_params)

    cti_settings = arctic_settings.setup(p=True, p_well_depth=84700, p_niter=1, p_express=1, p_n_levels=2000,
                                         p_charge_injection_mode=True, p_readout_offset=0,
                                         s=True, s_well_depth=84700, s_niter=1, s_express=1, s_n_levels=2000,
                                         s_charge_injection_mode=False, s_readout_offset=0)

    tools.simulate_integration_quadrant(data_name, cti_params, cti_settings)
    ci_datas = tools.load_ci_datas(data_name)

    conf.instance.output_path = output_path

    # if os.path.exists(output_path+pipeline_name):
    #     shutil.rmtree(output_path+pipeline_name)

    pipeline = make_parallel_serial_x1s_pipeline(pipeline_name=pipeline_name)
    results = pipeline.run(ci_datas=ci_datas, cti_settings=cti_settings)

    for result in results:
        print(result)

def make_parallel_serial_x1s_pipeline(pipeline_name):

    class ParallelPhase(ph.ParallelPhase):
        def pass_priors(self, previous_results):
            self.parallel.well_fill_alpha = 1.0
            self.parallel.well_fill_gamma = 0.0

    phase1 = ParallelPhase(optimizer_class=nl.MultiNest, parallel=arctic_params.ParallelOneSpecies,
                           columns=3, phase_name="{}/phase1".format(pipeline_name))

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.sampling_efficiency = 0.2

    class ParallelPhase(ph.ParallelPhase):
        def pass_priors(self, previous_results):
            prev_dens = previous_results[0].constant.parallel.trap_densities[0]
            self.parallel.trap_densities.trap_densities_0 = mm.UniformPrior(lower_limit=0.0, upper_limit=prev_dens*2.0)
            self.parallel.trap_densities.trap_densities_1 = mm.UniformPrior(lower_limit=0.0, upper_limit=prev_dens*2.0)
            self.parallel.trap_densities.trap_densities_2 = mm.UniformPrior(lower_limit=0.0, upper_limit=prev_dens*2.0)
            self.parallel.trap_lifetimes.trap_lifetimes_0 = mm.UniformPrior(lower_limit=0.0, upper_limit=30.0)
            self.parallel.trap_lifetimes.trap_lifetimes_1 = mm.UniformPrior(lower_limit=0.0, upper_limit=30.0)
            self.parallel.trap_lifetimes.trap_lifetimes_2 = mm.UniformPrior(lower_limit=0.0, upper_limit=30.0)
            self.parallel.well_notch_depth = previous_results[-1].variable.parallel.well_notch_depth
            self.parallel.well_fill_beta = previous_results[-1].variable.parallel.well_fill_beta
            self.parallel.well_fill_alpha = 1.0
            self.parallel.well_fill_gamma = 0.0

    phase2 = ParallelPhase(optimizer_class=nl.MultiNest, parallel=arctic_params.ParallelThreeSpecies,
                            columns=3, phase_name="{}/phase2".format(pipeline_name))

    phase2.optimizer.n_live_points = 60
    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.sampling_efficiency = 0.2

    class SerialParallelFixedPhase(ph.ParallelSerialPhase):
        def pass_priors(self, previous_results):
            self.parallel = previous_results[1].constant.parallel
            self.serial.well_fill_alpha = 1.0
            self.serial.well_fill_gamma = 0.0

    phase3 = SerialParallelFixedPhase(optimizer_class=nl.MultiNest, serial=arctic_params.SerialOneSpecies,
                         phase_name="{}/phase3".format(pipeline_name))

    phase3.optimizer.n_live_points = 60
    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.sampling_efficiency = 0.2

    class SerialParallelFixedPhase(ph.ParallelSerialPhase):
        def pass_priors(self, previous_results):
            self.parallel = previous_results[1].constant.parallel
            prev_dens = previous_results[-1].constant.serial.trap_densities[0]
            self.serial.trap_densities.trap_densities_0 = mm.UniformPrior(lower_limit=0.0, upper_limit=0.3)
            self.serial.trap_densities.trap_densities_1 = mm.UniformPrior(lower_limit=0.0, upper_limit=0.3)
            self.serial.trap_densities.trap_densities_2 = mm.UniformPrior(lower_limit=0.0, upper_limit=prev_dens*2.0)
            self.serial.trap_lifetimes.trap_lifetimes_0 = mm.UniformPrior(lower_limit=0.0, upper_limit=30.0)
            self.serial.trap_lifetimes.trap_lifetimes_1 = mm.UniformPrior(lower_limit=0.0, upper_limit=30.0)
            self.serial.trap_lifetimes.trap_lifetimes_2 = mm.UniformPrior(lower_limit=0.0, upper_limit=30.0)
            self.serial.well_notch_depth = previous_results[-1].variable.serial.well_notch_depth
            self.serial.well_fill_beta = previous_results[-1].variable.serial.well_fill_beta
            self.serial.well_fill_alpha = 1.0
            self.serial.well_fill_gamma = 0.0

    phase4 = SerialParallelFixedPhase(optimizer_class=nl.MultiNest, serial=arctic_params.SerialThreeSpecies,
                         phase_name="{}/phase4".format(pipeline_name))

    phase4.optimizer.n_live_points = 60
    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.sampling_efficiency = 0.2

    class ParallelSerialPhase(ph.ParallelSerialPhase):
        def pass_priors(self, previous_results):
            self.parallel = previous_results[1].variable.parallel
            self.serial = previous_results[3].variable.serial
            self.parallel.well_fill_alpha = 1.0
            self.parallel.well_fill_gamma = 0.0
            self.serial.well_fill_alpha = 1.0
            self.serial.well_fill_gamma = 0.0

    phase5 = ParallelSerialPhase(optimizer_class=nl.MultiNest, phase_name="{}/phase5".format(pipeline_name))

    phase5.optimizer.n_live_points = 80
    phase5.optimizer.const_efficiency_mode = True
    phase5.optimizer.sampling_efficiency = 0.2

    phase5h = ph.ParallelSerialHyperOnlyPhase(optimizer_class=nl.MultiNest,
                                              phase_name="{}/phase5h".format(pipeline_name))

    class ParallelSerialHyperFixedPhase(ph.ParallelSerialHyperPhase):
        def pass_priors(self, previous_results):
            self.serial = previous_results[-1].variable.serial
            self.parallel = previous_results[-1].variable.parallel
            self.hyp_ci_regions = previous_results[-1].hyper.constant.hyp_ci_regions
            self.hyp_parallel_trails = previous_results[-1].hyper.constant.hyp_parallel_trails
            self.hyp_serial_trails = previous_results[-1].hyper.constant.hyp_serial_trails
            self.hyp_parallel_serial_trails = previous_results[-1].hyper.constant.hyp_parallel_serial_trails
            self.serial.well_fill_alpha = 1.0
            self.serial.well_fill_gamma = 0.0
            self.parallel.well_fill_alpha = 1.0
            self.parallel.well_fill_gamma = 0.0

    phase6 = ParallelSerialHyperFixedPhase(optimizer_class=nl.MultiNest, phase_name="{}/phase6".format(pipeline_name))

    phase6.optimizer.n_live_points = 80
    phase6.optimizer.const_efficiency_mode = True
    phase6.optimizer.sampling_efficiency = 0.2

    return pl.Pipeline(phase1, phase2, phase3, phase4, phase5, phase5h, phase6)

if __name__ == "__main__":
    test_pipeline_parallel_serial_1_species()
