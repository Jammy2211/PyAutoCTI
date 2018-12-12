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

def test_pipeline_parallel_3_species():

    pipeline_name = 'Parallel_x3s'
    data_name = '/int_x1_p3_e1'

    if os.path.exists("{}/data/{}".format(output_path, data_name)):
        shutil.rmtree("{}/data/{}".format(output_path, data_name))

    parallel_params = arctic_params.Species(trap_densities=(0.5, 1.5, 2.5), trap_lifetimes=(2.0, 5.0, 20.0),
                             well_notch_depth=1e-4, well_fill_alpha=1.0, well_fill_beta=0.5, well_fill_gamma=0.0)

    cti_params = arctic_params.ArcticParams(parallel=parallel_params)

    cti_settings = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=1, p_express=1, p_n_levels=2000,
                                  p_charge_injection_mode=True, p_readout_offset=0)

    tools.simulate_integration_quadrant(data_name, cti_params, cti_settings)
    ci_datas = tools.load_ci_datas(data_name)

    conf.instance.output_path = output_path

    if os.path.exists(output_path+pipeline_name):
        shutil.rmtree(output_path+pipeline_name)

    pipeline = make_parallel_x3s_pipeline(pipeline_name=pipeline_name)
    results = pipeline.run(ci_datas=ci_datas, cti_settings=cti_settings)

    for result in results:
        print(result)

def make_parallel_x3s_pipeline(pipeline_name):

    class ParallelPhase(ph.ParallelPhase):
        def pass_priors(self, previous_results):
            self.parallel.well_fill_alpha = 1.0
            self.parallel.well_fill_gamma = 0.0

    phase1 = ParallelPhase(optimizer_class=nl.MultiNest, parallel=arctic_params.Species,
                           columns=3, phase_name="{}/phase1".format(pipeline_name))

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.sampling_efficiency = 0.2

    class ParallelPhase(ph.ParallelPhase):
        def pass_priors(self, previous_results):
            prev_dens = previous_results[-1].constant.parallel.trap_densities[0]
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

    phase2 = ParallelPhase(optimizer_class=nl.MultiNest, parallel=arctic_params.Species,
                           phase_name="{}/phase2".format(pipeline_name))

    phase2.optimizer.n_live_points = 60
    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.sampling_efficiency = 0.2

    phase2h = ph.ParallelHyperOnlyPhase(optimizer_class=nl.MultiNest, phase_name="{}/phase2h".format(pipeline_name))

    class ParallelHyperFixedPhase(ph.ParallelHyperPhase):
        def pass_priors(self, previous_results):
            self.parallel = previous_results[-1].variable.parallel
            self.hyp_ci_regions = previous_results[-1].hyper.constant.hyp_ci_regions
            self.hyp_parallel_trails = previous_results[-1].hyper.constant.hyp_parallel_trails
            self.parallel.well_fill_alpha = 1.0
            self.parallel.well_fill_gamma = 0.0

    phase3 = ParallelHyperFixedPhase(optimizer_class=nl.MultiNest, phase_name="{}/phase3".format(pipeline_name))

    phase3.optimizer.n_live_points = 60
    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.sampling_efficiency = 0.2

    return pl.Pipeline(phase1, phase2, phase2h, phase3)

if __name__ == "__main__":
    test_pipeline_parallel_3_species()
