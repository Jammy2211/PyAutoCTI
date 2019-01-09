import os
import shutil

from autocti.pyarctic import arctic_params
from autocti.pyarctic import arctic_settings
from autofit import conf
from autofit.core import non_linear as nl

from autocti.pipeline import phase as ph
from autocti.pipeline import pipeline as pl
from test.integration import tools

output_path = '/gpfs/data/pdtw24/CTI/integration/'

shape = (36, 36)
ci_regions = [(1, 7, 1, 30), (17, 23, 1, 30)]
normalizations = [84700.0]
cti_geometry = tools.CIQuadGeometryIntegration()


def test_pipeline_serial_1_species():
    pipeline_name = 'Serial_x1s'
    data_name = '/int_x1_s1_e1'

    if os.path.exists("{}/data/{}".format(output_path, data_name)):
        shutil.rmtree("{}/data/{}".format(output_path, data_name))

    serial_params = arctic_params.Species(trap_densities=(1.0,), trap_lifetimes=(1.5,), well_notch_depth=1e-4,
                                          well_fill_alpha=1.0, well_fill_beta=0.5, well_fill_gamma=0.0)

    cti_params = arctic_params.ArcticParams(serial=serial_params)

    cti_settings = arctic_settings.setup(include_serial=True, s_well_depth=84700, s_niter=1, s_express=1,
                                         s_n_levels=2000,
                                         s_charge_injection_mode=False, s_readout_offset=0)

    tools.simulate_integration_quadrant(data_name, cti_params, cti_settings)
    ci_datas = tools.load_ci_datas(data_name)

    conf.instance.output_path = output_path

    if os.path.exists(output_path + pipeline_name):
        shutil.rmtree(output_path + pipeline_name)

    pipeline = make_serial_x1s_pipeline(pipeline_name=pipeline_name)
    results = pipeline.run(ci_datas=ci_datas, cti_settings=cti_settings)

    for result in results:
        print(result)


def make_serial_x1s_pipeline(pipeline_name):
    class SerialPhase(ph.SerialPhase):
        def pass_priors(self, previous_results):
            self.serial.well_fill_alpha = 1.0
            self.serial.well_fill_gamma = 0.0

    phase1 = SerialPhase(optimizer_class=nl.MultiNest, serial=arctic_params.Species,
                         phase_name="{}/phase1".format(pipeline_name))

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.sampling_efficiency = 0.2

    phase1h = ph.SerialHyperOnlyPhase(optimizer_class=nl.MultiNest, phase_name="{}/phase1h".format(pipeline_name))

    class SerialHyperFixedPhase(ph.SerialHyperPhase):
        def pass_priors(self, previous_results):
            self.serial = previous_results[-1].variable.serial
            self.hyp_ci_regions = previous_results[-1].hyper.constant.hyp_ci_regions
            self.hyp_serial_trails = previous_results[-1].hyper.constant.hyp_serial_trails
            self.serial.well_fill_alpha = 1.0
            self.serial.well_fill_gamma = 0.0

    phase2 = SerialHyperFixedPhase(optimizer_class=nl.MultiNest, phase_name="{}/phase2".format(pipeline_name))

    phase2.optimizer.n_live_points = 60
    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.sampling_efficiency = 0.2

    return pl.Pipeline(phase1, phase1h, phase2)


if __name__ == "__main__":
    test_pipeline_serial_1_species()
