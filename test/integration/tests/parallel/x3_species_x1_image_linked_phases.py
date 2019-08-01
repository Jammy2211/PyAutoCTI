import os

import autofit as af
from autocti.model import arctic_params
from autocti.model import arctic_settings
from autocti.pipeline import phase as ph
from autocti.pipeline import pipeline as pl
from test.integration import integration_util
from test.simulation import simulation_util

test_type = "parallel"
test_name = "x3_species_x1_image_linked_phases"

test_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + "output/"
config_path = test_path + "config"
af.conf.instance = af.conf.Config(config_path=config_path, output_path=output_path)


def pipeline():
    integration_util.reset_paths(test_name=test_name, output_path=output_path)

    parallel_settings = arctic_settings.Settings(
        well_depth=84700,
        niter=1,
        express=2,
        n_levels=2000,
        charge_injection_mode=False,
        readout_offset=0,
    )
    cti_settings = arctic_settings.ArcticSettings(parallel=parallel_settings)
    data = simulation_util.load_test_ci_data(
        ci_data_type="ci_uniform",
        ci_data_model="parallel_x3",
        ci_data_resolution="patch",
        normalization=84700.0,
    )
    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(ci_datas=[data], cti_settings=cti_settings)


def make_pipeline(test_name):
    class ParallelPhase(ph.ParallelPhase):
        def pass_priors(self, results):
            self.parallel_ccd.well_fill_alpha = 1.0
            self.parallel_ccd.well_fill_gamma = 0.0

    phase1 = ParallelPhase(
        phase_name="phase_1",
        phase_folders=[test_type, test_name],
        optimizer_class=af.MultiNest,
        parallel_species=[af.PriorModel(arctic_params.Species)],
        parallel_ccd=arctic_params.CCD,
        columns=None,
    )

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.sampling_efficiency = 0.2

    class ParallelPhase(ph.ParallelPhase):
        def pass_priors(self, results):

            previous_total_density = (
                results[-1].constant.parallel_species[0].trap_density
            )

            self.parallel_species[0].trap_density = af.UniformPrior(
                lower_limit=0.0, upper_limit=previous_total_density
            )
            self.parallel_species[1].trap_density = af.UniformPrior(
                lower_limit=0.0, upper_limit=previous_total_density
            )
            self.parallel_species[2].trap_density = af.UniformPrior(
                lower_limit=0.0, upper_limit=previous_total_density
            )
            self.parallel_species[0].trap_lifetime = af.UniformPrior(
                lower_limit=0.0, upper_limit=30.0
            )
            self.parallel_species[1].trap_lifetime = af.UniformPrior(
                lower_limit=0.0, upper_limit=30.0
            )
            self.parallel_species[2].trap_lifetime = af.UniformPrior(
                lower_limit=0.0, upper_limit=30.0
            )

            self.parallel_ccd.well_notch_depth = results.from_phase(
                "phase_1"
            ).variable.parallel_ccd.well_notch_depth
            self.parallel_ccd.well_fill_beta = results.from_phase(
                "phase_1"
            ).variable.parallel_ccd.well_fill_beta

    phase2 = ParallelPhase(
        phase_name="phase_2",
        phase_folders=[test_type, test_name],
        optimizer_class=af.MultiNest,
        parallel_species=[
            af.PriorModel(arctic_params.Species),
            af.PriorModel(arctic_params.Species),
            af.PriorModel(arctic_params.Species),
        ],
        parallel_ccd=arctic_params.CCD,
        columns=3,
    )

    phase2.optimizer.n_live_points = 60
    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.sampling_efficiency = 0.2

    return pl.Pipeline(test_type, phase1, phase2)


if __name__ == "__main__":
    pipeline()
