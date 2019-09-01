import autofit as af
import autocti as ac
from test.integration.tests import runner

test_type = "parallel_x1__serial_x1"
test_name = "x1_species__x2_image__hyper"
ci_data_type = "ci_uniform"
ci_data_model = "parallel_x1__serial_x1"
ci_data_resolution = "patch"
ci_normalizations = [84700.0]



parallel_settings = ac.Settings(
    well_depth=84700,
    niter=1,
    express=2,
    n_levels=2000,
    charge_injection_mode=False,
    readout_offset=0,
)
serial_settings = ac.Settings(
    well_depth=84700,
    niter=1,
    express=1,
    n_levels=2000,
    charge_injection_mode=False,
    readout_offset=0,
)
cti_settings = ac.ArcticSettings(parallel=parallel_settings, serial=serial_settings)


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    class ParallelSerialPhase(ac.ParallelSerialPhase):
        def pass_priors(self, results):

            self.parallel_ccd.well_fill_alpha = 1.0
            self.parallel_ccd.well_fill_gamma = 0.0
            self.serial_ccd.well_fill_alpha = 1.0
            self.serial_ccd.well_fill_gamma = 0.0

    phase1 = ParallelSerialPhase(
        phase_name="phase_1",
        phase_folders=phase_folders,
        optimizer_class=optimizer_class,
        parallel_species=[af.PriorModel(ac.Species)],
        parallel_ccd=ac.CCDVolume,
        serial_species=[af.PriorModel(ac.Species)],
        serial_ccd=ac.CCDVolume,
    )

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.sampling_efficiency = 0.2

    class ParallelSerialHyperModelFixedPhase(ac.ParallelSerialHyperPhase):
        def pass_priors(self, results):

            self.parallel_species = results.from_phase(
                "phase_1"
            ).constant.parallel_species
            self.parallel_ccd = results.from_phase("phase_1").constant.parallel_ccd
            self.serial_species = results.from_phase("phase_1").constant.serial_species
            self.serial_ccd = results.from_phase("phase_1").constant.serial_ccd

    phase2 = ParallelSerialHyperModelFixedPhase(
        phase_name="phase_2",
        phase_folders=phase_folders,
        parallel_species=[af.PriorModel(ac.Species)],
        parallel_ccd=ac.CCDVolume,
        serial_species=[af.PriorModel(ac.Species)],
        serial_ccd=ac.CCDVolume,
        optimizer_class=optimizer_class,
    )

    class SerialHyperFixedPhase(ac.SerialHyperPhase):
        def pass_priors(self, results):

            self.hyper_noise_scalars = results.from_phase(
                "phase_2"
            ).constant.hyper_noise_scalars
            self.parallel_species = results.from_phase(
                "phase_1"
            ).variable.parallel_species
            self.parallel_ccd = results.from_phase("phase_1").variable.parallel_ccd
            self.parallel_ccd.well_fill_alpha = 1.0
            self.parallel_ccd.well_fill_gamma = 0.0
            self.serial_species = results.from_phase("phase_1").variable.serial_species
            self.serial_ccd = results.from_phase("phase_1").variable.serial_ccd
            self.serial_ccd.well_fill_alpha = 1.0
            self.serial_ccd.well_fill_gamma = 0.0

    phase3 = SerialHyperFixedPhase(
        phase_name="phase_3",
        phase_folders=phase_folders,
        optimizer_class=optimizer_class,
        rows=None,
    )

    # For the final CTI model, constant efficiency mode has a tendancy to sample parameter space too fast and infer an
    # inaccurate model. Thus, we turn it off for phase 2.

    phase3.optimizer.const_efficiency_mode = False
    phase3.optimizer.n_live_points = 50
    phase3.optimizer.sampling_efficiency = 0.3

    return ac.Pipeline(name, phase1, phase2, phase3)


if __name__ == "__main__":

    import sys

    runner.run(sys.modules[__name__], cti_settings=cti_settings)
