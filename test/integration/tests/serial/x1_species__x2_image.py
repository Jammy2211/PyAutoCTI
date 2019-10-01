import autofit as af
import autocti as ac
from test import runner

test_type = "serial"
test_name = "x1_species__x2_image"
ci_data_type = "ci__uniform"
ci_data_model = "serial_x1"
ci_data_resolution = "patch"
ci_normalizations = [10000.0, 84700.0]

serial_settings = ac.Settings(
    well_depth=84700,
    niter=1,
    express=2,
    n_levels=2000,
    charge_injection_mode=False,
    readout_offset=0,
)
cti_settings = ac.ArcticSettings(serial=serial_settings)


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    class PhaseCI(ac.PhaseCI):
        def customize_priors(self, results):
            self.serial_ccd_volume.well_fill_alpha = 1.0
            self.serial_ccd_volume.well_fill_gamma = 0.0

    phase1 = PhaseCI(
        phase_name="phase_1",
        phase_folders=phase_folders,
        optimizer_class=optimizer_class,
        serial_species=[af.PriorModel(ac.Species)],
        serial_ccd_volume=ac.CCDVolume,
        columns=40,
    )

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.sampling_efficiency = 0.2

    phase1 = phase1.extend_with_hyper_noise_phases()

    phase2 = ac.PhaseCI(
        phase_name="phase_2",
        phase_folders=phase_folders,
        serial_species=phase1.result.variable.serial_species,
        serial_ccd_volume=phase1.result.variable.serial_ccd_volume,
        hyper_noise_scalar_of_ci_regions=phase1.result.hyper_combined.constant.hyper_noise_scalar_of_ci_regions,
        hyper_noise_scalar_of_serial_trails=phase1.result.hyper_combined.constant.hyper_noise_scalar_of_serial_trails,
        optimizer_class=optimizer_class,
        columns=None,
    )

    # For the final CTI model, constant efficiency mode has a tendancy to sample parameter space too fast and infer an
    # inaccurate model. Thus, we turn it off for phase 2.

    phase2.optimizer.const_efficiency_mode = False
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    return ac.Pipeline(name, phase1, phase2)


if __name__ == "__main__":

    import sys

    runner.run(sys.modules[__name__], cti_settings=cti_settings)
