import autocti as ac
import autofit as af
from test_autocti.integration.tests import runner

test_type = "serial"
test_name = "x3_species__x1_image"
ci_data_type = "ci_uniform"
ci_data_model = "serial_x3"
resolution = "patch"
ci_normalizations = [10000.0, 84700.0]


clocker = ac.Clocker(serial_express=2)


def make_pipeline(name, phase_folders, non_linear_class=af.MultiNest):

    serial_ccd_volume = af.PriorModel(ac.CCDVolume)

    serial_ccd_volume.well_max_height = 8.47e4
    serial_ccd_volume.well_notch_depth = 1e-7

    phase1 = ac.PhaseCIImaging(
        phase_name="phase_1",
        phase_folders=phase_folders,
        non_linear_class=non_linear_class,
        serial_traps=[
            af.PriorModel(ac.Trap),
            af.PriorModel(ac.Trap),
            af.PriorModel(ac.Trap),
        ],
        serial_ccd_volume=serial_ccd_volume,
        columns=3,
    )

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.sampling_efficiency = 0.2

    phase1 = phase1.extend_with_hyper_noise_phases()

    phase2 = ac.PhaseCIImaging(
        phase_name="phase_2",
        phase_folders=phase_folders,
        serial_traps=phase1.result.model.serial_traps,
        serial_ccd_volume=phase1.result.model.serial_ccd_volume,
        hyper_noise_scalar_of_ci_regions=phase1.result.hyper_combined.instance.hyper_noise_scalar_of_ci_regions,
        hyper_noise_scalar_of_serial_trails=phase1.result.hyper_combined.instance.hyper_noise_scalar_of_serial_trails,
        non_linear_class=non_linear_class,
        columns=None,
    )

    # For the final CTI model, constant efficiency mode has a tendancy to sample parameter space too fast and infer an
    # inaccurate model. Thus, we turn it off for phase 2.

    phase2.optimizer.const_efficiency_mode = False
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    return ac.Pipeline(name, phase1)


if __name__ == "__main__":

    import sys

    runner.run(sys.modules[__name__], clocker=clocker)
