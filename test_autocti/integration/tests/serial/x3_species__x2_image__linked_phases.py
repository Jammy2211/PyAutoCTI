import autocti as ac
import autofit as af
from test_autocti.integration.tests import runner

test_type = "serial"
test_name = "x3_species__x2_image__linked_phases"
ci_data_type = "ci_uniform"
ci_data_model = "serial_x3"
resolution = "patch"
ci_normalizations = [84700.0]


clocker = ac.Clocker(serial_express=2)


def make_pipeline(name, folders, search=af.DynestyStatic()):

    serial_ccd_volume = af.PriorModel(ac.CCDVolume)

    serial_ccd_volume.well_max_height = 8.47e4
    serial_ccd_volume.well_notch_depth = 1e-7

    phase1 = ac.PhaseCIImaging(
        phase_name="phase_1",
        folders=folders,
        search=search,
        serial_traps=[af.PriorModel(ac.Trap)],
        serial_ccd_volume=serial_ccd_volume,
        columns=None,
    )

    phase1.search.n_live_points = 60
    phase1.search.const_efficiency_mode = True
    phase1.search.facc = 0.2

    class PhaseCIImaging(ac.PhaseCIImaging):
        def customize_priors(self, results):

            previous_total_density = results[-1].instance.serial_traps[0].trap_density

            self.serial_traps[0].trap_density = af.UniformPrior(
                lower_limit=0.0, upper_limit=previous_total_density
            )
            self.serial_traps[1].trap_density = af.UniformPrior(
                lower_limit=0.0, upper_limit=previous_total_density
            )
            self.serial_traps[2].trap_density = af.UniformPrior(
                lower_limit=0.0, upper_limit=previous_total_density
            )
            self.serial_traps[0].trap_lifetime = af.UniformPrior(
                lower_limit=0.0, upper_limit=30.0
            )
            self.serial_traps[1].trap_lifetime = af.UniformPrior(
                lower_limit=0.0, upper_limit=30.0
            )
            self.serial_traps[2].trap_lifetime = af.UniformPrior(
                lower_limit=0.0, upper_limit=30.0
            )

    phase2 = PhaseCIImaging(
        phase_name="phase_2",
        folders=folders,
        search=search,
        serial_traps=[
            af.PriorModel(ac.Trap),
            af.PriorModel(ac.Trap),
            af.PriorModel(ac.Trap),
        ],
        serial_ccd_volume=phase1.result.model.serial_ccd_volume,
    )

    phase2.search.n_live_points = 60
    phase2.search.const_efficiency_mode = True
    phase2.search.facc = 0.2

    phase2 = phase1.extend_with_hyper_noise_phases()

    phase3 = ac.PhaseCIImaging(
        phase_name="phase_3",
        folders=folders,
        serial_traps=phase1.result.model.serial_traps,
        serial_ccd_volume=phase1.result.model.serial_ccd_volume,
        hyper_noise_scalar_of_ci_regions=phase1.result.hyper_combined.instance.hyper_noise_scalar_of_ci_regions,
        hyper_noise_scalar_of_serial_trails=phase1.result.hyper_combined.instance.hyper_noise_scalar_of_serial_trails,
        search=search,
        columns=None,
    )

    # For the final CTI model, constant efficiency mode has a tendancy to sample parameter space too fast and infer an
    # inaccurate model. Thus, we turn it off for phase 2.

    phase3.search.const_efficiency_mode = False
    phase3.search.n_live_points = 50
    phase3.search.facc = 0.3

    return ac.Pipeline(name, phase1, phase2, phase3)


if __name__ == "__main__":

    import sys

    runner.run(sys.modules[__name__], clocker=clocker)
