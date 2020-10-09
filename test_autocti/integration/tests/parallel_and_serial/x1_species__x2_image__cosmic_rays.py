import autocti as ac
import autofit as af
from test_autocti.integration.tests import runner

test_type = "parallel_and_serial"
test_name = "x1_species__x2_image__cosmic_rays"
ci_data_type = "ci_uniform_cosmic_rays"
ci_data_model = "parallel_x1__serial_x1"
resolution = "patch"
ci_normalizations = [10000.0, 84700.0]

clocker = ac.Clocker(parallel_express=2, serial_express=2)


def make_pipeline(name, folders, search=af.DynestyStatic()):

    parallel_ccd = af.PriorModel(ac.CCD)

    parallel_ccd.full_well_depth = 8.47e4
    parallel_ccd.well_notch_depth = 1e-7

    serial_ccd = af.PriorModel(ac.CCD)

    serial_ccd.full_well_depth = 8.47e4
    serial_ccd.well_notch_depth = 1e-7

    phase1 = ac.PhaseCIImaging(
        name="phase_1",
        folders=folders,
        search=search,
        parallel_traps=[af.PriorModel(ac.TrapInstantCapture)],
        parallel_ccd=parallel_ccd,
        serial_traps=[af.PriorModel(ac.TrapInstantCapture)],
        serial_ccd=serial_ccd,
    )

    return ac.Pipeline(name, phase1)


if __name__ == "__main__":

    import sys

    runner.run(sys.modules[__name__], clocker=clocker, load_cosmic_ray_map=True)
