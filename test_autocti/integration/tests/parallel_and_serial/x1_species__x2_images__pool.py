from multiprocessing import Pool

import autocti as ac
import autofit as af
from test_autocti.integration.tests import runner

test_type = "parallel_and_serial"
test_name = "x1_species__x2_images__pool"
ci_data_type = "ci_uniform"
ci_data_model = "parallel_x1__serial_x1"
resolution = "patch"
ci_normalizations = [10000.0, 84700.0]


clocker = ac.Clocker(parallel_express=2, serial_express=2)


def make_pipeline(name, phase_folders, non_linear_class=af.MultiNest):

    parallel_ccd_volume = af.PriorModel(ac.CCDVolume)

    parallel_ccd_volume.well_max_height = 8.47e4
    parallel_ccd_volume.well_notch_depth = 1e-7

    serial_ccd_volume = af.PriorModel(ac.CCDVolume)

    serial_ccd_volume.well_max_height = 8.47e4
    serial_ccd_volume.well_notch_depth = 1e-7

    phase1 = ac.PhaseCIImaging(
        phase_name="phase_1",
        phase_folders=phase_folders,
        non_linear_class=non_linear_class,
        parallel_traps=[af.PriorModel(ac.Trap)],
        parallel_ccd_volume=parallel_ccd_volume,
        serial_traps=[af.PriorModel(ac.Trap)],
        serial_ccd_volume=serial_ccd_volume,
    )
    phase1.optimizer.n_live_points = 60
    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.sampling_efficiency = 0.2

    return ac.Pipeline(name, phase1)


if __name__ == "__main__":

    import sys

    runner.run(sys.modules[__name__], clocker=clocker, pool=Pool(processes=2))
