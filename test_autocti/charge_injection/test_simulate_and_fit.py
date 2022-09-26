import pytest

import autocti as ac


def test__simulate_and_fit_non_uniform():

    shape_native = (500, 100)

    parallel_overscan = ac.Region2D((480, 500, 5, 95))
    serial_prescan = ac.Region2D((0, 500, 0, 5))
    serial_overscan = ac.Region2D((0, 80, 95, 100))

    region_2d_list = [
        (0, 150, serial_prescan[3], serial_overscan[2]),
        (200, 450, serial_prescan[3], serial_overscan[2]),
    ]

    norm = 100

    column_sigma = 0.1 * norm
    row_slope = 0.0

    layout_ci = ac.Layout2DCI(
        shape_2d=shape_native,
        region_list=region_2d_list,
        parallel_overscan=parallel_overscan,
        serial_prescan=serial_prescan,
        serial_overscan=serial_overscan,
    )

    clocker_2d = ac.Clocker2D(
        parallel_express=2,
        parallel_roe=ac.ROEChargeInjection(),
        parallel_fast_mode=True,
        serial_express=2,
    )

    parallel_trap_0 = ac.TrapInstantCapture(density=1.0, release_timescale=5.0)
    parallel_trap_list = [parallel_trap_0]

    parallel_ccd = ac.CCDPhase(
        well_fill_power=0.58, well_notch_depth=0.0, full_well_depth=200000.0
    )

    serial_trap_0 = ac.TrapInstantCapture(density=1.0, release_timescale=5.0)
    serial_trap_list = [serial_trap_0]

    serial_ccd = ac.CCDPhase(
        well_fill_power=0.58, well_notch_depth=0.0, full_well_depth=200000.0
    )

    cti_2d = ac.CTI2D(
        parallel_trap_list=parallel_trap_list,
        parallel_ccd=parallel_ccd,
        serial_trap_list=serial_trap_list,
        serial_ccd=serial_ccd,
    )

    simulator = ac.SimulatorImagingCI(
        read_noise=None,
        pixel_scales=0.1,
        norm=norm,
        column_sigma=column_sigma,
        row_slope=row_slope,
        max_norm=200000.0,
    )

    imaging_ci = simulator.via_layout_from(
        clocker=clocker_2d, layout=layout_ci, cti=cti_2d
    )

    post_cti_data_2d = clocker_2d.add_cti(data=imaging_ci.pre_cti_data, cti=cti_2d)

    fit = ac.FitImagingCI(dataset=imaging_ci, post_cti_data=post_cti_data_2d)

    assert fit.chi_squared_map == pytest.approx(0.0, 1.0e-4)
    assert fit.figure_of_merit == pytest.approx(69182.327989468, 1.0e-4)


def test__simulate_and_extract_non_uniform_normalizations():

    shape_native = (500, 100)

    parallel_overscan = ac.Region2D((480, 500, 5, 95))
    serial_prescan = ac.Region2D((0, 500, 0, 5))
    serial_overscan = ac.Region2D((0, 80, 95, 100))

    region_2d_list = [
        (0, 150, serial_prescan[3], serial_overscan[2]),
        (200, 450, serial_prescan[3], serial_overscan[2]),
    ]

    norm = 100

    column_sigma = 0.1 * norm
    row_slope = 0.0

    layout_ci = ac.Layout2DCI(
        shape_2d=shape_native,
        region_list=region_2d_list,
        parallel_overscan=parallel_overscan,
        serial_prescan=serial_prescan,
        serial_overscan=serial_overscan,
    )

    clocker_2d = ac.Clocker2D(
        parallel_express=2,
        parallel_roe=ac.ROEChargeInjection(),
        parallel_fast_mode=True,
    )

    parallel_trap_0 = ac.TrapInstantCapture(density=1.0, release_timescale=5.0)
    parallel_trap_list = [parallel_trap_0]

    parallel_ccd = ac.CCDPhase(
        well_fill_power=0.58, well_notch_depth=0.0, full_well_depth=200000.0
    )

    cti_2d = ac.CTI2D(parallel_trap_list=parallel_trap_list, parallel_ccd=parallel_ccd)

    simulator = ac.SimulatorImagingCI(
        read_noise=None,
        pixel_scales=0.1,
        norm=norm,
        column_sigma=column_sigma,
        row_slope=row_slope,
        max_norm=200000.0,
        ci_seed=1,
    )

    imaging_ci = simulator.via_layout_from(
        clocker=clocker_2d, layout=layout_ci, cti=cti_2d
    )

    injection_norm_list = imaging_ci.layout.extract.parallel_fpr.median_list_from(
        array=imaging_ci.image, pixels=(120, 150)
    )

    assert injection_norm_list[0] == pytest.approx(116.23434, 1.0e-2)
    assert injection_norm_list[1] == pytest.approx(93.8824, 1.0e-2)
    assert injection_norm_list[2] == pytest.approx(94.7182, 1.0e-2)

    injection_norm_lists = imaging_ci.layout.extract.parallel_fpr.median_lists_of_individual_regions_from(
        array=imaging_ci.image, pixels=(120, 150)
    )

    assert injection_norm_lists[0][0] == pytest.approx(116.23434, 1.0e-2)
    assert injection_norm_lists[0][1] == pytest.approx(93.8824, 1.0e-2)
    assert injection_norm_lists[1][0] == pytest.approx(116.2434536, 1.0e-2)
