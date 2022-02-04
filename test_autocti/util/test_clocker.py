import os

import pytest
import numpy as np
from arcticpy.src import cti
import autocti as ac
from autocti import exc

path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))


def test__clocker_1d__data_mapped_to_2d_and_then_1d():

    arr_1d = ac.Array1D.manual_native(
        array=[1.0, 2.0, 3.0, 4.0],
        pixel_scales=1.0,
        header=ac.Header(
            header_sci_obj=None, header_hdu_obj=None, readout_offsets=(3,)
        ),
    ).native

    arr_2d = ac.Array2D.manual(
        array=[[1.0], [2.0], [3.0], [4.0]],
        pixel_scales=1.0,
        header=ac.Header(
            header_sci_obj=None, header_hdu_obj=None, readout_offsets=(3,)
        ),
    ).native

    roe = ac.ROE(
        dwell_times=[1.0],
        empty_traps_between_columns=True,
        empty_traps_for_first_transfers=False,
        force_release_away_from_readout=True,
        use_integer_express_matrix=False,
    )
    ccd_phase = ac.CCDPhase(
        full_well_depth=1e3, well_notch_depth=0.0, well_fill_power=1.0
    )
    ccd = ac.CCD(phases=[ccd_phase], fraction_of_traps_per_phase=[1.0])
    traps = [ac.TrapInstantCapture(10.0, -1.0 / np.log(0.5))]

    image_via_arctic = cti.add_cti(
        image=arr_2d,
        parallel_traps=traps,
        parallel_ccd=ccd,
        parallel_roe=roe,
        parallel_express=3,
        parallel_offset=3,
    )

    clocker_1d = ac.Clocker1D(express=3, roe=roe)

    image_via_clocker = clocker_1d.add_cti(data=arr_1d, trap_list=traps, ccd=ccd_phase)

    assert image_via_arctic.flatten() == pytest.approx(image_via_clocker, 1.0e-4)


def test__clocker_2d__array_with_offset_through_arctic():

    arr = ac.Array2D.manual(
        array=[[1.0, 2.0], [3.0, 4.0]],
        pixel_scales=1.0,
        header=ac.Header(
            header_sci_obj=None, header_hdu_obj=None, readout_offsets=(3, 5)
        ),
    ).native

    roe = ac.ROE(
        dwell_times=[1.0],
        empty_traps_between_columns=True,
        empty_traps_for_first_transfers=False,
        force_release_away_from_readout=True,
        use_integer_express_matrix=False,
    )
    ccd_phase = ac.CCDPhase(
        full_well_depth=1e3, well_notch_depth=0.0, well_fill_power=1.0
    )
    ccd = ac.CCD(phases=[ccd_phase], fraction_of_traps_per_phase=[1.0])
    traps = [ac.TrapInstantCapture(10.0, -1.0 / np.log(0.5))]

    image_via_arctic = cti.add_cti(
        image=arr,
        parallel_traps=traps,
        parallel_ccd=ccd,
        parallel_roe=roe,
        parallel_express=3,
        parallel_offset=3,
    )

    clocker = ac.Clocker2D(parallel_express=3, parallel_roe=roe)

    image_via_clocker = clocker.add_cti(
        data=arr, parallel_trap_list=traps, parallel_ccd=ccd_phase
    )

    assert image_via_arctic == pytest.approx(image_via_clocker, 1.0e-4)

    image_via_arctic = cti.add_cti(
        image=arr,
        serial_traps=traps,
        serial_ccd=ccd,
        serial_roe=roe,
        serial_express=2,
        serial_offset=5,
    )

    clocker = ac.Clocker2D(serial_express=2, serial_roe=roe)

    image_via_clocker = clocker.add_cti(
        data=arr, serial_trap_list=traps, serial_ccd=ccd_phase
    )

    assert image_via_arctic == pytest.approx(image_via_clocker, 1.0e-4)


def test__clocker_2d__add_cti_with_poisson_trap_densities():

    arr = np.array(
        (
            [
                [1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    )

    arr = ac.Array2D.manual(array=arr, pixel_scales=1.0).native

    roe = ac.ROE(
        dwell_times=[1.0],
        empty_traps_between_columns=True,
        empty_traps_for_first_transfers=False,
        force_release_away_from_readout=True,
        use_integer_express_matrix=False,
    )
    ccd = ac.CCDPhase(full_well_depth=1e3, well_notch_depth=0.0, well_fill_power=1.0)

    trap_list = [
        ac.TrapInstantCapture(density=10.0, release_timescale=-1.0 / np.log(0.5))
    ]

    clocker = ac.Clocker2D(
        parallel_poisson_traps=True,
        poisson_seed=1,
        parallel_express=3,
        parallel_roe=roe,
        serial_express=3,
        serial_roe=roe,
    )

    image_via_clocker = clocker.add_cti(
        data=arr,
        parallel_trap_list=trap_list,
        parallel_ccd=ccd,
        serial_trap_list=trap_list,
        serial_ccd=ccd,
    )

    assert image_via_clocker[0, 0] == pytest.approx(0.980298, 1.0e-4)
    assert image_via_clocker[0, 1] == pytest.approx(0.9901042, 1.0e-4)
    assert image_via_clocker[0, 2] == pytest.approx(0.9901981, 1.0e-4)
    assert (image_via_clocker[:, 3] > 0.0).all()


def test__clocker_2d__add_cti_fast_parallel():

    arr = np.array(
        (
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
            ]
        )
    )

    arr = ac.Array2D.manual(array=arr, pixel_scales=1.0).native

    ccd = ac.CCDPhase(full_well_depth=1e3, well_notch_depth=0.0, well_fill_power=1.0)

    trap_list = [
        ac.TrapInstantCapture(density=10.0, release_timescale=-1.0 / np.log(0.5))
    ]

    clocker = ac.Clocker2D()

    image_via_clocker = clocker.add_cti(
        data=arr, parallel_trap_list=trap_list, parallel_ccd=ccd
    )

    clocker = ac.Clocker2D(parallel_fast_pixels=(1, 3))

    image_via_clocker_fast = clocker.add_cti(
        data=arr, parallel_trap_list=trap_list, parallel_ccd=ccd
    )

    assert image_via_clocker == pytest.approx(image_via_clocker_fast, 1.0e-6)


def test__clocker_2d__add_cti_fast_parallel__raises_exception_if_nonzero_outside_tuple():

    ccd = ac.CCDPhase(full_well_depth=1e3, well_notch_depth=0.0, well_fill_power=1.0)

    trap_list = [
        ac.TrapInstantCapture(density=10.0, release_timescale=-1.0 / np.log(0.5))
    ]

    clocker = ac.Clocker2D(parallel_fast_pixels=(1, 3))

    arr = np.array(
        (
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
            ]
        )
    )

    arr = ac.Array2D.manual(array=arr, pixel_scales=1.0).native

    with pytest.raises(exc.ClockerException):

        clocker.add_cti(data=arr, parallel_trap_list=trap_list, parallel_ccd=ccd)

    arr = np.array(
        (
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
            ]
        )
    )

    arr = ac.Array2D.manual(array=arr, pixel_scales=1.0).native

    with pytest.raises(exc.ClockerException):

        clocker.add_cti(data=arr, parallel_trap_list=trap_list, parallel_ccd=ccd)

    arr = np.array(
        (
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 2.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
            ]
        )
    )

    arr = ac.Array2D.manual(array=arr, pixel_scales=1.0).native

    with pytest.raises(exc.ClockerException):

        clocker.add_cti(data=arr, parallel_trap_list=trap_list, parallel_ccd=ccd)


def test__clocker_2d__raises_exception_if_no_traps_or_ccd_passed():

    arr = ac.Array2D.manual(
        array=[[1.0, 2.0], [3.0, 4.0]],
        pixel_scales=1.0,
        header=ac.Header(
            header_sci_obj=None, header_hdu_obj=None, readout_offsets=(3, 5)
        ),
    ).native

    ccd_phase = ac.CCDPhase(
        full_well_depth=1e3, well_notch_depth=0.0, well_fill_power=1.0
    )

    traps = [ac.TrapInstantCapture(10.0, -1.0 / np.log(0.5))]

    clocker = ac.Clocker2D(parallel_express=3)

    with pytest.raises(exc.ClockerException):
        clocker.add_cti(data=arr)

    with pytest.raises(exc.ClockerException):
        clocker.add_cti(data=arr, parallel_trap_list=traps)

    with pytest.raises(exc.ClockerException):
        clocker.add_cti(data=arr, serial_trap_list=traps)

    with pytest.raises(exc.ClockerException):
        clocker.add_cti(data=arr, parallel_trap_list=traps, serial_trap_list=traps)

    with pytest.raises(exc.ClockerException):
        clocker.add_cti(data=arr, parallel_ccd=ccd_phase)

    with pytest.raises(exc.ClockerException):
        clocker.add_cti(data=arr, serial_ccd=ccd_phase)

    with pytest.raises(exc.ClockerException):
        clocker.add_cti(data=arr, parallel_ccd=ccd_phase, serial_ccd=ccd_phase)

    with pytest.raises(exc.ClockerException):
        clocker.remove_cti(data=arr)

    with pytest.raises(exc.ClockerException):
        clocker.remove_cti(data=arr, parallel_trap_list=traps)

    with pytest.raises(exc.ClockerException):
        clocker.remove_cti(data=arr, serial_trap_list=traps)

    with pytest.raises(exc.ClockerException):
        clocker.remove_cti(data=arr, parallel_trap_list=traps, serial_trap_list=traps)

    with pytest.raises(exc.ClockerException):
        clocker.remove_cti(data=arr, parallel_ccd=ccd_phase)

    with pytest.raises(exc.ClockerException):
        clocker.remove_cti(data=arr, serial_ccd=ccd_phase)

    with pytest.raises(exc.ClockerException):
        clocker.remove_cti(data=arr, parallel_ccd=ccd_phase, serial_ccd=ccd_phase)
