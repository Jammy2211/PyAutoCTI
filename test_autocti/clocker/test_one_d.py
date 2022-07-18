import os

import pytest
import numpy as np

from arcticpy import add_cti

import autocti as ac

path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))


def test__data_mapped_to_2d_and_then_1d():

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

    image_via_arctic =add_cti(
        image=arr_2d,
        parallel_traps=traps,
        parallel_ccd=ccd,
        parallel_roe=roe,
        parallel_express=3,
        parallel_window_offset=3,
    )

    cti = ac.CTI1D(trap_list=traps, ccd=ccd_phase)

    clocker_1d = ac.Clocker1D(express=3, roe=roe)

    image_via_clocker = clocker_1d.add_cti(data=arr_1d, cti=cti)

    assert image_via_arctic.flatten() == pytest.approx(image_via_clocker, 1.0e-4)

    image_corrected = clocker_1d.remove_cti(data=image_via_clocker, cti=cti)

    assert (image_corrected[:] > image_via_clocker[:]).all()
