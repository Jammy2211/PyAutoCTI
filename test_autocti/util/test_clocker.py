import os

import pytest
import numpy as np
from arcticpy.src import cti
import autocti as ac
from autocti import exc

path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))


class TestClocker:
    def test__array_with_offset_through_arctic__same_as_clocker(self):

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

        clocker = ac.Clocker(parallel_express=3, parallel_roe=roe)

        image_via_clocker = clocker.add_cti(
            image_pre_cti=arr, parallel_traps=traps, parallel_ccd=ccd_phase
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

        clocker = ac.Clocker(serial_express=2, serial_roe=roe)

        image_via_clocker = clocker.add_cti(
            image_pre_cti=arr, serial_traps=traps, serial_ccd=ccd_phase
        )

        assert image_via_arctic == pytest.approx(image_via_clocker, 1.0e-4)

    def test__raises_exception_if_no_traps_or_ccd_passed(self):

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

        ccd = ac.CCD(phases=[ccd_phase], fraction_of_traps_per_phase=[1.0])
        traps = [ac.TrapInstantCapture(10.0, -1.0 / np.log(0.5))]

        clocker = ac.Clocker(parallel_express=3)

        with pytest.raises(exc.ClockerException):
            clocker.add_cti(image_pre_cti=arr)

        with pytest.raises(exc.ClockerException):
            clocker.add_cti(image_pre_cti=arr, parallel_traps=traps)

        with pytest.raises(exc.ClockerException):
            clocker.add_cti(image_pre_cti=arr, serial_traps=traps)

        with pytest.raises(exc.ClockerException):
            clocker.add_cti(image_pre_cti=arr, parallel_traps=traps, serial_traps=traps)
            
        with pytest.raises(exc.ClockerException):
            clocker.add_cti(image_pre_cti=arr, parallel_ccd=ccd)

        with pytest.raises(exc.ClockerException):
            clocker.add_cti(image_pre_cti=arr, serial_ccd=ccd)

        with pytest.raises(exc.ClockerException):
            clocker.add_cti(image_pre_cti=arr, parallel_ccd=ccd, serial_ccd=ccd)