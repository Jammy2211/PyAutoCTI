import os

import pytest
import numpy as np
from arcticpy import main, roe as r
import autocti as ac


path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))


class TestFrameAPI:
    def test__array_with_offset_through_arctic__same_as_clocker(self):

        frame = ac.Frame2D.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            pixel_scales=1.0,
            roe_corner=(1, 0),
            exposure_info=ac.ExposureInfo(readout_offsets=(5, 10)),
        )

        traps = [ac.TrapInstantCapture(density=10, release_timescale=-1 / np.log(0.5))]
        ccd = ac.CCD(well_fill_power=1, full_well_depth=1000, well_notch_depth=0)
        roe = r.ROE(empty_traps_for_first_transfers=False)

        image_via_arctic = main.add_cti(
            image=frame,
            parallel_traps=traps,
            parallel_ccd=ccd,
            parallel_roe=roe,
            parallel_express=2,
            parallel_offset=5,
        )

        clocker = ac.Clocker(parallel_express=2, parallel_roe=roe)

        image_via_clocker = clocker.add_cti(
            image=frame, parallel_traps=traps, parallel_ccd=ccd
        )

        assert image_via_arctic == pytest.approx(image_via_clocker, 1.0e-4)

        image_via_arctic = main.add_cti(
            image=frame,
            serial_traps=traps,
            serial_ccd=ccd,
            serial_roe=roe,
            serial_express=2,
            serial_offset=10,
        )

        clocker = ac.Clocker(serial_express=2, serial_roe=roe)

        image_via_clocker = clocker.add_cti(
            image=frame, serial_traps=traps, serial_ccd=ccd
        )

        assert image_via_arctic == pytest.approx(image_via_clocker, 1.0e-4)
