import numpy as np
import pytest
import autocti as ac
from arcticpy_cpp import pyarctic


def test__add_cti_to_image_with_both_clockers__parallel__identical_results(
    traps_x2, ccd_volume
):

    image = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [0.0, 1.0, 2.0]]
    )

    py_clocker = ac.Clocker(
        iterations=1,
        parallel_express=4,
        parallel_charge_injection_mode=False,
        parallel_readout_offset=0,
    )

    py_image_with_cti = py_clocker.add_cti(
        image=image, parallel_traps=traps_x2, parallel_ccd_volume=ccd_volume
    )

    cpp_clocker = pyarctic.Clocker(
        iterations=1,
        parallel_express=4,
        parallel_charge_injection_mode=False,
        parallel_readout_offset=0,
    )

    cpp_image_with_cti = cpp_clocker.add_cti(
        image=image, parallel_traps=traps_x2, parallel_ccd_volume=ccd_volume
    )

    assert py_image_with_cti == pytest.approx(cpp_image_with_cti, 0.2)


def test__add_cti_to_image_with_both_clockers__serial__identical_results(
    traps_x2, ccd_volume
):

    image = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [0.0, 1.0, 2.0]]
    )

    py_clocker = ac.Clocker(iterations=1, serial_express=3, serial_readout_offset=0)

    py_image_with_cti = py_clocker.add_cti(
        image=image, serial_traps=traps_x2, serial_ccd_volume=ccd_volume
    )

    cpp_clocker = pyarctic.Clocker(
        iterations=1, serial_express=3, serial_readout_offset=0
    )

    cpp_image_with_cti = cpp_clocker.add_cti(
        image=image, serial_traps=traps_x2, serial_ccd_volume=ccd_volume
    )

    assert py_image_with_cti == pytest.approx(cpp_image_with_cti, 0.4)
