from arcticpy.arctic import clock
from arcticpy_cpp import pyarctic

import numpy as np
import pytest


def test__add_cti_to_image_with_both_clockers__identical_results(
    traps_x2, ccd_volume_complex
):

    image = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [0.0, 1.0, 2.0]]
    )

    py_clocker = clock.Clocker(
        iterations=1,
        parallel_express=5,
        parallel_charge_injection_mode=False,
        parallel_readout_offset=0,
        serial_express=5,
        serial_readout_offset=0,
    )

    py_image_with_cti = py_clocker.add_cti(
        image=image, parallel_traps=traps_x2, parallel_ccd_volume=ccd_volume_complex
    )

    print(py_image_with_cti)

    cpp_clocker = pyarctic.Clocker(
        iterations=1,
        parallel_express=5,
        parallel_charge_injection_mode=False,
        parallel_readout_offset=0,
        serial_express=5,
        serial_readout_offset=0,
    )

    cpp_image_with_cti = cpp_clocker.add_cti(
        image=image, parallel_traps=traps_x2, parallel_ccd_volume=ccd_volume_complex
    )

    print(cpp_image_with_cti)

    assert py_image_with_cti == pytest.approx(cpp_image_with_cti, 1.0e-4)
