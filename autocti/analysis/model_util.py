import autofit as af

from autocti.util import ccd
from autocti.util import traps

from typing import Optional, List


class CTI:
    def __init__(
        self,
        parallel_traps: Optional[List[traps.TrapInstantCapture]] = None,
        parallel_ccd: Optional[ccd.CCD] = None,
        serial_traps: Optional[List[traps.TrapInstantCapture]] = None,
        serial_ccd: Optional[ccd.CCD] = None,
    ):

        self.parallel_traps = parallel_traps
        self.parallel_ccd = parallel_ccd
        self.serial_traps = serial_traps
        self.serial_ccd = serial_ccd


def is_parallel_fit(model):
    if model.parallel_ccd is not None and model.serial_ccd is None:
        return True
    return False


def is_serial_fit(model):
    if model.parallel_ccd is None and model.serial_ccd is not None:
        return True
    return False


def is_parallel_and_serial_fit(model):
    if model.parallel_ccd is not None and model.serial_ccd is not None:
        return True
    return False
