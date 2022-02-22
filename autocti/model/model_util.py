import numpy as np
from typing import Optional, List

from arcticpy.src import ccd
from arcticpy.src import traps


class CTI1D:
    def __init__(
        self,
        traps: Optional[List[traps.TrapInstantCapture]] = None,
        ccd: Optional[ccd.CCDPhase] = None,
    ):

        self.traps = traps
        self.ccd = ccd


class CTI2D:
    def __init__(
        self,
        parallel_traps: Optional[List[traps.TrapInstantCapture]] = None,
        parallel_ccd: Optional[ccd.CCDPhase] = None,
        serial_traps: Optional[List[traps.TrapInstantCapture]] = None,
        serial_ccd: Optional[ccd.CCDPhase] = None,
    ):

        self.parallel_traps = parallel_traps
        self.parallel_ccd = parallel_ccd
        self.serial_traps = serial_traps
        self.serial_ccd = serial_ccd

    @property
    def trap_list(self) -> List[traps.AbstractTrap]:
        """
        Combine the parallel and serial trap lists to make an overall list of traps in the model.

        This is not a straight forward list addition, because **PyAutoFit** model's store the `parallel_traps` and
        `serial_traps` entries as a `ModelInstance`. This object does not allow for straight forward list addition.

        Returns
        -------

        """
        parallel_traps = self.parallel_traps or []
        serial_traps = self.serial_traps or []

        return [trap for trap in parallel_traps] + [trap for trap in serial_traps]

    @property
    def delta_ellipticity(self):
        return sum([trap.delta_ellipticity for trap in self.trap_list])


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
