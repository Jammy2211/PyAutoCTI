from typing import Optional, List

from arcticpy.src import ccd
from arcticpy.src import traps

from autoconf.dictable import Dictable


class CTI1D(Dictable):
    def __init__(
        self,
        traps: Optional[List[traps.TrapInstantCapture]] = None,
        ccd: Optional[ccd.CCDPhase] = None,
    ):
        """
        An object which determines the behaviour of CTI during 1D clocking.
        
        This includes the traps that capture and trail electrons and the CCD volume filling behaviour.
        
        Parameters
        ----------
        traps
            The traps on the dataset that capture and release electrons during clocking.
        ccd
            The CCD volume filling parameterization which dictates how an electron cloud fills pixels and thus
            how it is subject to traps.
        """
        self.traps = traps
        self.ccd = ccd


class CTI2D(Dictable):
    def __init__(
        self,
        parallel_traps: Optional[List[traps.TrapInstantCapture]] = None,
        parallel_ccd: Optional[ccd.CCDPhase] = None,
        serial_traps: Optional[List[traps.TrapInstantCapture]] = None,
        serial_ccd: Optional[ccd.CCDPhase] = None,
    ):
        """
        An object which determines the behaviour of CTI during 2D parallel and serial clocking.

        This includes the traps that capture and trail electrons and the CCD volume filling behaviour.

        Parameters
        ----------
        parallel_traps
            The traps on the dataset that capture and release electrons during parallel clocking.
        parallel_ccd
            The CCD volume filling parameterization which dictates how an electron cloud fills pixel in the parallel
             direction and thus how it is subject to traps.
        serial_traps
            The traps on the dataset that capture and release electrons during serial clocking.
        serial_ccd
            The CCD volume filling parameterization which dictates how an electron cloud fills pixel in the serial
             direction and thus how it is subject to traps.             
        """
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
