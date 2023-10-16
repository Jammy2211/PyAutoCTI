from typing import List, Optional

from arcticpy import CCD
from arcticpy import CCDPhase
from arcticpy import TrapInstantCapture

from autoconf.dictable import from_json, output_to_json

from autocti import exc


class AbstractClocker:
    def __init__(self, iterations: int = 1, verbosity: int = 0):
        """
        An abstract clocker, which wraps the c++ arctic CTI clocking algorithm in **PyAutoCTI**.

        Parameters
        ----------
        iterations
            The number of iterations used to correct CTI from an image.
        verbosity
            Whether to silence print statements and output from the c++ arctic call.
        """

        self.iterations = iterations
        self.verbosity = verbosity

    def ccd_from(self, ccd_phase: CCDPhase) -> CCD:
        """
        Returns a `CCD` object from a `CCDPhase` object.

        The `CCDPhase` describes the volume-filling behaviour of the CCD and is therefore used as a model-component
        in CTI calibration. To call arctic it needs converting to a `CCD` object.

        Parameters
        ----------
        ccd_phase
            The ccd phase describing the volume-filling behaviour of the CCD.

        Returns
        -------
        A `CCD` object based on the input phase which is passed to the c++ arctic.
        """
        if ccd_phase is not None:
            return CCD(phases=[ccd_phase], fraction_of_traps_per_phase=[1.0])

    def check_traps(
        self,
        trap_list_0: Optional[List[TrapInstantCapture]],
        trap_list_1: Optional[List[TrapInstantCapture]] = None,
    ):
        """
        Checks that there are trap species passed to the clocking algorithm and raises an exception if not.

        Parameters
        ----------
        trap_list
            A list of the trap species on the CCD which capture and release electrons during clock to as to add CTI.
        """
        if not any([trap_list_0, trap_list_1]):
            raise exc.ClockerException(
                "No Trap species were passed to the add_cti method"
            )

    def check_ccd(self, ccd_list: List[CCDPhase]):
        """
        Checks that there are trap species passed to the clocking algorithm and raises an exception if not.

        Parameters
        ----------
        ccd_list
            The ccd phase settings describing the volume-filling behaviour of the CCD which characterises the capture
            and release of electrons and therefore CTI.
        """
        if not any(ccd_list):
            raise exc.ClockerException("No CCD object was passed to the add_cti method")

    @classmethod
    def from_json(cls, file_path):
        return from_json(file_path=file_path)

    def output_to_json(self, file_path):
        output_to_json(obj=self, file_path=file_path)
