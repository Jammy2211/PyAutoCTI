from typing import List, Optional

from arcticpy import CCD
from arcticpy import CCDPhase
from arcticpy import PixelBounce
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

    @classmethod
    def from_json(cls, file_path):
        return from_json(file_path=file_path)

    def output_to_json(self, file_path):
        output_to_json(obj=self, file_path=file_path)
