import copy

import autoarray as aa

from typing import Tuple


class SettingsImagingCI(aa.SettingsImaging):
    def __init__(
        self,
        parallel_pixels: Tuple[int, int] = None,
        serial_pixels: Tuple[int, int] = None,
    ):

        super().__init__()

        self.parallel_pixels = parallel_pixels
        self.serial_pixels = serial_pixels

    def modify_via_fit_type(self, is_parallel_fit, is_serial_fit):
        """
        Modify the settings based on the type of fit being performed where:

        - If the fit is a parallel only fit (is_parallel_fit=True, is_serial_fit=False) the serial_pixels are set to None
          and all other settings remain the same.

        - If the fit is a serial only fit (is_parallel_fit=False, is_serial_fit=True) the parallel_pixels are set to
          None and all other settings remain the same.

        - If the fit is a parallel and serial fit (is_parallel_fit=True, is_serial_fit=True) the *parallel_pixels* and
          *serial_pixels* are set to None and all other settings remain the same.

        These settings reflect the appropriate way to extract the charge injection imaging data for fits which use a
        parallel only CTI model, serial only CTI model or fit both.

        Parameters
        ----------
        is_parallel_fit
            If True, the CTI model that is used to fit the charge injection data includes a parallel CTI component.
        is_serial_fit
            If True, the CTI model that is used to fit the charge injection data includes a serial CTI component.
        """

        settings = copy.copy(self)

        if is_parallel_fit:
            settings.serial_pixels = None

        if is_serial_fit:
            settings.parallel_pixels = None

        return settings
