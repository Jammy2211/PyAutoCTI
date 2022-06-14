from autocti.charge_injection.extract.parallel_fpr import Extract2DParallelFPRCI
from autocti.extract.two_d.master import Extract2DMaster


class Extract2DMasterCI(Extract2DMaster):
    """
    Extends the parallel 2D First Pixel Response (FPR) extractor class with functionality that extracts
    charge injection data's charge injectionr regions and estimates the properties of the charge injection.
    """

    @property
    def parallel_fpr(self):
        return Extract2DParallelFPRCI(
            region_list=self.region_list,
            parallel_overscan=self._parallel_overscan,
            serial_prescan=self._serial_prescan,
            serial_overscan=self._serial_overscan,
        )
