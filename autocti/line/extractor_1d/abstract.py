import numpy as np

import autoarray as aa


class Extractor1D:
    def __init__(self, region_list: aa.type.Region1DList):
        """
        Abstract class containing methods for extracting regions from a 1D line dataset which contains some sort of
        original signal whose profile before CTI is known (e.g. warm pixel, charge injection).

        This uses the `region_list`, which contains the signal's regions in pixel coordinates (x0, x1).

        Parameters
        ----------
        region_list
            Integer pixel coordinates specifying the corners of signal (x0, x1).
        """
        self.region_list = list(map(aa.Region1D, region_list))

    @property
    def total_pixels_min(self) -> int:
        """
        The number of rows between the read-out electronics and the signal closest to them.
        """
        return np.min([region.total_pixels for region in self.region_list])
