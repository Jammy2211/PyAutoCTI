import numpy as np
from typing import List, Tuple

import autoarray as aa

from autocti.extract.one_d.abstract import Extract1D


class Extract1DEPER(Extract1D):
    def region_list_from(self, pixels: Tuple[int, int]) -> List[aa.Region1D]:
        """
        Returns a list of the (x0, x1) regions containing the EPERs of a 1D CTI dataset.

        These are used for extracting the EPER regions of 1D data.

        Negative pixel values are supported to the `pixels` tuple, whereby pixels in front of the EPERs (e.g.
        the FPR) are also extracted.

        Parameters
        -----------
        pixels
            The row indexes to extract the trails between (e.g. pixels(0, 3) extracts the 1st, 2nd and 3rd pixels)
        """

        return list(
            map(
                lambda region: region.trailing_region_from(pixels=pixels),
                self.region_list,
            )
        )

    def array_1d_from(self, array: aa.Array1D) -> aa.Array1D:
        """
        Extract all of the data values in an input `array1D` that do not overlap the charge regions or the
        prescan / overscan regions.

        This  extracts a `array1D` that contains only regions of the data where there are parallel trails (e.g. those
        that follow the charge-injection regions).
        """

        array_1d_eper = array.native.copy()

        for region in self.region_list:
            array_1d_eper[region.slice] = 0.0

        array_1d_eper.native[self.prescan.slice] = 0.0
        array_1d_eper.native[self.overscan.slice] = 0.0

        return array_1d_eper
