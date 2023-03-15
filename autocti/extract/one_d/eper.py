import numpy as np
from typing import List, Optional, Tuple

import autoarray as aa

from autocti.extract.one_d.abstract import Extract1D
from autocti.extract.settings import SettingsExtract


class Extract1DEPER(Extract1D):
    def region_list_from(self, settings: SettingsExtract) -> List[aa.Region1D]:
        """
        Returns a list of the (x0, x1) regions containing the EPERs of a 1D CTI dataset.

        These are used for extracting the EPER regions of 1D data.

        Negative pixel values can be input into the `pixels` tuple, whereby pixels in front of the EPERs (e.g.
        the FPR) are extracted.

        Parameters
        ----------
        pixels
            The row indexes to extract the trails between (e.g. pixels(0, 3) extracts the 1st, 2nd and 3rd pixels)
        """

        pixels = settings.pixels

        region_list = []

        for i, region in enumerate(self.region_list):

            if settings.pixels_from_end is not None:

                parallel_row_spaces = self.parallel_rows_between_regions + [
                    self.trail_size_to_array_edge
                ]

                pixels = (
                    parallel_row_spaces[i] - settings.pixels_from_end,
                    parallel_row_spaces[i],
                )

            region_list.append(region.trailing_region_from(pixels=pixels))

        return region_list

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
