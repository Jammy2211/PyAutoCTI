import numpy as np
from typing import List, Tuple

import autoarray as aa

from autocti.extract.one_d.abstract import Extract1D


class Extract1DFPR(Extract1D):
    def region_list_from(self, pixels: Tuple[int, int]) -> List[aa.Region1D]:
        """
        Calculate a list of the front edge regions of a line dataset

        The diagram below illustrates the region that calculated from a array for pixels=(0, 1):

        -> Direction of Clocking

        [fffcccccccccccttt]

        The extracted array keeps just the front edges of all regions.

        Parameters
        ------------
        array
            A 1D array of data containing a CTI line.
        pixels
            The row indexes to extract the front edge between (e.g. pixels(0, 3) extracts the 1st, 2nd and 3rd pixels)
        """
        return list(
            map(
                lambda region: region.front_region_from(pixels=pixels), self.region_list
            )
        )
