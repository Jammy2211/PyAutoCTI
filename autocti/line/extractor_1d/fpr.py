import numpy as np
from typing import List, Tuple

import autoarray as aa

from autocti.line.extractor_1d.abstract import Extractor1D


class Extractor1DFPR(Extractor1D):
    def array_1d_list_from(
        self, array: aa.Array1D, pixels: Tuple[int, int]
    ) -> List[aa.Array1D]:
        """
        Extract a list of the front edges of a 1D line array.

        The diagram below illustrates the arrays that is extracted from a array for pixels=(0, 1):

        -> Direction of Clocking

        [fffcccccccccccttt]

        The extracted array keeps just the front edges corresponding to the `f` entries.
        Parameters
        ------------
        array
            A 1D array of data containing a CTI line.
        pixels
            The row indexes to extract the front edge between (e.g. pixels(0, 3) extracts the 1st, 2nd and 3rd pixels)
        """
        front_region_list = self.region_list_from(pixels=pixels)
        front_array_list = list(
            map(lambda region: array.native[region.slice], front_region_list)
        )
        front_mask_list = list(
            map(lambda region: array.mask[region.slice], front_region_list)
        )
        front_array_list = list(
            map(
                lambda front_array, front_mask: np.ma.array(
                    front_array, mask=front_mask
                ),
                front_array_list,
                front_mask_list,
            )
        )
        return front_array_list

    def stacked_array_1d_from(
        self, array: aa.Array1D, pixels: Tuple[int, int]
    ) -> aa.Array1D:
        front_arrays = self.array_1d_list_from(array=array, pixels=pixels)
        return np.ma.mean(np.ma.asarray(front_arrays), axis=0)

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

    def add_to_array(
        self, new_array: aa.Array1D, array: aa.Array1D, pixels: Tuple[int, int]
    ) -> aa.Array1D:

        region_list = [
            region.front_region_from(pixels=pixels) for region in self.region_list
        ]

        array_1d_list = self.array_1d_list_from(array=array, pixels=pixels)

        for arr, region in zip(array_1d_list, region_list):
            new_array[region.x0 : region.x1] += arr

        return new_array
