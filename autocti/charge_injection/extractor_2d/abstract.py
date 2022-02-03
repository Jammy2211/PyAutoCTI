import numpy as np
from typing import List, Tuple, Union

import autoarray as aa


class Extractor2D:
    def __init__(self, region_list: aa.type.Region2DLike):
        """
        Abstract class containing methods for extracting regions from a 2D charge injection image.

        This uses the `region_list`, which contains the charge injection regions in pixel coordinates.

        Parameters
        ----------
        region_list
            Integer pixel coordinates specifying the corners of each charge injection region (top-row, bottom-row,
            left-column, right-column).
        """
        self.region_list = list(map(aa.Region2D, region_list))

    @property
    def total_rows_min(self) -> int:
        """
        The number of rows between the read-out electronics and the charge injection region closest to them.
        """
        return np.min([region.total_rows for region in self.region_list])

    @property
    def total_columns_min(self) -> int:
        """
        The number of columns between the read-out electronics and the charge injection region closest to them.
        """
        return np.min([region.total_columns for region in self.region_list])

    def region_list_from(self, pixels: Tuple[int, int]) -> List[aa.Region2D]:
        raise NotImplementedError

    def array_2d_list_from(
        self, array: aa.Array2D, pixels: Tuple[int, int]
    ) -> List[aa.Array2D]:
        """
        Extract a specific region from every charge injection region on the charge injection image and return as a list
        of 2D arrays.

        For example, this might extract the parallel EPERs of every charge injection region.

        The `region_2d_list_from()` of each `Extractor` class describes the exact extraction performed for each
        extractor when this function is called..
        """
        return [
            np.ma.array(data=array.native[region.slice], mask=array.mask[region.slice])
            for region in self.region_list_from(pixels=pixels)
        ]

    def add_to_array(
        self, new_array: aa.Array2D, array: aa.Array2D, pixels: Tuple[int, int]
    ) -> aa.Array2D:
        """
        Extracts the parallel FPRs from a charge injection image and adds them to a new image.

        Parameters
        ----------
        new_array
            The 2D array which the extracted parallel FPRs are added to.
        array
            The 2D array which contains the charge injeciton image from which the parallel FPRs are extracted.
        pixels
            The row pixel index which determines the region of the FPR (e.g. `pixels=(0, 3)` will compute the region
            corresponding to the 1st, 2nd and 3rd FPR rows).
        """

        region_list = self.region_list_from(pixels=pixels)

        array_2d_list = self.array_2d_list_from(array=array, pixels=pixels)

        for arr, region in zip(array_2d_list, region_list):
            new_array[region.y0 : region.y1, region.x0 : region.x1] += arr

        return new_array
