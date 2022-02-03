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
