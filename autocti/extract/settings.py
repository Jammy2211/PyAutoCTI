import numpy as np
from typing import List, Optional, Tuple

import autoarray as aa


class SettingsExtract:
    def __init__(
        self,
        pixels: Optional[Tuple[int, int]] = None,
        pixels_from_end: Optional[int] = None,
        force_same_row_size: bool = False,
    ):
        """
        Class containing settings used by the `Extract` classes, which perform the extraction of a 2D array into
        another other 2D or 1D arrays.

        Parameters
        ----------
        pixels
            The row or column pixel indexes to extract the region (e.g. FPR / EPER) between. For example, for the
            ``Extract2DParallelFPR`` object the input `pixels=(0, 3)` extracts the 1st, 2nd and 3rd FPR rows). For
           ``Extract2DSerialFPR` objects the input `pixels=(0, 3)` extracts the 1st, 2nd and 3rd FPR columns).
        pixels_from_end
            Alternative row or column pixel index specification, which extracts this number of pixels from the end of
            each region (e.g. FPR / EPER). For example, if each FPR is 100 pixels and `pixels_from_end=10`, the
            last 10 pixels of each FPR (pixels (90, 100)) are extracted.
        force_same_row_size
            If `True`, the returned arrays are forced to have the same number of rows. This is useful for parallel
            EPER regions, where the number of rows in each region is not the same. If `False`, the returned arrays
            have the same number of rows as the region they were extracted from.
        """
        self.pixels = pixels
        self.pixels_from_end = pixels_from_end
        self.force_same_row_size = force_same_row_size

    def region_list_from(self, region_list: List[aa.Region2D]) -> List[aa.Region2D]:
        if self.force_same_row_size:
            row_size = np.min([region.total_rows for region in region_list])

            region_list = [
                aa.Region2D(
                    region=(region.y0, region.y0 + row_size, region.x0, region.x1)
                )
                for region in region_list
            ]

        return region_list
