import numpy as np
from typing import List, Optional, Tuple

import autoarray as aa

from autocti.extract.one_d.abstract import Extract1D


class Extract1DOverscan(Extract1D):
    def region_list_from(
        self,
        pixels: Optional[Tuple[int, int]] = None,
        pixels_from_end: Optional[int] = None,
    ) -> List[aa.Region1D]:
        """
        Returns a list of the (x0, x1) regions containing the overscan of a 1D CTI dataset.

        These are used for extracting the overscan regions of 1D data.

        Negative pixel values can be input into the `pixels` tuple, whereby pixels in front of the FPRs  are also
        extracted.

        Parameters
        ----------
        pixels
            The row indexes to extract the overscan between (e.g. pixels(0, 3) extracts the 1st, 2nd and 3rd pixels)
        """

        if pixels_from_end is not None:
            pixels = (
                self.overscan.total_pixels - pixels_from_end,
                self.overscan.total_pixels,
            )

        return [
            aa.Region1D(
                region=(self.overscan.x0 + pixels[0], self.overscan.x0 + pixels[1])
            )
        ]
