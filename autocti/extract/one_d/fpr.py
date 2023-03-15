from typing import List, Optional, Tuple

import autoarray as aa

from autocti.extract.one_d.abstract import Extract1D
from autocti.extract.settings import SettingsExtract


class Extract1DFPR(Extract1D):
    def region_list_from(self, settings: SettingsExtract) -> List[aa.Region1D]:
        """
        Returns a list of the (x0, x1) regions containing the FPRs of a 1D CTI dataset.

        These are used for extracting the FPR regions of 1D data.

        Negative pixel values can be input into the `pixels` tuple, whereby pixels in front of the FPRs  are also
        extracted.

        Parameters
        ----------
        pixels
            The row indexes to extract the front edge between (e.g. pixels(0, 3) extracts the 1st, 2nd and 3rd pixels)
        """
        return list(
            map(
                lambda region: region.front_region_from(
                    pixels=settings.pixels, pixels_from_end=settings.pixels_from_end
                ),
                self.region_list,
            )
        )
