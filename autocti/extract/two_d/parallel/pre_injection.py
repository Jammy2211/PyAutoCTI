from typing import List

import autoarray as aa

from autocti.extract.two_d.parallel.abstract import Extract2DParallel
from autocti.extract.settings import SettingsExtract

from autocti.extract.two_d import extract_2d_util


class Extract2DParallelPreInjection(Extract2DParallel):
    def region_list_from(self, settings: SettingsExtract) -> List[aa.Region2D]:
        """
        Returns a list of the 2D parallel regions before the first injected signal (e.g. before the FPR) from
        the `region_list` containing signal  (e.g. the charge injection regions of charge injection data), between
        two input `pixels` indexes.

        Negative pixel values can be input into the `pixels` tuple, whereby columns in front of the region in front of
        the parallel FPRs are extracted.

        A 2D region is defined following the convention:

        (top-row, bottom-row, left-column, right-column) = (y0, y1, x0, x1)

        The pre injection region is defined to contain all pixels before the FPR, not including the rows of pixels
        in the serial prescan and overscans. This is because the region is used to estimate and subtract constant
        background signals across the full CCD (e.g. bias, stray light, etc.). The x0 and x1 coordinates are therefore
        updated to include the serial prescan and overscan regions.

        The y0 and y1 coordinates are updated based on the `pixels` input, but also computed based on where the
        injection occurs (and therefore are before the FPRs).

        The diagram below illustrates the extraction for `pixels=(0, 1)`:

        [] = read-out electronics
        [==========] = read-out register
        [..........] = serial prescan
        [pppppppppp] = parallel overscan
        [ssssssssss] = serial overscan
        [f#ff#f#f#f] = signal region (FPR) (0 / 1 indicate the region index)
        [tttttttttt] = parallel / serial charge injection region trail
        [ppippippipp] The pre injection region to be extracted

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
           [...][ttttttttttttttttttttt][sss]
           [...][c1c1cc1c1cc1cc1ccc1cc][sss]
        |  [...][1c1c1cc1c1cc1ccc1cc1][sss]    |
        |  [...][ttttttttttttttttttttt][sss]    | Direction
        Par [...][ttttttttttttttttttttt][sss]    | of
        |  [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
        |/  [...][cc0ccc0cccc0cccc0cccc][sss]    \/
        |/  [...][ppippippippipppipppip][sss]

        []     [=====================]
               <---------Ser--------

        The extracted regions correspond to the single region before the first parallel FPR:

        region_list[0] = [0, 4, 0, 1] (serial prescan is 3 pixels)
        region_list[1] = [0, 4, 0, 1] (serial prescan is 3 pixels)

        For `pixels=(0,1)` the extracted arrays returned via the `array_2d_list_from()` function keep the first
        parallel FPR of each charge injection region:

        array_2d_list[0] = [ppippippippipppipppip]

        Parameters
        ----------
        settings
           The settings used to extract the parallel FPRs, which for example include the `pixels` tuple specifying the
           range of pixel rows they are extracted between.
        """

        pixels = settings.pixels
        pixels_from_end = settings.pixels_from_end

        if pixels_from_end is not None:
            y0_min = min([region.y0 for region in self.region_list])

            pixels = (y0_min - pixels_from_end, y0_min)

        y0 = pixels[1]

        return [
            aa.Region2D(
                region=(pixels[0], y0, self.region_list[0].x0, self.region_list[0].x1)
            )
        ]
