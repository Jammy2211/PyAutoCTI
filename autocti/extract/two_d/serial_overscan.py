from typing import List, Tuple

import autoarray as aa

from autocti.extract.two_d.abstract import Extract2D


class Extract2DSerialOverscan(Extract2D):
    @property
    def binning_axis(self) -> int:
        """
        The axis over which binning is performed to turn a 2D serial overscan into a 1D array (which likely
        contains the EPER of the main data).

        For a serial extract `axis=1` such that binning is performed over the rows containing the FPR.
        """
        return 0

    def region_list_from(self, pixels: Tuple[int, int]) -> List[aa.Region2D]:
        """
        Returns a list of the 2D serial overscan regions, which is simply the serial overscan input to the
        object.

        This is so that the extract API can be mimicked across all extractors.

        (top-row, bottom-row, left-column, right-column) = (y0, y1, x0, x1)

        The serial overscan spans all columns of the image, thus the coordinates x0 and x1 do not change. y0 and y1
        are updated based on the `pixels` input.

        Negative pixel values are supported to the `pixels` tuple, whereby columns in front of the serial overscan are
        also extracted.

        The diagram below illustrates the extraction for `pixels=(0, 1)`:

        [] = read-out electronics
        [==========] = read-out register
        [..........] = serial prescan
        [pppppppppp] = serial overscan
        [ssssssssss] = serial overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicate the region index)
        [tttttttttt] = serial / serial charge injection region trail

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
           [...][ttttttttttttttttttttt][sss]
           [...][c1c1cc1c1cc1cc1ccc1cc][sss]
        |  [...][1c1c1cc1c1cc1ccc1cc1][sss]    |
        |  [...][ttttttttttttttttttttt][sss]    | Direction
       Par [...][ttttttttttttttttttttt][sss]    | of
        |  [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
       \/  [...][cc0ccc0cccc0cccc0cccc][sss]    \/

        []     [=====================]
               <---------Ser--------

        The extracted regions correspond to the serial overscan [sss] regions.

        For `pixels=(0,1)` the extracted arrays returned via the `array_2d_list_from()` function keep the first
        serial pixels across the columns of the overscan:

        array_2d_list[0] = [s]

        Parameters
        ----------
        pixels
            The column pixel index which determines the region of the overscan (e.g. `pixels=(0, 3)` will compute the
            region corresponding to the 1st, 2nd and 3rd overscan columns).
        """
        serial_overscan_extract = aa.Region2D(
            (
                self.serial_overscan.y0,
                self.serial_overscan.y1,
                self.serial_overscan.x0 + pixels[0],
                self.serial_overscan.x0 + pixels[1],
            )
        )

        return [serial_overscan_extract]
