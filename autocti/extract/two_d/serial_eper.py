from typing import Optional, Tuple

import autoarray as aa

from autocti.extract.two_d.abstract import Extract2D


class Extract2DSerialEPER(Extract2D):
    @property
    def binning_axis(self) -> int:
        """
        The axis over which binning is performed to turn a 2D serial EPER into a 1D EPER.

        For a serial extract `axis=0` such that binning is performed over the columns containing the EPER.
        """
        return 0

    def region_list_from(self, pixels: Tuple[int, int]):
        """
        Extract the serial EPERs of every charge injection region on the charge injection image and return as a list
        of 2D arrays.

        The diagram below illustrates the extraction for `pixels=(0, 1)`:

        [] = read-out electronics
        [==========] = read-out register
        [..........] = serial prescan
        [pppppppppp] = parallel overscan
        [ssssssssss] = serial overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicate the region index)
        [tttttttttt] = parallel / serial charge injection region trail

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
           [...][ttttttttttttttttttttt][sss]
           [...][c1c1cc1c1cc1cc1ccc1cc][st1]
        |  [...][1c1c1cc1c1cc1ccc1cc1c][ts0]    |
        |  [...][ttttttttttttttttttttt][sss]    | Direction
       Par [...][ttttttttttttttttttttt][sss]    | of
        |  [...][0ccc0cccc0cccc0cccc0c][st1]    | clocking
       \/  [...][cc0ccc0cccc0cccc0cccc][ts0]   \/

        []     [=====================]
               <---------Ser--------

        The extracted regions correspond to the first serial EPER all charge injection regions:

        region_list[0] = [0, 2, 22, 225 (serial prescan is 3 pixels)
        region_list[1] = [4, 6, 22, 25] (serial prescan is 3 pixels)

        For `pixels=(0,1)` the extracted arrays returned via the `array_2d_list_from()` function keep the first
        serial EPER of each charge injection region:

        array_2d_list[0] = [st0]
        array_2d_list[1] = [st1]

        Parameters
        ----------
        pixels
            The column indexes to extract the trails between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd columns)
        """
        return [
            region.serial_trailing_region_from(pixels=pixels)
            for region in self.region_list
        ]

    def array_2d_from(self, array: aa.Array2D) -> aa.Array2D:
        """
        Extract an arrays of all of the serial EPERs in the serial overscan region, that are to the side of a
        charge-injection scans from a charge injection array.

        The diagram below illustrates the arrays that is extracted from a array:

        [] = read-out electronics
        [==========] = read-out register
        [..........] = serial prescan
        [pppppppppp] = parallel overscan
        [ssssssssss] = serial overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicate the region index)
        [tttttttttt] = parallel / serial charge injection region trail

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][xxxxxxxxxxxxxxxxxxxxx][sss]
          [...][ccccccccccccccccccccc][tst]
        | [...][ccccccccccccccccccccc][sts]    |
        | [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | Direction
        P [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | of
        | [...][ccccccccccccccccccccc][tst]    | clocking
        \/[...][ccccccccccccccccccccc][sts]    \/

        []     [=====================]
               <--------Ser---------

        The extracted array keeps just the trails following all charge injection scans and replaces all other
        values with 0s:

               [000000000000000000000]
               [000000000000000000000]
          [000][000000000000000000000][000]
          [000][000000000000000000000][tst]
        | [000][000000000000000000000][sts]    |
        | [000][000000000000000000000][000]    | Direction
        P [000][000000000000000000000][000]    | of
        | [000][000000000000000000000][tst]    | clocking
          [000][000000000000000000000][sts]    |

        []     [=====================]
               <--------Ser---------
        """

        serial_array = array.native.copy() * 0.0

        return self.add_to_array(
            new_array=serial_array,
            array=array,
            pixels=(0, self.serial_overscan.total_columns),
        )
