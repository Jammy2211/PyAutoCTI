from typing import Tuple

import autoarray as aa

from autocti.extract.two_d.serial.abstract import Extract2DSerial

from autocti.extract.two_d import extract_2d_util


class Extract2DSerialEPER(Extract2DSerial):
    def region_list_from(self, pixels: Tuple[int, int]):
        """
        Returns a list of the 2D serial EPER regions from the `region_list` containing signal  (e.g. the charge
        injection regions of charge injection data), extracted between two input `pixels` indexes.

        Negative pixel values are supported to the `pixels` tuple, whereby columns in front of the serial EPERs
        (e.g. the FPRs) are also extracted.

        The diagram below illustrates the extraction for `pixels=(0, 1)`:

        [] = read-out electronics
        [==========] = read-out register
        [..........] = serial prescan
        [pppppppppp] = parallel overscan
        [ssssssssss] = serial overscan
        [f#ff#f#f#f] = signal region (FPR) (0 / 1 indicate the region index)
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

    def binned_region_1d_from(self, pixels: Tuple[int, int]) -> aa.Region1D:
        """
        The `Extract` objects allow one to extract a `Dataset1D` from a 2D CTI dataset, which is used to perform
        CTI modeling in 1D.

        This is performed by binning up the data via the `binned_array_1d_from` function.

        In order to create the 1D dataset a `Layout1D` is required, which requires the `region_list` containing the
        charge regions on the 1D dataset (e.g. where the FPR appears in 1D after binning).

        The function returns the this region if the 1D dataset is extracted from theserial EPERs. The
        charge region is only included if there are negative entries in the `pixels` tuple, meaning that pixels
        before the EPERs (e.g. the FPR) are extracted.

        Parameters
        ----------
        pixels
            The column pixel index to extract the EPERs between (e.g. `pixels=(0, 3)` extracts the 1st, 2nd and 3rd EPER
            columns)
        """
        return extract_2d_util.binned_region_1d_eper_from(pixels=pixels)

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
        [f#ff#f#f#f] = signal region (FPR) (0 / 1 indicate the region index)
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
