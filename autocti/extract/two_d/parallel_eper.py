from typing import Tuple

import autoarray as aa

from autocti.extract.two_d import extract_2d_util

from autocti.extract.two_d.abstract import Extract2D


class Extract2DParallelEPER(Extract2D):
    @property
    def binning_axis(self) -> int:
        """
        The axis over which binning is performed to turn a 2D parallel EPER into a 1D EPER.

        For a parallel extract `axis=1` such that binning is performed over the rows containing the EPER.
        """
        return 1

    def region_list_from(self, pixels: Tuple[int, int]):
        """
        Extract the parallel EPERs of every charge injection region on the charge injection image and return as a list
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
          [...][t1t1t1t1t1t1t1t1t1t1t][sss]
          [...][c1c1cc1c1cc1cc1ccc1cc][sss]
        | [...][1c1c1cc1c1cc1ccc1cc1c][sss]     |
        | [...][t0t0t0t0t0t0t0t0t0t0t][sss]    | Direction
      Par [...][0t0t0t0t0t0t0t0t0t0t0][sss]    | of
        | [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
       \/ [...][cc0ccc0cccc0cccc0cccc][sss]   \/

        []     [=====================]
               <---------Ser----------

        The extracted regions correspond to the first parallel EPER all charge injection regions:

        region_list[0] = [2, 4, 3, 21] (serial prescan is 3 pixels)
        region_list[1] = [6, 7, 3, 21] (serial prescan is 3 pixels)

        For `pixels=(0,1)` the extracted arrays returned via the `array_2d_list_from()` function keep the first
        parallel EPER of each charge injection region:

        array_2d_list[0] = [t0t0t0tt0t0t0t0t0t0t0]
        array_2d_list[1] = [1t1t1t1t1t1t1t1t1t1t1]

        Parameters
        ----------
        pixels
            The row indexes to extract the trails between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows)
        """
        return [
            region.parallel_trailing_region_from(pixels=pixels)
            for region in self.region_list
        ]

    def binned_region_1d_from(self, pixels: Tuple[int, int]) -> aa.Region1D:
        """
        The `Extract` objects allow one to extract a `Dataset1D` from a 2D CTI dataset, which is used to perform
        CTI modeling in 1D.

        This is performed by binning up the data via the `binned_array_1d_from` function.

        In order to create the 1D dataset a `Layout1D` is required, which requires the `region_list` containing the
        charge regions on the 1D dataset (e.g. where the FPR appears in 1D after binning).

        The function returns the this region if the 1D dataset is extracted from the parallel EPERs. The
        charge region is only included if there are negative entries in the `pixels` tuple, meaning that pixels
        before the EPERs (e.g. the FPR) are extracted.

        Parameters
        ----------
        pixels
            The row pixel index to extract the EPERs between (e.g. `pixels=(0, 3)` extracts the 1st, 2nd and 3rd EPER
            rows)
        """
        return extract_2d_util.binned_region_1d_eper_from(pixels=pixels)

    def array_2d_from(self, array: aa.Array2D) -> aa.Array2D:
        """
        Extract all of the areas of an `Array2D` that contain the parallel EPERs and return them as a new `Array2D`
        where these extracted regions are included and everything else (e.g. the charge injection regions, serial
        EPERs) are zeros.

        The dimensions of the input array therefore do not change (unlike other `Layout2DCI` methods).

        Negative pixel values are supported to the `pixels` tuple, whereby rows in front of the parallel EPERs (e.g.
        the FPR) are also extracted.

        The diagram below illustrates the extraction:

        [] = read-out electronics
        [==========] = read-out register
        [..........] = serial prescan
        [pppppppppp] = parallel overscan
        [ssssssssss] = serial overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicate the region index)
        [tttttttttt] = parallel / serial charge injection region trail

               [tptpptptptpptpptpptpt]
               [tptptptpptpttptptptpt]
          [...][ttttttttttttttttttttt][sss]
          [...][ccccccccccccccccccccc][sss]
        | [...][ccccccccccccccccccccc][sss]    |
        | [...][ttttttttttttttttttttt][sss]    | Direction
       Par[...][ttttttttttttttttttttt][sss]    | of
        | [...][ccccccccccccccccccccc][sss]    | clocking
       \/ [...][ccccccccccccccccccccc][sss]   \/

        []     [=====================]
               <--------Ser---------

        The extracted array keeps only the parallel EPERs, everything else become 0s:

               [tptpptptptpptpptpptpt]
               [tptptptpptpttptptptpt]
          [000][ttttttttttttttttttttt][000]
          [000][000000000000000000000][000]
        | [000][000000000000000000000][000]    |
        | [000][ttttttttttttttttttttt][000]    | Direction
       Par[000][ttttttttttttttttttttt][000]    | of
        | [000][000000000000000000000][000]    | clocking
       \/ [000][000000000000000000000][000]   \/

        []     [=====================]
               <--------Ser---------
        """

        parallel_array = array.native.copy()

        for region in self.region_list:
            parallel_array[region.slice] = 0.0

        parallel_array.native[self.serial_prescan.slice] = 0.0
        parallel_array.native[self.serial_overscan.slice] = 0.0

        return parallel_array
