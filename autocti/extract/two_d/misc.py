from typing import Tuple

import autoarray as aa

from autocti.extract.two_d.abstract import Extract2D


class Extract2DMisc(Extract2D):
    def regions_array_2d_from(self, array: aa.Array2D) -> aa.Array2D:
        """
        Extract all of the charge-injection regions from an input `Array2D` object and returns them as a new `Array2D`
        where these extracted regions are included and all other entries are zeros.

        The dimensions of the input array therefore do not change (unlike other `Layout2DCI` methods).

        The diagram below illustrates the extraction:

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
          [...][ccccccccccccccccccccc][sss]
        | [...][ccccccccccccccccccccc][sss]    |
        | [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | Direction
      Par [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | of
        | [...][ccccccccccccccccccccc][sss]    | clocking
       \/  [...][ccccccccccccccccccccc][sss]   \/

        []     [=====================]
               <--------Ser---------

        The extracted array keeps just the charge injection region, all other values become 0:

               [000000000000000000000]
               [000000000000000000000]
          [000][000000000000000000000][000]
          [000][ccccccccccccccccccccc][000]
        | [000][ccccccccccccccccccccc][000]    |
        | [000][000000000000000000000][000]    | Direction
       Par[000][000000000000000000000][000]    | of
        | [000][ccccccccccccccccccccc][000]    | clocking
       \/ [000][ccccccccccccccccccccc][000]   \/

        []     [=====================]
               <--------Ser---------
        """

        new_array = array.native.copy() * 0.0

        for region in self.region_list:
            new_array[region.slice] += array.native[region.slice]

        return new_array

    def non_regions_array_2d_from(self, array: aa.Array2D) -> aa.Array2D:
        """
        Extract all of the areas of an `Array2D` that are not within any of the layout's charge-injection regions
        and return them as a new `Array2D` where these extracted regions are included and the charge injection regions
        are zeros

        The extracted array therefore includes all EPER trails and other regions of the image which may contain
        signal but are not in the FPR.

        The dimensions of the input array therefore do not change (unlike other `Layout2DCI` methods).

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

        The extracted array keeps everything except the charge injection  region,which become 0s:

               [tptpptptptpptpptpptpt]
               [tptptptpptpttptptptpt]
          [000][ttttttttttttttttttttt][000]
          [000][000000000000000000000][000]
        | [000][000000000000000000000][000]    |
        | [000][ttttttttttttttttttttt][000]    | Direction
       Par[000][ttttttttttttttttttttt][000]    | of
        | [000][000000000000000000000][000]    | clocking
          [000][000000000000000000000][000]   \/

        []     [=====================]
               <--------Ser---------
        """

        non_regions_ci_array = array.native.copy()

        for region in self.region_list:
            non_regions_ci_array[region.slice] = 0.0

        return non_regions_ci_array
