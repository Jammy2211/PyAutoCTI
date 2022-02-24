from typing import Optional, Tuple

import autoarray as aa

from autocti.extract.two_d.abstract import Extract2D


class Extract2DMisc(Extract2D):
    def __init__(
        self,
        region_list: aa.type.Region2DList,
        serial_prescan: Optional[aa.type.Region2DLike] = None,
        serial_overscan: Optional[aa.type.Region2DLike] = None,
    ):
        """
        Abstract class containing methods for extracting regions from a 2D charge injection image.

        This uses the `region_list`, which contains the charge injection regions in pixel coordinates.

        Parameters
        ----------
        region_list
            Integer pixel coordinates specifying the corners of each charge injection region (top-row, bottom-row,
            left-column, right-column).
        """

        super().__init__(region_list=region_list)

        if isinstance(serial_prescan, tuple):
            serial_prescan = aa.Region2D(region=serial_prescan)

        if isinstance(serial_overscan, tuple):
            serial_overscan = aa.Region2D(region=serial_overscan)

        self.serial_prescan = serial_prescan
        self.serial_overscan = serial_overscan

        from autocti.extract.two_d.parallel_fpr import Extract2DParallelFPR
        from autocti.extract.two_d.parallel_eper import Extract2DParallelEPER
        from autocti.extract.two_d.serial_fpr import Extract2DSerialFPR
        from autocti.extract.two_d.serial_eper import Extract2DSerialEPER

        self.extract_parallel_fpr = Extract2DParallelFPR(region_list=region_list)
        self.extract_parallel_eper = Extract2DParallelEPER(region_list=region_list)
        self.extract_serial_fpr = Extract2DSerialFPR(region_list=region_list)
        self.extract_serial_eper = Extract2DSerialEPER(region_list=region_list)

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

    def parallel_epers_array_2d_from(self, array: aa.Array2D) -> aa.Array2D:
        """
        Extract all of the areas of an `Array2D` that contain the parallel EPERs and return them as a new `Array2D`
        where these extracted regions are included and everything else (e.g. the charge injection regions, serial
        EPERs) are zeros.

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

        parallel_array = self.non_regions_array_2d_from(array=array)

        parallel_array.native[self.serial_prescan.slice] = 0.0
        parallel_array.native[self.serial_overscan.slice] = 0.0

        return parallel_array

    def parallel_fprs_and_epers_array_2d_from(
        self,
        array: aa.Array2D,
        fpr_pixels: Tuple[int, int] = None,
        trails_pixels: Tuple[int, int] = None,
    ) -> aa.Array2D:
        """
        Extract all of the data values in an input `array2D` corresponding to the parallel front edges and trails of
        each the charge-injection region.

        One can specify the range of rows that are extracted, for example:

        fpr_pixels = (0, 1) will extract just the first leading front edge row.
        fpr_pixels = (0, 2) will extract the leading two front edge rows.
        trails_pixels = (0, 1) will extract the first row of trails closest to the charge injection region.

        The diagram below illustrates the arrays that are extracted from the input array for `fpr_pixels=(0,1)`
        and `trails_pixels=(0,1)`:

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
        P [...][ttttttttttttttttttttt][sss]    | of
        | [...][ccccccccccccccccccccc][sss]    | clocking
          [...][ccccccccccccccccccccc][sss]    |

        []     [=====================]
               <--------Ser---------

        The extracted array keeps just the leading edges and trails following all charge injection scans and
        replaces all other values with 0s:

               [000000000000000000000]
               [000000000000000000000]
          [000][ttttttttttttttttttttt][000]
          [000][000000000000000000000][000]
        | [000][ccccccccccccccccccccc][000]    |
        | [000][000000000000000000000][000]    | Direction
        P [000][ttttttttttttttttttttt][000]    | of
        | [000][000000000000000000000][000]    | clocking
          [000][ccccccccccccccccccccc][000]    |

        []     [=====================]
               <--------Ser---------

        Parameters
        ------------
        fpr_pixels
            The row indexes to extract the front edge between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows).
        trails_pixels
            The row indexes to extract the trails between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows)
        """
        new_array = array.native.copy() * 0.0

        if fpr_pixels is not None:

            new_array = self.extract_parallel_fpr.add_to_array(
                new_array=new_array, array=array, pixels=fpr_pixels
            )

        if trails_pixels is not None:

            new_array = self.extract_parallel_eper.add_to_array(
                new_array=new_array, array=array, pixels=trails_pixels
            )

        return new_array

    def serial_epers_array_2d_from(self, array: aa.Array2D) -> aa.Array2D:
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
        array = self.serial_fprs_and_epers_array_2d_from(
            array=array, trails_pixels=(0, self.serial_overscan.total_columns)
        )
        return array

    def serial_fprs_and_epers_array_2d_from(
        self, array: aa.Array2D, fpr_pixels=None, trails_pixels=None
    ) -> aa.Array2D:
        """
        Extract an array of all of the serial FPRs and EPERs of each the charge-injection scans from a charge
        injection array.

        One can specify the range of columns that are extracted, for example:

        fpr_pixels = (0, 1) will extract just the leading front edge column.
        fpr_pixels = (0, 2) will extract the leading two front edge columns.
        trails_pixels = (0, 1) will extract the first column of trails closest to the charge injection region.

        The diagram below illustrates the arrays that is extracted from a array for `fpr_pixels=(0,2)` and
        `trails_pixels=(0,2)`:

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
          [...][ccccccccccccccccccccc][sts]
        | [...][ccccccccccccccccccccc][tst]    |
        | [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | Direction
        P [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | of
        | [...][ccccccccccccccccccccc][sts]    | clocking
          [...][ccccccccccccccccccccc][tst]    |

        []     [=====================]
               <--------Ser---------

        The extracted array keeps just the leading edge and trails following all charge injection scans and
        replaces all other values with 0s:

               [000000000000000000000]
               [000000000000000000000]
          [000][000000000000000000000][000]
          [000][cc0000000000000000000][st0]
        | [000][cc0000000000000000000][ts0]    |
        | [000][000000000000000000000][000]    | Direction
        P [000][000000000000000000000][000]    | of
        | [000][cc0000000000000000000][st0]    | clocking
          [000][cc0000000000000000000][st0]    |

        []     [=====================]
               <--------Ser---------

        Parameters
        ------------
        array
        fpr_pixels
            The column indexes to extract the front edge between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd rows)
        trails_pixels
            The column indexes to extract the trails between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd rows)
        """
        new_array = array.native.copy() * 0.0

        if fpr_pixels is not None:

            new_array = self.extract_serial_fpr.add_to_array(
                new_array=new_array, array=array, pixels=fpr_pixels
            )

        if trails_pixels is not None:

            new_array = self.extract_serial_eper.add_to_array(
                new_array=new_array, array=array, pixels=trails_pixels
            )

        return new_array
