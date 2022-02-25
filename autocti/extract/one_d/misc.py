from typing import Tuple

from autocti.extract.one_d.abstract import Extract1D

import autoarray as aa


class Extract1DMisc(Extract1D):
    def __init__(
        self,
        region_list: aa.type.Region1DList,
        prescan: aa.type.Region1DLike = None,
        overscan: aa.type.Region1DLike = None,
    ):
        """
        Abstract class containing methods for extracting regions from a 1D line dataset which contains some sort of
        original signal whose profile before CTI is known (e.g. warm pixel, charge injection).

        This uses the `region_list`, which contains the signal's regions in pixel coordinates (x0, x1).

        Parameters
        ----------
        region_list
            Integer pixel coordinates specifying the corners of signal (x0, x1).
        """
        super().__init__(region_list=region_list)

        from autocti.extract.one_d.fpr import Extract1DFPR
        from autocti.extract.one_d.eper import Extract1DEPER

        self.extract_fpr = Extract1DFPR(region_list=region_list)
        self.extract_eper = Extract1DEPER(region_list=region_list)

        if isinstance(prescan, tuple):
            prescan = aa.Region1D(region=prescan)

        if isinstance(overscan, tuple):
            overscan = aa.Region1D(region=overscan)

        self.prescan = prescan
        self.overscan = overscan

    def regions_array_1d_from(self, array: aa.Array1D) -> aa.Array1D:
        """
        Extract all of the charge-injection regions from an input `array1D` object and returns them as a new `array1D`
        object where these extracted regions are included and all other entries are zeros.

        The diagram below illustrates the regions that are extracted from the input array:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCDPhase panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
        [xxxxxxxxxx]

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][xxxxxxxxxxxxxxxxxxxxx][sss]
          [...][ccccccccccccccccccccc][sss]
        | [...][ccccccccccccccccccccc][sss]    |
        | [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | Direction
        P [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | of
        | [...][ccccccccccccccccccccc][sss]    | clocking
          [...][ccccccccccccccccccccc][sss]    |

        []     [=====================]
               <---------S----------

        The extracted array keeps just the charge injection region and replaces all other values with 0s:

               [000000000000000000000]
               [000000000000000000000]
          [000][000000000000000000000][000]
          [000][ccccccccccccccccccccc][000]
        | [000][ccccccccccccccccccccc][000]    |
        | [000][000000000000000000000][000]    | Direction
        P [000][000000000000000000000][000]    | of
        | [000][ccccccccccccccccccccc][000]    | clocking
          [000][ccccccccccccccccccccc][000]    |

        []     [=====================]
               <---------S----------

        """

        array_1d_of_regions = array.native.copy() * 0.0

        for region in self.region_list:

            array_1d_of_regions[region.slice] += array.native[region.slice]

        return array_1d_of_regions

    def non_regions_array_1d_from(self, array: aa.Array1D) -> aa.Array1D:
        """
        Extract all of the data values in an input `array1D` that do not overlap the charge injection regions. This
        includes many areas of the image (e.g. the serial prescan, serial overscan) but is typically used to extract
        a `array1D` that contains the parallel trails that follow the charge-injection regions.

        The diagram below illustrates the `array1D` that is extracted from the input array:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCDPhase panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
        [xxxxxxxxxx]                [tttttttttt] = parallel / serial charge injection region trail

        P = Parallel Direction      S = Serial Direction

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
               <---------S----------

        The extracted array keeps just the trails following all charge injection scans and replaces all other
        values with 0s:

               [tptpptptptpptpptpptpt]
               [tptptptpptpttptptptpt]
          [000][ttttttttttttttttttttt][000]
          [000][000000000000000000000][000]
        | [000][000000000000000000000][000]    |
        | [000][ttttttttttttttttttttt][000]    | Direction
        P [000][ttttttttttttttttttttt][000]    | of
        | [000][000000000000000000000][000]    | clocking
          [000][000000000000000000000][000]    |

        []     [=====================]
               <---------S----------
        """

        array_1d_non_regions_ci = array.native.copy()

        for region in self.region_list:
            array_1d_non_regions_ci[region.slice] = 0.0

        return array_1d_non_regions_ci

    def array_1d_of_edges_and_epers_from(
        self,
        array: aa.Array1D,
        fpr_pixels: Tuple[int, int] = None,
        trails_pixels: Tuple[int, int] = None,
    ) -> aa.Array1D:
        """
        Extract all of the data values in an input `array1D` corresponding to the parallel front edges and trails of
        each the charge-injection region.

        One can specify the range of rows that are extracted, for example:

        fpr_pixels = (0, 1) will extract just the first leading front edge row.
        fpr_pixels = (0, 2) will extract the leading two front edge rows.
        trails_pixels = (0, 1) will extract the first row of trails closest to the charge injection region.

        The diagram below illustrates the arrays that are extracted from the input array for `fpr_pixels=(0,1)`
        and `trails_pixels=(0,1)`:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCDPhase panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
        [xxxxxxxxxx]                [tttttttttt] = parallel / serial charge injection region trail

        P = Parallel Direction      S = Serial Direction

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
               <---------S----------

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
               <---------S----------

        Parameters
        ------------
        fpr_pixels
            The row indexes to extract the front edge between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows).
        trails_pixels
            The row indexes to extract the trails between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows)
        """
        array_1d_of_edges_and_epers = array.native.copy() * 0.0

        if fpr_pixels is not None:

            array_1d_of_edges_and_epers = self.extract_fpr.add_to_array(
                new_array=array_1d_of_edges_and_epers, array=array, pixels=fpr_pixels
            )

        if trails_pixels is not None:

            array_1d_of_edges_and_epers = self.extract_eper.add_to_array(
                new_array=array_1d_of_edges_and_epers, array=array, pixels=trails_pixels
            )

        return array_1d_of_edges_and_epers
