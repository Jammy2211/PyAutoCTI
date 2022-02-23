import numpy as np
from typing import Tuple

import autoarray as aa

from autocti.extract.one_d.fpr import Extract1DFPR
from autocti.extract.one_d.eper import Extract1DEPER

from autocti import exc


class Layout1DLine(aa.Layout1D):
    def __init__(self, shape_1d: Tuple[int], region_list, prescan=None, overscan=None):
        """
        Abstract base class for a charge injection layout_ci, which defines the regions charge injections appears \
         on a charge-injection array, the input normalization and other properties.

        Parameters
        -----------
        region_list: [(int,)]
            A list of the integer coordinates specifying the corners of each charge injection region \
            (top-row, bottom-row, left-column, right-column).
        """

        self.region_list = list(map(aa.Region1D, region_list))

        for region in self.region_list:

            if region.x1 > shape_1d[0]:
                raise exc.LayoutException(
                    "The charge injection layout_ci regions are bigger than the image image_shape"
                )

        self.extract_front_edge = Extract1DFPR(region_list=region_list)
        self.extract_trails = Extract1DEPER(region_list=region_list)

        super().__init__(shape_1d=shape_1d, prescan=prescan, overscan=overscan)

    @property
    def pixels_between_regions(self):
        return [
            self.region_list[i + 1].x0 - self.region_list[i].x1
            for i in range(len(self.region_list) - 1)
        ]

    @property
    def trail_size_to_array_edge(self):
        return self.shape_1d[0] - np.max([region.x1 for region in self.region_list])

    @property
    def smallest_trails_pixels_to_array_edge(self):

        pixels_between_regions = self.pixels_between_regions
        pixels_between_regions.append(self.trail_size_to_array_edge)
        return np.min(pixels_between_regions)

    def array_1d_of_regions_from(self, array: aa.Array1D) -> aa.Array1D:
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

    def array_1d_of_non_regions_from(self, array: aa.Array1D) -> aa.Array1D:
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

    def array_1d_of_trails_from(self, array: aa.Array1D) -> aa.Array1D:
        """
        Extract all of the data values in an input `array1D` that do not overlap the charge injection regions or the
        serial prescan / serial overscan regions.

        This  extracts a `array1D` that contains only regions of the data where there are parallel trails (e.g. those
        that follow the charge-injection regions).

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

        array_1d_trails = self.array_1d_of_non_regions_from(array=array)

        array_1d_trails.native[self.prescan.slice] = 0.0
        array_1d_trails.native[self.overscan.slice] = 0.0

        return array_1d_trails

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

            array_1d_of_edges_and_epers = self.extract_front_edge.add_to_array(
                new_array=array_1d_of_edges_and_epers, array=array, pixels=fpr_pixels
            )

        if trails_pixels is not None:

            array_1d_of_edges_and_epers = self.extract_trails.add_to_array(
                new_array=array_1d_of_edges_and_epers, array=array, pixels=trails_pixels
            )

        return array_1d_of_edges_and_epers

    def extract_line_from(self, array: aa.Array1D, line_region: str):

        if line_region == "front_edge":
            return self.extract_front_edge.stacked_array_1d_from(
                array=array, pixels=(0, self.extract_front_edge.total_pixels_min)
            )
        elif line_region == "trails":
            return self.extract_trails.stacked_array_1d_from(
                array=array, pixels=(0, self.smallest_trails_pixels_to_array_edge)
            )
        else:
            raise exc.PlottingException(
                "The line region specified for the plotting of a line was invalid"
            )
