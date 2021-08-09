"""
File: python/VIS_CTICalibrate/LinePattern.py

Created on: 02/14/18
Author: James Nightingale
"""

import numpy as np
from autocti import exc
from autoarray.structures.arrays.one_d import array_1d
from autoarray.layout import region as reg
from autoarray.layout import layout as lo

from typing import Tuple


class Extractor1D:
    def __init__(self, region_list):

        self.region_list = list(map(reg.Region1D, region_list))

    @property
    def total_pixels_min(self):
        return np.min([region.total_pixels for region in self.region_list])


class Extractor1DFrontEdge(Extractor1D):
    def array_1d_list_from(self, array: array_1d.Array1D, pixels):
        """
        Extract a list of the front edges of a 1D line array.

        The diagram below illustrates the arrays that is extracted from a array for pixels=(0, 1):

        -> Direction of Clocking

        [fffcccccccccccttt]

        The extracted array keeps just the front edges corresponding to the `f` entries.
        Parameters
        ------------
        array
            A 1D array of data containing a CTI line.
        pixels
            The row indexes to extract the front edge between (e.g. pixels(0, 3) extracts the 1st, 2nd and 3rd pixels)
        """
        front_region_list = self.region_list_from(pixels=pixels)
        front_array_list = list(
            map(lambda region: array.native[region.slice], front_region_list)
        )
        front_mask_list = list(
            map(lambda region: array.mask[region.slice], front_region_list)
        )
        front_array_list = list(
            map(
                lambda front_array, front_mask: np.ma.array(
                    front_array, mask=front_mask
                ),
                front_array_list,
                front_mask_list,
            )
        )
        return front_array_list

    def stacked_array_1d_from(self, array: array_1d.Array1D, pixels):
        front_arrays = self.array_1d_list_from(array=array, pixels=pixels)
        return np.ma.mean(np.ma.asarray(front_arrays), axis=0)

    def region_list_from(self, pixels):
        """
        Calculate a list of the front edge regions of a line dataset

        The diagram below illustrates the region that calculated from a array for pixels=(0, 1):

        -> Direction of Clocking

        [fffcccccccccccttt]

        The extracted array keeps just the front edges of all regions.

        Parameters
        ------------
        array
            A 1D array of data containing a CTI line.
        pixels
            The row indexes to extract the front edge between (e.g. pixels(0, 3) extracts the 1st, 2nd and 3rd pixels)
        """
        return list(
            map(
                lambda region: region.front_edge_region_from(pixels=pixels),
                self.region_list,
            )
        )

    def add_to_array(self, new_array, array, pixels):

        region_list = [
            region.front_edge_region_from(pixels=pixels) for region in self.region_list
        ]

        array_1d_list = self.array_1d_list_from(array=array, pixels=pixels)

        for arr, region in zip(array_1d_list, region_list):
            new_array[region.x0 : region.x1] += arr

        return new_array


class Extractor1DTrails(Extractor1D):
    def array_1d_list_from(self, array: array_1d.Array1D, pixels):
        """
        Extract the parallel trails of a charge injection array.


        The diagram below illustrates the arrays that is extracted from a array for pixels=(0, 1):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCDPhase panel    [pppppppppp] = parallel overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicates ci region index)
        [xxxxxxxxxx]
        [t#t#t#t#t#] = parallel / serial charge injection region trail (0 / 1 indicates ci region index)

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][t1t1t1t1t1t1t1t1t1t1t][sss]
          [...][c1c1cc1c1cc1cc1ccc1cc][sss]
        | [...][1c1c1cc1c1cc1ccc1cc1][sss]    |
        | [...][t0t0t0t0t0t0t0t0t0t0t][sss]    | Direction
        P [...][0t0t0t0t0t0t0t0t0t0t0][sss]    | of
        | [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
          [...][cc0ccc0cccc0cccc0cccc][sss]    |

        []     [=====================]
               <---------S----------

        The extracted array keeps just the trails following all charge injection scans:

        list index 0:

        [t0t0t0tt0t0t0t0t0t0t0]

        list index 1:

        [1t1t1t1t1t1t1t1t1t1t1]

        Parameters
        ------------
        array
        pixels
            The row indexes to extract the trails between (e.g. pixels(0, 3) extracts the 1st, 2nd and 3rd pixels)
        """

        trails_region_list = self.region_list_from(pixels=pixels)
        trails_arrays = list(
            map(lambda region: array.native[region.slice], trails_region_list)
        )
        trails_masks = list(
            map(lambda region: array.mask[region.slice], trails_region_list)
        )
        trails_arrays = list(
            map(
                lambda trails_array, front_mask: np.ma.array(
                    trails_array, mask=front_mask
                ),
                trails_arrays,
                trails_masks,
            )
        )
        return trails_arrays

    def stacked_array_1d_from(self, array: array_1d.Array1D, pixels):
        trails_arrays = self.array_1d_list_from(array=array, pixels=pixels)
        return np.ma.mean(np.ma.asarray(trails_arrays), axis=0)

    def region_list_from(self, pixels):
        """
        Returns the parallel scans of a charge injection array.

            The diagram below illustrates the region that is calculated from a array for pixels=(0, 1):

            ---KEY---
            ---------

            [] = read-out electronics   [==========] = read-out register

            [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
            [xxxxxxxxxx] = CCDPhase panel    [pppppppppp] = parallel overscan
            [c#cc#c#c#c] = charge injection region (0 / 1 indicates ci region index)
            [xxxxxxxxxx]
            [t#t#t#t#t#] = parallel / serial charge injection region trail (0 / 1 indicates ci region index)

            P = Parallel Direction      S = Serial Direction

                   [ppppppppppppppppppppp]
                   [ppppppppppppppppppppp]
              [...][t1t1t1t1t1t1t1t1t1t1t][sss]
              [...][c1c1cc1c1cc1cc1ccc1cc][sss]
            | [...][1c1c1cc1c1cc1ccc1cc1][sss]    |
            | [...][t0t0t0t0t0t0t0t0t0t0t][sss]    | Direction
            P [...][0t0t0t0t0t0t0t0t0t0t0][sss]    | of
            | [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
              [...][cc0ccc0cccc0cccc0cccc][sss]    |

            []     [=====================]
                   <---------S----------

            The extracted array keeps just the trails following all charge injection scans:

            list index 0:

            [2, 4, 3, 21] (serial prescan is 3 pixels)

            list index 1:

            [6, 7, 3, 21] (serial prescan is 3 pixels)

            Parameters
            ------------
            arrays
            pixels
                The row indexes to extract the trails between (e.g. pixels(0, 3) extracts the 1st, 2nd and 3rd pixels)
        """

        return list(
            map(
                lambda region: region.trails_region_from(pixels=pixels),
                self.region_list,
            )
        )

    def add_to_array(self, new_array, array, pixels):

        region_list = [
            region.trails_region_from(pixels=pixels) for region in self.region_list
        ]

        array_1d_list = self.array_1d_list_from(array=array, pixels=pixels)

        for arr, region in zip(array_1d_list, region_list):
            new_array[region.x0 : region.x1] += arr

        return new_array


class AbstractLayout1DLine(lo.Layout1D):
    def __init__(
        self, shape_1d, normalization, region_list, prescan=None, overscan=None
    ):
        """
        Abstract base class for a charge injection layout_ci, which defines the regions charge injections appears \
         on a charge-injection array, the input normalization and other properties.

        Parameters
        -----------
        normalization : float
            The normalization of the charge injection lines.
        region_list: [(int,)]
            A list of the integer coordinates specifying the corners of each charge injection region \
            (top-row, bottom-row, left-column, right-column).
        """

        self.region_list = list(map(reg.Region1D, region_list))

        for region in self.region_list:

            if region.x1 > shape_1d[0]:
                raise exc.LayoutException(
                    "The charge injection layout_ci regions are bigger than the image image_shape"
                )

        self.extractor_front_edge = Extractor1DFrontEdge(region_list=region_list)
        self.extractor_trails = Extractor1DTrails(region_list=region_list)

        super().__init__(shape_1d=shape_1d, prescan=prescan, overscan=overscan)

        self.normalization = normalization

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

        rows_between_regions = self.pixels_between_regions
        rows_between_regions.append(self.trail_size_to_array_edge)
        return np.min(rows_between_regions)

    def array_1d_of_regions_from(self, array: array_1d.Array1D) -> array_1d.Array1D:
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

    def array_1d_of_non_regions_from(self, array: array_1d.Array1D) -> array_1d.Array1D:
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

    def array_1d_of_trails_from(self, array: array_1d.Array1D) -> array_1d.Array1D:
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

    def array_1d_of_edges_and_trails_from(
        self,
        array: array_1d.Array1D,
        front_edge_rows: Tuple[int, int] = None,
        trails_rows: Tuple[int, int] = None,
    ) -> array_1d.Array1D:
        """
        Extract all of the data values in an input `array1D` corresponding to the parallel front edges and trails of
        each the charge-injection region.

        One can specify the range of rows that are extracted, for example:

        front_edge_rows = (0, 1) will extract just the first leading front edge row.
        front_edge_rows = (0, 2) will extract the leading two front edge rows.
        trails_rows = (0, 1) will extract the first row of trails closest to the charge injection region.

        The diagram below illustrates the arrays that are extracted from the input array for `front_edge_rows=(0,1)`
        and `trails_rows=(0,1)`:

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
        front_edge_rows
            The row indexes to extract the front edge between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows).
        trails_rows
            The row indexes to extract the trails between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows)
        """
        array_1d_of_edges_and_trails = array.native.copy() * 0.0

        if front_edge_rows is not None:

            array_1d_of_edges_and_trails = self.extractor_front_edge.add_to_array(
                new_array=array_1d_of_edges_and_trails,
                array=array,
                pixels=front_edge_rows,
            )

        if trails_rows is not None:

            array_1d_of_edges_and_trails = self.extractor_trails.add_to_array(
                new_array=array_1d_of_edges_and_trails, array=array, pixels=trails_rows
            )

        return array_1d_of_edges_and_trails

    def extract_line_from(self, array: array_1d.Array1D, line_region: str):

        if line_region == "front_edge":
            return self.extractor_front_edge.stacked_array_1d_from(
                array=array, pixels=(0, self.extractor_front_edge.total_pixels_min)
            )
        elif line_region == "trails":
            return self.extractor_trails.stacked_array_1d_from(
                array=array, pixels=(0, self.smallest_trails_pixels_to_array_edge)
            )
        else:
            raise exc.PlottingException(
                "The line region specified for the plotting of a line was invalid"
            )


class Layout1DLine(AbstractLayout1DLine):
    """
    A uniform charge injection layout_ci, which is defined by the regions it appears on the charge injection \
    array and its normalization.
    """

    def pre_cti_data_from(self, shape_native: Tuple[int,], pixel_scales):
        """Use this charge injection layout_ci to generate a pre-cti charge injection image. This is performed by \
        going to its charge injection regions and adding the charge injection normalization value.

        Parameters
        -----------
        shape_native
            The image_shape of the pre_cti_images to be created.
        """

        pre_cti_image = np.zeros(shape_native)

        for region in self.region_list:
            pre_cti_image[region.slice] += self.normalization

        return array_1d.Array1D.manual_native(
            array=pre_cti_image, pixel_scales=pixel_scales
        )
