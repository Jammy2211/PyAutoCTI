"""
File: python/VIS_CTICalibrate/LinePattern.py

Created on: 02/14/18
Author: James Nightingale
"""

import numpy as np
from autocti import exc
from autoarray.structures.arrays.one_d import array_1d
from autoarray.layout import region as reg

from typing import List, Tuple


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
            The row indexes to extract the front edge between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows)
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

    def stacked_array_1d_from(self, array: array_1d.Array1D, rows):
        front_arrays = self.array_1d_list_from(array=array, pixels=rows)
        return np.ma.mean(np.ma.asarray(front_arrays), axis=0)

    def binned_array_1d_from(self, array: array_1d.Array1D, rows):
        front_stacked_array = self.stacked_array_1d_from(array=array, rows=rows)
        return np.ma.mean(np.ma.asarray(front_stacked_array), axis=1)

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
            The row indexes to extract the front edge between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows)
        """
        return list(
            map(
                lambda region: region.front_edge_region_from(pixels=pixels),
                self.region_list,
            )
        )

    def add_to_array(self, new_array, array, rows):

        region_list = [
            region.front_edge_region_from(pixels=rows) for region in self.region_list
        ]

        array_1d_list = self.array_1d_list_from(array=array, pixels=rows)

        for arr, region in zip(array_1d_list, region_list):
            new_array[region.y0 : region.y1, region.x0 : region.x1] += arr

        return new_array


class Extractor1DTrails(Extractor1D):
    def array_1d_list_from(self, array: array_1d.Array1D, rows):
        """
        Extract the parallel trails of a charge injection array.


        The diagram below illustrates the arrays that is extracted from a array for rows=(0, 1):

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
        rows : (int, int)
            The row indexes to extract the trails between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows)
        """

        trails_region_list = self.region_list_from(rows=rows)
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

    def stacked_array_1d_from(self, array: array_1d.Array1D, rows):
        trails_arrays = self.array_1d_list_from(array=array, rows=rows)
        return np.ma.mean(np.ma.asarray(trails_arrays), axis=0)

    def binned_array_1d_from(self, array: array_1d.Array1D, rows):
        trails_stacked_array = self.stacked_array_1d_from(array=array, rows=rows)
        return np.ma.mean(np.ma.asarray(trails_stacked_array), axis=1)

    def region_list_from(self, rows):
        """
        Returns the parallel scans of a charge injection array.

            The diagram below illustrates the region that is calculated from a array for rows=(0, 1):

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
            rows : (int, int)
                The row indexes to extract the trails between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows)
        """

        return list(
            map(
                lambda ci_region: ci_region.parallel_trails_region_from(rows=rows),
                self.region_list,
            )
        )

    def add_to_array(self, new_array, array, rows):

        region_list = [
            region.parallel_trails_region_from(rows=rows) for region in self.region_list
        ]

        array_1d_list = self.array_1d_list_from(array=array, rows=rows)

        for arr, region in zip(array_1d_list, region_list):
            new_array[region.y0 : region.y1, region.x0 : region.x1] += arr

        return new_array


# class PatternLine(object):
#     def __init__(self, normalization: float, region_list: List[region.Region1D]):
#         """
#         Class for the pattern of a `Line` data structure, which defines the 1D regions the line charge appears
#         on a CTI line dataset, e input normalization and other properties.
#
#         Parameters
#         -----------
#         normalization
#             The normalization of the line.
#         region_list:
#             A list of the integer coordinates specifying the (x0, x1) left and right pixel coordinates of each line
#             region.
#         """
#         self.normalization = normalization
#         self.region_list = list(map(region.Region2D, region_list))
#
#     def check_pattern_is_within_image_dimensions(self, dimensions: Tuple[int]):
#
#         for region in self.region_list:
#
#             if region.x1 > dimensions[1]:
#                 raise exc.PatternLineException(
#                     "The line pattern_line regions are bigger than the image image_shape"
#                 )
#
#     @property
#     def total_pixels_min(self) -> int:
#         return np.min(list(map(lambda region: region.total_pixels, self.region_list)))
#
#     @property
#     def pixels_between_regions(self) -> List[int]:
#         return [
#             self.region_list[i + 1].x0 - self.region_list[i].x1
#             for i in range(len(self.region_list) - 1)
#         ]
#
#     def pre_cti_line_from(
#         self, shape_native: Tuple[int], pixel_scales: Tuple[float]
#     ) -> array_1d.Array1D:
#         """
#         Use this `PatternLine` pattern to generate a pre-cti line `Array1D`. This is performed by going to its each
#         line region and adding the line normalization value.
#
#         Parameters
#         -----------
#         shape_native
#             The native shape of the of the pre-cti-line `Array1D` which is created.
#         """
#
#         self.check_pattern_is_within_image_dimensions(shape_native)
#
#         pre_cti_line = np.zeros(shape_native)
#
#         for region in self.region_list:
#             pre_cti_line[region.slice] += self.normalization
#
#         return array_1d.Array1D(
#             array=pre_cti_line, pattern_line=self, pixel_scales=pixel_scales
#         )


def regions_Line_from(
    injection_on: int,
    injection_off: int,
    injection_total: int,
    parallel_size: int,
    serial_size: int,
    serial_prescan_size: int,
    serial_overscan_size: int,
    roe_corner: (int, int),
):

    regions_Line = []

    injection_start_count = 0

    for index in range(injection_total):

        if roe_corner == (0, 0):

            Line_region = (
                parallel_size - (injection_start_count + injection_on),
                parallel_size - injection_start_count,
                serial_prescan_size,
                serial_size - serial_overscan_size,
            )

        elif roe_corner == (1, 0):

            Line_region = (
                injection_start_count,
                injection_start_count + injection_on,
                serial_prescan_size,
                serial_size - serial_overscan_size,
            )

        elif roe_corner == (0, 1):

            Line_region = (
                parallel_size - (injection_start_count + injection_on),
                parallel_size - injection_start_count,
                serial_overscan_size,
                serial_size - serial_prescan_size,
            )

        elif roe_corner == (1, 1):

            Line_region = (
                injection_start_count,
                injection_start_count + injection_on,
                serial_overscan_size,
                serial_size - serial_prescan_size,
            )

        regions_Line.append(Line_region)

        injection_start_count += injection_on + injection_off

    return regions_Line
