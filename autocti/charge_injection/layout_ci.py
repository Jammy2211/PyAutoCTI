"""
File: python/VIS_CTICalibrate/ChargeInjectPattern.py

Created on: 02/14/18
Author: James Nightingale
"""

from autocti import exc
from autoarray.structures.arrays.two_d import array_2d
from autoarray.layout import layout_util
from autoarray.layout import layout as lo
from autoarray.layout import region as reg

from copy import deepcopy

import numpy as np
from autocti.charge_injection import mask_2d_ci

from typing import Tuple


class AbstractLayout2DCI(lo.Layout2D):
    def __init__(
        self,
        shape_2d,
        normalization,
        region_list,
        original_roe_corner=(1, 0),
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
    ):
        """
        Abstract base class for a charge injection layout_ci, which defines the regions charge injections appears \
         on a charge-injection array_ci, the input normalization and other properties.

        Parameters
        -----------
        normalization : float
            The normalization of the charge injection lines.
        regions: [(int,)]
            A list of the integer coordinates specifying the corners of each charge injection region \
            (top-row, bottom-row, left-column, right-column).
        """

        self.region_list = list(map(reg.Region2D, region_list))

        for region in self.region_list:

            if region.y1 > shape_2d[0] or region.x1 > shape_2d[1]:
                raise exc.Layout2DCIException(
                    "The charge injection layout_ci regions are bigger than the image image_shape"
                )

        super().__init__(
            shape_2d=shape_2d,
            original_roe_corner=original_roe_corner,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

        self.normalization = normalization

    def after_extraction(self, extraction_region):

        layout = super().after_extraction(extraction_region=extraction_region)

        region_list = [
            layout_util.region_after_extraction(
                original_region=region, extraction_region=extraction_region
            )
            for region in self.region_list
        ]

        return self.__class__(
            original_roe_corner=self.original_roe_corner,
            shape_2d=self.shape_2d,
            normalization=self.normalization,
            region_list=region_list,
            parallel_overscan=layout.parallel_overscan,
            serial_prescan=layout.serial_prescan,
            serial_overscan=layout.serial_overscan,
        )

    @property
    def total_rows_min(self):
        return np.min(list(map(lambda region: region.total_rows, self.region_list)))

    @property
    def total_columns_min(self):
        return np.min(list(map(lambda region: region.total_columns, self.region_list)))

    @property
    def rows_between_regions(self):
        return [
            self.region_list[i + 1].y0 - self.region_list[i].y1
            for i in range(len(self.region_list) - 1)
        ]

    def array_of_regions_ci_from(self, array: array_2d.Array2D) -> array_2d.Array2D:
        """
        Extract all of the charge-injection regions from an input `array2D` object and returns them as a new `array2D`
        object where these extracted regions are included and all other entries are zeros.

        The diagram below illustrates the regions that are extracted from the input array:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
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

        The extracted array_ci keeps just the charge injection region and replaces all other values with 0s:

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

        new_array = array.copy() * 0.0

        for region in self.region_list:
            new_array[region.slice] += array[region.slice]

        return new_array

    def array_of_non_regions_ci_from(self, array: array_2d.Array2D) -> array_2d.Array2D:
        """
        Extract all of the data values in an input `array2D` that do not overlap the charge injection regions. This
        includes many areas of the image (e.g. the serial prescan, serial overscan) but is typically used to extract
        a `array2D` that contains the parallel trails that follow the charge-injection regions.

        The diagram below illustrates the `array2D` that is extracted from the input array:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
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

        The extracted array_ci keeps just the trails following all charge injection scans and replaces all other
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

        non_regions_ci_array = array.copy()

        for region in self.regions:
            non_regions_ci_array[region.slice] = 0.0

        return non_regions_ci_array

    def array_of_parallel_trails_from(
        self, array: array_2d.Array2D
    ) -> array_2d.Array2D:
        """
        Extract all of the data values in an input `array2D` that do not overlap the charge injection regions or the
        serial prescan / serial overscan regions.

        This  extracts a `array2D` that contains only regions of the data where there are parallel trails (e.g. those
        that follow the charge-injection regions).

        The diagram below illustrates the `array2D` that is extracted from the input array:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
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

        The extracted array_ci keeps just the trails following all charge injection scans and replaces all other
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

        parallel_array = self.array_of_non_regions_ci_from(array=array)

        parallel_array[array.layout.serial_prescan.slice] = 0.0
        parallel_array[array.layout.serial_overscan.slice] = 0.0

        return parallel_array

    def array_of_parallel_edges_and_trails_from(
        self,
        array: array_2d.Array2D,
        front_edge_rows: Tuple[int, int] = None,
        trails_rows: Tuple[int, int] = None,
    ) -> array_2d.Array2D:
        """
        Extract all of the data values in an input `array2D` corresponding to the parallel front edges and trails of
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
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
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

        The extracted array_ci keeps just the leading edges and trails following all charge injection scans and
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
        new_array = array.copy() * 0.0

        if front_edge_rows is not None:

            front_regions = list(
                map(
                    lambda ci_region: ci_region.parallel_front_edge_region_from(
                        rows=front_edge_rows
                    ),
                    self.regions,
                )
            )

            front_edges = self.extract_parallel_front_edge_arrays_from(
                array=array, rows=front_edge_rows
            )

            for i, region in enumerate(front_regions):
                new_array[region.y0 : region.y1, region.x0 : region.x1] += front_edges[
                    i
                ]

        if trails_rows is not None:

            trails_regions = list(
                map(
                    lambda ci_region: ci_region.parallel_front_edge_region_from(
                        rows=trails_rows
                    ),
                    self.regions,
                )
            )

            trails = self.extract_parallel_trails_arrays_from(
                array=array, rows=trails_rows
            )

            for i, region in enumerate(trails_regions):
                new_array[region.y0 : region.y1, region.x0 : region.x1] += trails[i]

        return new_array

    def array_for_parallel_calibration_from(
        self, array: array_2d.Array2D, columns: Tuple[int, int]
    ):
        """
        Extract a parallel calibration array from an input array, where this array contains a sub-set of the input
        array which is specifically used for only parallel CTI calibration. This array is simply a specified number
        of columns that are closest to the read-out electronics.

        The diagram below illustrates the arrays that is extracted from a array_ci with columns=(0, 3):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
        [xxxxxxxxxx]                [tttttttttt] = parallel / parallel charge injection region trail

        P = Parallel Direction      S = Serial Direction

               [ptptptptptptptptptptp]
               [tptptptptptptptptptpt]
          [...][xxxxxxxxxxxxxxxxxxxxx][sss]
          [...][ccccccccccccccccccccc][sss]
        | [...][ccccccccccccccccccccc][sts]    |
        | [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | Direction
        P [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | of
        | [...][ccccccccccccccccccccc][sss]    | clocking
          [...][ccccccccccccccccccccc][sss]    |

        []     [=====================]
               <---------S----------

        The extracted array_ci keeps just the trails following all charge injection scans and replaces all other
        values with 0s:

               [ptp]
               [tpt]
               [xxx]
               [ccc]
        |      [ccc]                           |
        |      [xxx]                           | Direction
        P      [xxx]                           | of
        |      [ccc]                           | clocking
               [ccc]                           |

        []     [=====================]
               <---------S----------
        """
        extraction_region = self.region_list[
            0
        ].parallel_side_nearest_read_out_region_from(
            shape_2d=self.shape_2d, columns=columns
        )
        return array_2d.Array2D.manual_native(
            array=array.native[extraction_region.slice],
            exposure_info=array.exposure_info,
            layout=self.after_extraction(extraction_region=extraction_region),
            pixel_scales=array.pixel_scales,
        )

    def mask_for_parallel_calibration_from(self, mask, columns):
        extraction_region = self.region_list[
            0
        ].parallel_side_nearest_read_out_region_from(
            shape_2d=self.shape_2d, columns=columns
        )
        return mask_2d_ci.Mask2DCI(
            mask=mask[extraction_region.slice], pixel_scales=mask.pixel_scales
        )

    def extract_parallel_front_edge_arrays_from(
        self, array: array_2d.Array2D, rows=None
    ):
        """
        Extract a list of structures of the parallel front edge scans of a charge injection array_ci.

        The diagram below illustrates the arrays that is extracted from a array_ci for rows=(0, 1):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicates ci_region index)
        [xxxxxxxxxx]                [tttttttttt] = parallel / serial charge injection region trail

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][ttttttttttttttttttttt][sss]
          [...][c1c1cc1c1cc1cc1ccc1cc][sss]
        | [...][1c1c1cc1c1cc1ccc1cc1][sss]    |
        | [...][ttttttttttttttttttttt][sss]    | Direction
        P [...][ttttttttttttttttttttt][sss]    | of
        | [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
          [...][cc0ccc0cccc0cccc0cccc][sss]    |

        []     [=====================]
               <---------S----------

        The extracted array_ci keeps just the front edges of all charge injection scans.

        list index 0:

        [c0c0c0cc0c0c0c0c0c0c0]

        list index 1:

        [1c1c1c1c1c1c1c1c1c1c1]

        Parameters
        ------------
        array
        rows : (int, int)
            The row indexes to extract the front edge between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows)
        """
        front_regions = self.parallel_front_edge_regions(rows=rows)
        front_arrays = list(map(lambda region: array[region.slice], front_regions))
        front_masks = list(map(lambda region: array.mask[region.slice], front_regions))
        front_arrays = list(
            map(
                lambda front_array, front_mask: np.ma.array(
                    front_array, mask=front_mask
                ),
                front_arrays,
                front_masks,
            )
        )
        return front_arrays

    def extract_parallel_trails_arrays_from(self, array: array_2d.Array2D, rows=None):
        """Extract the parallel trails of a charge injection array_ci.


        The diagram below illustrates the arrays that is extracted from a array_ci for rows=(0, 1):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan
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

        The extracted array_ci keeps just the trails following all charge injection scans:

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

        trails_regions = self.parallel_trails_regions(array=array, rows=rows)
        trails_arrays = list(map(lambda region: array[region.slice], trails_regions))
        trails_masks = list(
            map(lambda region: array.mask[region.slice], trails_regions)
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

    def extract_serial_trails_array_from(self, array: array_2d.Array2D):
        """Extract an arrays of all of the serial trails in the serial overscan region, that are to the side of a
        charge-injection scans from a charge injection array_ci.

        The diagram below illustrates the arrays that is extracted from a array_ci:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
        [xxxxxxxxxx]                [tttttttttt] = parallel / serial charge injection region trail

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][xxxxxxxxxxxxxxxxxxxxx][sss]
          [...][ccccccccccccccccccccc][tst]
        | [...][ccccccccccccccccccccc][sts]    |
        | [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | Direction
        P [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | of
        | [...][ccccccccccccccccccccc][tst]    | clocking
          [...][ccccccccccccccccccccc][sts]    |

        []     [=====================]
               <---------S----------

        The extracted array_ci keeps just the trails following all charge injection scans and replaces all other
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
               <---------S----------
        """
        array = self.serial_edges_and_trails_array(
            array=array, trails_columns=(0, array.layout.serial_overscan.total_columns)
        )
        return array

    def serial_overscan_no_trails_array_from(self, array: array_2d.Array2D):
        """
        Extract an arrays of all of the scans of the serial overscan that don't contain trails from a
        charge injection region (i.e. are not to the side of one).

        The diagram below illustrates the arrays that is extracted from a array_ci:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
        [xxxxxxxxxx]                [tttttttttt] = parallel / serial charge injection region trail

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][xxxxxxxxxxxxxxxxxxxxx][sss]
          [...][ccccccccccccccccccccc][tst]
        | [...][ccccccccccccccccccccc][sts]    |
        | [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | Direction
        P [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | of
        | [...][ccccccccccccccccccccc][tst]    | clocking
          [...][ccccccccccccccccccccc][sts]    |

        []     [=====================]
               <---------S----------

        The extracted array_ci keeps just the trails following all charge injection scans and replaces all other
        values with 0s:

               [000000000000000000000]
               [000000000000000000000]
          [000][000000000000000000000][sss]
          [000][000000000000000000000][000]
        | [000][000000000000000000000][000]    |
        | [000][000000000000000000000][sss]    | Direction
        P [000][000000000000000000000][sss]    | of
        | [000][000000000000000000000][000]    | clocking
          [000][000000000000000000000][000]    |

        []     [=====================]
               <---------S----------
        """
        new_array = array.copy() * 0.0
        overscan_slice = array.layout.serial_overscan.slice

        new_array[overscan_slice] = array[overscan_slice]

        trails_regions = list(
            map(
                lambda ci_region: array.serial_trails_of_region(
                    ci_region, (0, array.layout.serial_overscan.total_columns)
                ),
                self.regions,
            )
        )

        for region in trails_regions:
            new_array[region.slice] = 0

        return new_array

    def serial_edges_and_trails_array(
        self, array: array_2d.Array2D, front_edge_columns=None, trails_columns=None
    ):
        """
        Extract an arrays of all of the serial front edges and trails of each the charge-injection scans from
        a charge injection array_ci.

        One can specify the range of columns that are extracted, for example:

        front_edge_columns = (0, 1) will extract just the leading front edge column.
        front_edge_columns = (0, 2) will extract the leading two front edge columns.
        trails_columns = (0, 1) will extract the first column of trails closest to the charge injection region.

        The diagram below illustrates the arrays that is extracted from a array_ci for front_edge_columns=(0,2) and
        trails_columns=(0,2):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
        [xxxxxxxxxx]                [tttttttttt] = parallel / serial charge injection region trail

        P = Parallel Direction      S = Serial Direction

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
               <---------S----------

        The extracted array_ci keeps just the leading edge and trails following all charge injection scans and
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
               <---------S----------

        Parameters
        ------------
        array
        front_edge_columns : (int, int)
            The column indexes to extract the front edge between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd rows)
        trails_columns : (int, int)
            The column indexes to extract the trails between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd rows)
        """
        new_array = array.copy() * 0.0

        if front_edge_columns is not None:

            front_regions = list(
                map(
                    lambda ci_region: array.serial_front_edge_of_region(
                        ci_region, front_edge_columns
                    ),
                    self.regions,
                )
            )

            front_edges = self.serial_front_edge_arrays_from(columns=front_edge_columns)

            for i, region in enumerate(front_regions):
                new_array[region.y0 : region.y1, region.x0 : region.x1] += front_edges[
                    i
                ]

        if trails_columns is not None:

            trails_regions = list(
                map(
                    lambda ci_region: array.serial_trails_of_region(
                        ci_region, trails_columns
                    ),
                    self.regions,
                )
            )

            trails = self.serial_trails_arrays_from(columns=trails_columns)

            for i, region in enumerate(trails_regions):
                new_array[region.y0 : region.y1, region.x0 : region.x1] += trails[i]

        return new_array

    def serial_calibration_array_from(
        self, array: array_2d.Array2D, rows: Tuple[int, int]
    ):
        """
        Extract a serial calibration array from a charge injection array_ci, where this arrays is a sub-set of the
        array_ci which can be used for serial-only calibration. Specifically, this array_ci is all charge injection
        scans and their serial over-scan trails.

        The diagram below illustrates the arrays that is extracted from a array_ci with column=5:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
        [xxxxxxxxxx]                [tttttttttt] = parallel / serial charge injection region trail

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [pppppppppppppppppppp ]
          [...][xxxxxxxxxxxxxxxxxxxxx][sss]
          [...][ccccccccccccccccccccc][tst]
        | [...][ccccccccccccccccccccc][sts]    |
        | [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | Direction
        P [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | of
        | [...][ccccccccccccccccccccc][tst]    | clocking
          [...][ccccccccccccccccccccc][sts]    |

        []     [=====================]
               <---------S----------

        The extracted array_ci keeps just the trails following all charge injection scans and replaces all other
        values with 0s:

        |                                      |
        |      [cccccccccccccccc][tst]         | Direction
        P      [cccccccccccccccc][tst]         | of
        |      [cccccccccccccccc][tst]         | clocking
               [cccccccccccccccc][tst]         |

        []     [=====================]
               <---------S----------
        """
        calibration_images = self.serial_calibration_sub_arrays_from
        calibration_images = list(
            map(lambda image: image[rows[0] : rows[1], :], calibration_images)
        )

        array = np.concatenate(calibration_images, axis=0)

        # TODO : can we generalize this method for multiple extracts? Feels too complicated so just doing it for this
        # TODO : specific case for now.

        serial_prescan = (
            (
                0,
                array.shape[0],
                array.layout.serial_prescan[2],
                array.layout.serial_prescan[3],
            )
            if array.layout.serial_prescan is not None
            else None
        )
        serial_overscan = (
            (
                0,
                array.shape[0],
                array.layout.serial_overscan[2],
                array.layout.serial_overscan[3],
            )
            if array.layout.serial_overscan is not None
            else None
        )

        x0 = self.regions[0][2]
        x1 = self.regions[0][3]
        offset = 0
        new_pattern_regions_ci = []

        for region in self.regions:

            labelsize = rows[1] - rows[0]
            new_pattern_regions_ci.append((offset, offset + labelsize, x0, x1))
            offset += labelsize

        new_layout_ci = deepcopy(self)
        new_layout_ci.regions = new_pattern_regions_ci

        return array_2d.Array2D.manual(
            array=array,
            roe_corner=array.original_roe_corner,
            scans=abstract_array.Scans(
                parallel_overscan=None,
                serial_prescan=serial_prescan,
                serial_overscan=serial_overscan,
            ),
            pixel_scales=array.pixel_scales,
        )

    def serial_calibration_mask_from_mask_and_rows(
        self, array: array_2d.Array2D, mask, rows: Tuple[int, int]
    ):
        """
        Extract a serial calibration array from a charge injection array_ci, where this arrays is a sub-set of the
        array_ci which can be used for serial-only calibration. Specifically, this array_ci is all charge injection
        scans and their serial over-scan trails.

        The diagram below illustrates the arrays that is extracted from a array_ci with column=5:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
        [xxxxxxxxxx]                [tttttttttt] = parallel / serial charge injection region trail

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [pppppppppppppppppppp ]
          [...][xxxxxxxxxxxxxxxxxxxxx][sss]
          [...][ccccccccccccccccccccc][tst]
        | [...][ccccccccccccccccccccc][sts]    |
        | [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | Direction
        P [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | of
        | [...][ccccccccccccccccccccc][tst]    | clocking
          [...][ccccccccccccccccccccc][sts]    |

        []     [=====================]
               <---------S----------

        The extracted array_ci keeps just the trails following all charge injection scans and replaces all other
        values with 0s:

        |                                      |
        |      [cccccccccccccccc][tst]         | Direction
        P      [cccccccccccccccc][tst]         | of
        |      [cccccccccccccccc][tst]         | clocking
               [cccccccccccccccc][tst]         |

        []     [=====================]
               <---------S----------
        """

        calibration_regions = list(
            map(
                lambda ci_region: array.serial_entire_rows_of_region(region=ci_region),
                self.regions,
            )
        )
        calibration_masks = list(
            map(lambda region: mask[region.slice], calibration_regions)
        )

        calibration_masks = list(
            map(lambda mask: mask[rows[0] : rows[1], :], calibration_masks)
        )
        return mask_2d_ci.Mask2DCI(
            mask=np.concatenate(calibration_masks, axis=0),
            pixel_scales=mask.pixel_scales,
        )

    def serial_calibration_sub_arrays_from(self, array: array_2d.Array2D):
        """
        Extract each charge injection region image for the serial calibration arrays above.
        """

        calibration_regions = list(
            map(
                lambda ci_region: array.serial_entire_rows_of_region(region=ci_region),
                self.regions,
            )
        )
        return list(map(lambda region: array[region.slice], calibration_regions))

    def parallel_front_edge_line_binned_over_columns_from(
        self, array: array_2d.Array2D, rows=None
    ):
        front_stacked_array = self.parallel_front_edge_stacked_array_from(
            array=array, rows=rows
        )
        return np.ma.mean(np.ma.asarray(front_stacked_array), axis=1)

    def parallel_front_edge_stacked_array_from(
        self, array: array_2d.Array2D, rows=None
    ):
        front_arrays = self.extract_parallel_front_edge_arrays_from(
            array=array, rows=rows
        )
        return np.ma.mean(np.ma.asarray(front_arrays), axis=0)

    def parallel_trails_line_binned_over_columns_from(
        self, array: array_2d.Array2D, rows=None
    ):
        trails_stacked_array = self.parallel_trails_stacked_array_from(
            array=array, rows=rows
        )
        return np.ma.mean(np.ma.asarray(trails_stacked_array), axis=1)

    def parallel_trails_stacked_array_from(self, array: array_2d.Array2D, rows=None):
        trails_arrays = self.extract_parallel_trails_arrays_from(array=array, rows=rows)
        return np.ma.mean(np.ma.asarray(trails_arrays), axis=0)

    def serial_front_edge_line_binned_over_rows_from(
        self, array: array_2d.Array2D, columns=None
    ):
        front_stacked_array = self.serial_front_edge_stacked_array_from(
            arrays=array, columns=columns
        )
        return np.ma.mean(np.ma.asarray(front_stacked_array), axis=0)

    def serial_front_edge_stacked_array_from(
        self, array: array_2d.Array2D, columns=None
    ):
        front_arrays = self.serial_front_edge_arrays_from(array=array, columns=columns)
        return np.ma.mean(np.ma.asarray(front_arrays), axis=0)

    def serial_front_edge_arrays_from(self, array: array_2d.Array2D, columns=None):
        """
        Extract a list of the serial front edge structures of a charge injection array_ci.

        The diagram below illustrates the arrays that is extracted from a array_ci for columnss=(0, 3):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicates ci_region index)
        [xxxxxxxxxx]
        [tttttttttt] = parallel / serial charge injection region trail ((0 / 1 indicates ci_region index)

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][ttttttttttttttttttttt][sss]
          [...][c1c1cc1c1cc1cc1ccc1cc][sss]
        | [...][1c1c1cc1c1cc1ccc1cc1c][sss]    |
        | [...][ttttttttttttttttttttt][sss]    | Direction
        P [...][ttttttttttttttttttttt][sss]    | of
        | [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
          [...][cc0ccc0cccc0cccc0cccc][sss]    |

        []     [=====================]
               <---------S----------

        The extracted array_ci keeps just the serial front edges of all charge injection scans.

        list index 0:

        [c0c0]

        list index 1:

        [1c1c]

        Parameters
        ------------
        array
        columns : (int, int)
            The column indexes to extract the front edge between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd
            columns)
        """
        front_regions = self.serial_front_edge_regions(array=array, columns=columns)
        front_arrays = list(map(lambda region: array[region.slice], front_regions))
        front_masks = list(map(lambda region: array.mask[region.slice], front_regions))
        front_arrays = list(
            map(
                lambda front_array, front_mask: np.ma.array(
                    front_array, mask=front_mask
                ),
                front_arrays,
                front_masks,
            )
        )
        return front_arrays

    def serial_trails_line_binned_over_rows_from(
        self, array: array_2d.Array2D, columns: Tuple[int, int] = None
    ):
        trails_stacked_array = self.serial_trails_stacked_array_from(
            array=array, columns=columns
        )
        return np.ma.mean(np.ma.asarray(trails_stacked_array), axis=0)

    def serial_trails_stacked_array_from(
        self, array: array_2d.Array2D, columns: Tuple[int, int] = None
    ):
        front_arrays = self.serial_trails_arrays_from(array=array, columns=columns)
        return np.ma.mean(np.ma.asarray(front_arrays), axis=0)

    def serial_trails_arrays_from(self, array: array_2d.Array2D, columns=None):
        """
        Extract a list of the serial trails of a charge injection array_ci.

        The diagram below illustrates the arrays that is extracted from a array_ci for columnss=(0, 3):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicates ci_region index)
        [xxxxxxxxxx]
        [tttttttttt] = parallel / serial charge injection region trail ((0 / 1 indicates ci_region index)

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][ttttttttttttttttttttt][sss]
          [...][c1c1cc1c1cc1cc1ccc1cc][st1]
        | [...][1c1c1cc1c1cc1ccc1cc1c][ts0]    |
        | [...][ttttttttttttttttttttt][sss]    | Direction
        P [...][ttttttttttttttttttttt][sss]    | of
        | [...][0ccc0cccc0cccc0cccc0c][st1]    | clocking
          [...][cc0ccc0cccc0cccc0cccc][ts0]    |

        []     [=====================]
               <---------S----------

        The extracted array_ci keeps just the serial front edges of all charge injection scans.

        list index 0:

        [st0]

        list index 1:

        [st1]

        Parameters
        ------------
        array
        columns : (int, int)
            The column indexes to extract the trails between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd columns)
        """
        trails_regions = self.serial_trails_regions_from(array=array, columns=columns)
        trails_arrays = list(map(lambda region: array[region.slice], trails_regions))
        trails_masks = list(
            map(lambda region: array.mask[region.slice], trails_regions)
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

    def parallel_front_edge_regions(self, rows=None):
        """
        Calculate a list of the parallel front edge scans of a charge injection array_ci.

        The diagram below illustrates the region that calculaed from a array_ci for rows=(0, 1):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicates ci_region index)
        [xxxxxxxxxx]                [tttttttttt] = parallel / serial charge injection region trail

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][ttttttttttttttttttttt][sss]
          [...][c1c1cc1c1cc1cc1ccc1cc][sss]
        | [...][1c1c1cc1c1cc1ccc1cc1][sss]    |
        | [...][ttttttttttttttttttttt][sss]    | Direction
        P [...][ttttttttttttttttttttt][sss]    | of
        | [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
          [...][cc0ccc0cccc0cccc0cccc][sss]    |

        []     [=====================]
               <---------S----------

        The extracted array_ci keeps just the front edges of all charge injection scans.

        list index 0:

        [0, 1, 3, 21] (serial prescan is 3 pixels)

        list index 1:

        [3, 4, 3, 21] (serial prescan is 3 pixels)

        Parameters
        ------------
        arrays
        rows : (int, int)
            The row indexes to extract the front edge between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows)
        """
        if rows is None:
            rows = (0, self.total_rows_min)
        return list(
            map(
                lambda ci_region: ci_region.parallel_front_edge_region_from(rows=rows),
                self.regions,
            )
        )

    def parallel_trails_regions(self, rows=None):
        """
        Returns the parallel scans of a charge injection array_ci.

            The diagram below illustrates the region that is calculated from a array_ci for rows=(0, 1):

            ---KEY---
            ---------

            [] = read-out electronics   [==========] = read-out register

            [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
            [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan
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

            The extracted array_ci keeps just the trails following all charge injection scans:

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

        if rows is None:
            rows = (0, self.smallest_parallel_trails_rows_to_array_edge)

        return list(
            map(
                lambda ci_region: ci_region.parallel_trails_of_region_from(rows=rows),
                self.regions,
            )
        )

    def serial_front_edge_regions(self, array: array_2d.Array2D, columns=None):
        """
        Returns a list of the serial front edges scans of a charge injection array_ci.

        The diagram below illustrates the region that is calculated from a array_ci for columns=(0, 4):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicates ci_region index)
        [xxxxxxxxxx]
        [tttttttttt] = parallel / serial charge injection region trail ((0 / 1 indicates ci_region index)

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][ttttttttttttttttttttt][sss]
          [...][c1c1cc1c1cc1cc1ccc1cc][sss]
        | [...][1c1c1cc1c1cc1ccc1cc1c][sss]    |
        | [...][ttttttttttttttttttttt][sss]    | Direction
        P [...][ttttttttttttttttttttt][sss]    | of
        | [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
          [...][cc0ccc0cccc0cccc0cccc][sss]    |

        []     [=====================]
               <---------S----------

        The extracted array_ci keeps just the serial front edges of all charge injection scans.

        list index 0:

        [0, 2, 3, 21] (serial prescan is 3 pixels)

        list index 1:

        [4, 6, 3, 21] (serial prescan is 3 pixels)

        Parameters
        ------------
        arrays
        columns : (int, int)
            The column indexes to extract the front edge between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd
            columns)
        """
        if columns is None:
            columns = (0, self.total_columns_min)
        return list(
            map(
                lambda ci_region: array.serial_front_edge_of_region(ci_region, columns),
                self.regions,
            )
        )

    def serial_trails_regions_from(self, array: array_2d.Array2D, columns=None):
        """
        Returns a list of the serial trails scans of a charge injection array_ci.

        The diagram below illustrates the region is calculated from a array_ci for columnss=(0, 4):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicates ci_region index)
        [xxxxxxxxxx]
        [tttttttttt] = parallel / serial charge injection region trail ((0 / 1 indicates ci_region index)

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][ttttttttttttttttttttt][sss]
          [...][c1c1cc1c1cc1cc1ccc1cc][st1]
        | [...][1c1c1cc1c1cc1ccc1cc1c][ts0]    |
        | [...][ttttttttttttttttttttt][sss]    | Direction
        P [...][ttttttttttttttttttttt][sss]    | of
        | [...][0ccc0cccc0cccc0cccc0c][st1]    | clocking
          [...][cc0ccc0cccc0cccc0cccc][ts0]    |

        []     [=====================]
               <---------S----------

        The extracted array_ci keeps just the serial front edges of all charge injection scans.

        list index 0:

        [0, 2, 22, 225 (serial prescan is 3 pixels)

        list index 1:

        [4, 6, 22, 25] (serial prescan is 3 pixels)

        Parameters
        ------------
        arrays
        columns : (int, int)
            The column indexes to extract the trails between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd columns)
        """
        if columns is None:
            columns = (0, array.layout.serial_trails_columns)
        return list(
            map(
                lambda ci_region: array.serial_trails_of_region(ci_region, columns),
                self.regions,
            )
        )

    @property
    def smallest_parallel_trails_rows_to_array_edge(self):

        rows_between_regions = self.rows_between_regions
        rows_between_regions.append(self.parallel_trail_size_to_array_edge)
        return np.min(rows_between_regions)

    @property
    def parallel_trail_size_to_array_edge(self):

        return self.shape_2d[0] - np.max([region.y1 for region in self.regions])

    def with_extracted_regions(self, extraction_region):

        layout_ci = deepcopy(self)

        extracted_regions = list(
            map(
                lambda region: layout_util.region_after_extraction(
                    original_region=region, extraction_region=extraction_region
                ),
                self.regions,
            )
        )
        extracted_regions = list(filter(None, extracted_regions))
        if not extracted_regions:
            extracted_regions = None

        layout_ci.regions = extracted_regions
        return layout_ci


class Layout2DCIUniform(AbstractLayout2DCI):
    """
    A uniform charge injection layout_ci, which is defined by the regions it appears on the charge injection \
    array_ci and its normalization.
    """

    def pre_cti_ci_from(self, shape_native, pixel_scales):
        """Use this charge injection layout_ci to generate a pre-cti charge injection image. This is performed by \
        going to its charge injection regions and adding the charge injection normalization value.

        Parameters
        -----------
        shape_native : (int, int)
            The image_shape of the pre_cti_cis to be created.
        """

        pre_cti_ci = np.zeros(shape_native)

        for region in self.regions:
            pre_cti_ci[region.slice] += self.normalization

        return array_ci.CIarray.manual(
            array=pre_cti_ci, layout_ci=self, pixel_scales=pixel_scales
        )


class Layout2DCINonUniform(AbstractLayout2DCI):
    def __init__(
        self,
        normalization,
        regions,
        row_slope,
        column_sigma=None,
        maximum_normalization=np.inf,
    ):
        """A non-uniform charge injection layout_ci, which is defined by the regions it appears on a charge injection
        array_ci and its average normalization.

        Non-uniformity across the columns of a charge injection layout_ci is due to spikes / drops in the current that
        injects the charge. This is a noisy process, leading to non-uniformity with no regularity / smoothness. Thus,
        it cannot be modeled with an analytic profile, and must be assumed as prior-knowledge about the charge
        injection electronics or estimated from the observed charge injection ci_data.

        Non-uniformity across the rows of a charge injection layout_ci is due to a drop-off in voltage in the current.
        Therefore, it appears smooth and be modeled as an analytic function, which this code assumes is a
        power-law with slope row_slope.

        Parameters
        -----------
        normalization : float
            The normalization of the charge injection region.
        regions : [(int,)]
            A list of the integer coordinates specifying the corners of each charge injection region
            (top-row, bottom-row, left-column, right-column).
        row_slope : float
            The power-law slope of non-uniformity in the row charge injection profile.
        """
        super(Layout2DCINonUniform, self).__init__(normalization, regions)
        self.row_slope = row_slope
        self.column_sigma = column_sigma
        self.maximum_normalization = maximum_normalization

    def pre_cti_ci_from(self, shape_native, pixel_scales, ci_seed=-1):
        """Use this charge injection layout_ci to generate a pre-cti charge injection image. This is performed by going \
        to its charge injection regions and adding its non-uniform charge distribution.

        For one column of a non-uniform charge injection pre_cti_cis, it is assumed that each non-uniform charge \
        injection region has the same overall normalization value (after drawing this value randomly from a Gaussian \
        distribution). Physically, this is true provided the spikes / troughs in the current that cause \
        non-uniformity occur in an identical fashion for the generation of each charge injection region.

        Parameters
        -----------
        column_sigma
        shape_native
            The image_shape of the pre_cti_cis to be created.
        maximum_normalization

        ci_seed : int
            Input ci_seed for the random number generator to give reproducible results. A new ci_seed is always used for each \
            pre_cti_cis, ensuring each non-uniform ci_region has the same column non-uniformity layout_ci.
        """

        pre_cti_ci = np.zeros(shape_native)

        if ci_seed == -1:
            ci_seed = np.random.randint(
                0, int(1e9)
            )  # Use one ci_seed, so all regions have identical column
            # non-uniformity.

        for region in self.regions:
            pre_cti_ci[region.slice] += self.ci_region_from_region(
                region_dimensions=region.shape, ci_seed=ci_seed
            )

        return array_ci.CIarray.manual(
            array=pre_cti_ci, layout_ci=self, pixel_scales=pixel_scales
        )

    def ci_region_from_region(self, region_dimensions, ci_seed):
        """Generate the non-uniform charge distribution of a charge injection region. This includes non-uniformity \
        across both the rows and columns of the charge injection region.

        Before adding non-uniformity to the rows and columns, we assume an input charge injection level \
        (e.g. the average current being injected). We then simulator non-uniformity in this region.

        Non-uniformity in the columns is caused by sharp peaks and troughs in the input charge current. To simulator  \
        this, we change the normalization of each column by drawing its normalization value from a Gaussian \
        distribution which has a mean of the input normalization and standard deviation *column_sigma*. The seed \
        of the random number generator ensures that the non-uniform charge injection update_via_regions of each pre_cti_cis \
        are identical.

        Non-uniformity in the rows is caused by the charge smoothly decreasing as the injection is switched off. To \
        simulator this, we assume the charge level as a function of row number is not flat but defined by a \
        power-law with slope *row_slope*.

        Non-uniform charge injection images are generated using the function *simulate_pre_cti*, which uses this \
        function.

        Parameters
        -----------
        maximum_normalization
        column_sigma
        region_dimensions : (int, int)
            The size of the non-uniform charge injection region.
        ci_seed : int
            Input seed for the random number generator to give reproducible results.
        """

        np.random.seed(ci_seed)

        ci_rows = region_dimensions[0]
        ci_columns = region_dimensions[1]
        ci_region = np.zeros(region_dimensions)

        for column_number in range(ci_columns):

            column_normalization = 0
            while (
                column_normalization <= 0
                or column_normalization >= self.maximum_normalization
            ):
                column_normalization = np.random.normal(
                    self.normalization, self.column_sigma
                )

            ci_region[0:ci_rows, column_number] = self.generate_column(
                size=ci_rows, normalization=column_normalization
            )

        return ci_region

    def generate_column(self, size, normalization):
        """Generate a column of non-uniform charge, including row non-uniformity.

        The pixel-numbering used to generate non-uniformity across the charge injection rows runs from 1 -> size

        Parameters
        -----------
        size : int
            The size of the non-uniform column of charge
        normalization : float
            The input normalization of the column's charge e.g. the level of charge injected.

        """
        return normalization * (np.arange(1, size + 1)) ** self.row_slope


def regions_ci_from(
    injection_on: int,
    injection_off: int,
    injection_total: int,
    parallel_size: int,
    serial_size: int,
    serial_prescan_size: int,
    serial_overscan_size: int,
    roe_corner: (int, int),
):

    regions_ci = []

    injection_start_count = 0

    for index in range(injection_total):

        if roe_corner == (0, 0):

            ci_region = (
                parallel_size - (injection_start_count + injection_on),
                parallel_size - injection_start_count,
                serial_prescan_size,
                serial_size - serial_overscan_size,
            )

        elif roe_corner == (1, 0):

            ci_region = (
                injection_start_count,
                injection_start_count + injection_on,
                serial_prescan_size,
                serial_size - serial_overscan_size,
            )

        elif roe_corner == (0, 1):

            ci_region = (
                parallel_size - (injection_start_count + injection_on),
                parallel_size - injection_start_count,
                serial_overscan_size,
                serial_size - serial_prescan_size,
            )

        elif roe_corner == (1, 1):

            ci_region = (
                injection_start_count,
                injection_start_count + injection_on,
                serial_overscan_size,
                serial_size - serial_prescan_size,
            )

        regions_ci.append(ci_region)

        injection_start_count += injection_on + injection_off

    return regions_ci
