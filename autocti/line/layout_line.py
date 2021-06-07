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


class ExtractorFrontEdge(Extractor1D):
    def array_1d_list_from(self, array: array_1d.Array1D, pixels):
        """
        Extract a list of the parallel front edges of a 1D line  array.

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
        front_arrays = list(
            map(lambda region: array.native[region.slice], front_region_list)
        )
        front_masks = list(
            map(lambda region: array.mask[region.slice], front_region_list)
        )
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
                lambda ci_region: ci_region.parallel_front_edge_region_from(
                    rows=pixels
                ),
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


class ExtractorTrails(Extractor):
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


class ExtractorSerialFrontEdge(Extractor):
    def array_1d_list_from(self, array: array_1d.Array1D, columns):
        """
        Extract a list of the serial front edge structures of a charge injection array.

        The diagram below illustrates the arrays that is extracted from a array for columnss=(0, 3):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCDPhase panel    [pppppppppp] = parallel overscan
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

        The extracted array keeps just the serial front edges of all charge injection scans.

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
        front_region_list = self.region_list_from(columns=columns)
        front_arrays = list(
            map(lambda region: array.native[region.slice], front_region_list)
        )
        front_masks = list(
            map(lambda region: array.mask[region.slice], front_region_list)
        )
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

    def stacked_array_1d_from(self, array: array_1d.Array1D, columns):
        front_arrays = self.array_1d_list_from(array=array, columns=columns)
        return np.ma.mean(np.ma.asarray(front_arrays), axis=0)

    def binned_array_1d_from(self, array: array_1d.Array1D, columns):
        front_stacked_array = self.stacked_array_1d_from(array=array, columns=columns)
        return np.ma.mean(np.ma.asarray(front_stacked_array), axis=0)

    def region_list_from(self, columns):
        """
        Returns a list of the serial front edges scans of a charge injection array.

        The diagram below illustrates the region that is calculated from a array for columns=(0, 4):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCDPhase panel    [pppppppppp] = parallel overscan
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

        The extracted array keeps just the serial front edges of all charge injection scans.

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
        return list(
            map(
                lambda ci_region: ci_region.serial_front_edge_region_from(columns),
                self.region_list,
            )
        )

    def add_to_array(self, new_array, array, columns):

        region_list = [
            region.serial_front_edge_region_from(columns=columns)
            for region in self.region_list
        ]

        array_1d_list = self.array_1d_list_from(array=array, columns=columns)

        for arr, region in zip(array_1d_list, region_list):
            new_array[region.y0 : region.y1, region.x0 : region.x1] += arr

        return new_array


class ExtractorSerialTrails(Extractor):
    def array_1d_list_from(self, array: array_1d.Array1D, columns):
        """
        Extract a list of the serial trails of a charge injection array.

        The diagram below illustrates the arrays that is extracted from a array for columnss=(0, 3):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCDPhase panel    [pppppppppp] = parallel overscan
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

        The extracted array keeps just the serial front edges of all charge injection scans.

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
        trails_region_list = self.region_list_from(columns=columns)
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

    def stacked_array_1d_from(self, array: array_1d.Array1D, columns: Tuple[int, int]):
        front_arrays = self.array_1d_list_from(array=array, columns=columns)
        return np.ma.mean(np.ma.asarray(front_arrays), axis=0)

    def binned_array_1d_from(self, array: array_1d.Array1D, columns: Tuple[int, int]):
        trails_stacked_array = self.stacked_array_1d_from(array=array, columns=columns)
        return np.ma.mean(np.ma.asarray(trails_stacked_array), axis=0)

    def region_list_from(self, columns):
        """
        Returns a list of the serial trails scans of a charge injection array.

        The diagram below illustrates the region is calculated from a array for columnss=(0, 4):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCDPhase panel    [pppppppppp] = parallel overscan
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

        The extracted array keeps just the serial front edges of all charge injection scans.

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

        return list(
            map(
                lambda ci_region: ci_region.serial_trails_region_from(columns),
                self.region_list,
            )
        )

    def add_to_array(self, new_array, array, columns):

        region_list = [
            region.serial_trails_region_from(columns=columns)
            for region in self.region_list
        ]

        array_1d_list = self.array_1d_list_from(array=array, columns=columns)

        for arr, region in zip(array_1d_list, region_list):
            new_array[region.y0 : region.y1, region.x0 : region.x1] += arr

        return new_array


class PatternLine(object):
    def __init__(self, normalization: float, region_list: List[region.Region1D]):
        """ 
        Class for the pattern of a `Line` data structure, which defines the 1D regions the line charge appears
        on a CTI line dataset, e input normalization and other properties.

        Parameters
        -----------
        normalization
            The normalization of the line.
        region_list:
            A list of the integer coordinates specifying the (x0, x1) left and right pixel coordinates of each line 
            region.
        """
        self.normalization = normalization
        self.region_list = list(map(region.Region2D, region_list))

    def check_pattern_is_within_image_dimensions(self, dimensions: Tuple[int]):

        for region in self.region_list:

            if region.x1 > dimensions[1]:
                raise exc.PatternLineException(
                    "The line pattern_line regions are bigger than the image image_shape"
                )

    @property
    def total_pixels_min(self) -> int:
        return np.min(list(map(lambda region: region.total_pixels, self.region_list)))

    @property
    def pixels_between_regions(self) -> List[int]:
        return [
            self.region_list[i + 1].x0 - self.region_list[i].x1
            for i in range(len(self.region_list) - 1)
        ]

    def pre_cti_line_from(
        self, shape_native: Tuple[int], pixel_scales: Tuple[float]
    ) -> array_1d.Array1D:
        """
        Use this `PatternLine` pattern to generate a pre-cti line `Array1D`. This is performed by going to its each
        line region and adding the line normalization value.

        Parameters
        -----------
        shape_native
            The native shape of the of the pre-cti-line `Array1D` which is created.
        """

        self.check_pattern_is_within_image_dimensions(shape_native)

        pre_cti_line = np.zeros(shape_native)

        for region in self.region_list:
            pre_cti_line[region.slice] += self.normalization

        return array_1d.Array1D(
            array=pre_cti_line, pattern_line=self, pixel_scales=pixel_scales
        )


class PatternLineNonUniform(AbstractPatternLine):
    def __init__(
        self,
        normalization,
        region_list,
        row_slope,
        column_sigma=None,
        maximum_normalization=np.inf,
    ):
        """A non-uniform line pattern_line, which is defined by the regions it appears on a line
        frame_Line and its average normalization.

        Non-uniformity across the columns of a line pattern_line is due to spikes / drops in the current that
        injects the charge. This is a noisy process, leading to non-uniformity with no regularity / smoothness. Thus,
        it cannot be modeled with an analytic profile, and must be assumed as prior-knowledge about the charge
        injection electronics or estimated from the observed line Line_data.

        Non-uniformity across the rows of a line pattern_line is due to a drop-off in voltage in the current.
        Therefore, it appears smooth and be modeled as an analytic function, which this code assumes is a
        power-law with slope row_slope.

        Parameters
        -----------
        normalization : float
            The normalization of the line region.
        region_list : [(int,)]
            A list of the integer coordinates specifying the corners of each line region
            (top-row, bottom-row, left-column, right-column).
        row_slope : float
            The power-law slope of non-uniformity in the row line profile.
        """
        super(PatternLineNonUniform, self).__init__(normalization, region_list)
        self.row_slope = row_slope
        self.column_sigma = column_sigma
        self.maximum_normalization = maximum_normalization

    def pre_cti_line_from(self, shape_native, pixel_scales, Line_seed=-1):
        """Use this line pattern_line to generate a pre-cti line image. This is performed by going \
        to its line regions and adding its non-uniform charge distribution.

        For one column of a non-uniform line pre_cti_lines, it is assumed that each non-uniform charge \
        injection region has the same overall normalization value (after drawing this value randomly from a Gaussian \
        distribution). Physically, this is true provided the spikes / troughs in the current that cause \
        non-uniformity occur in an identical fashion for the generation of each line region.

        Parameters
        -----------
        column_sigma
        shape_native
            The image_shape of the pre_cti_lines to be created.
        maximum_normalization

        Line_seed : int
            Input Line_seed for the random number generator to give reproduLineble results. A new Line_seed is always used for each \
            pre_cti_lines, ensuring each non-uniform Line_region has the same column non-uniformity pattern_line.
        """

        self.check_pattern_is_within_image_dimensions(shape_native)

        pre_cti_line = np.zeros(shape_native)

        if Line_seed == -1:
            Line_seed = np.random.randint(
                0, int(1e9)
            )  # Use one Line_seed, so all regions have identical column
            # non-uniformity.

        for region in self.regions:
            pre_cti_line[region.slice] += self.Line_region_from_region(
                region_dimensions=region.shape, Line_seed=Line_seed
            )

        return frame_Line.LineFrame.manual(
            array=pre_cti_line, pattern_line=self, pixel_scales=pixel_scales
        )

    def Line_region_from_region(self, region_dimensions, Line_seed):
        """Generate the non-uniform charge distribution of a line region. This includes non-uniformity \
        across both the rows and columns of the line region.

        Before adding non-uniformity to the rows and columns, we assume an input line level \
        (e.g. the average current being injected). We then simulator non-uniformity in this region.

        Non-uniformity in the columns is caused by sharp peaks and troughs in the input charge current. To simulator  \
        this, we change the normalization of each column by drawing its normalization value from a Gaussian \
        distribution which has a mean of the input normalization and standard deviation *column_sigma*. The seed \
        of the random number generator ensures that the non-uniform line update_via_regions of each pre_cti_lines \
        are identical.

        Non-uniformity in the rows is caused by the charge smoothly decreasing as the injection is switched off. To \
        simulator this, we assume the charge level as a function of row number is not flat but defined by a \
        power-law with slope *row_slope*.

        Non-uniform line images are generated using the function *simulate_pre_cti*, which uses this \
        function.

        Parameters
        -----------
        maximum_normalization
        column_sigma
        region_dimensions : (int, int)
            The size of the non-uniform line region.
        Line_seed : int
            Input seed for the random number generator to give reproduLineble results.
        """

        np.random.seed(Line_seed)

        Line_rows = region_dimensions[0]
        Line_columns = region_dimensions[1]
        Line_region = np.zeros(region_dimensions)

        for column_number in range(Line_columns):

            column_normalization = 0
            while (
                column_normalization <= 0
                or column_normalization >= self.maximum_normalization
            ):
                column_normalization = np.random.normal(
                    self.normalization, self.column_sigma
                )

            Line_region[0:Line_rows, column_number] = self.generate_column(
                size=Line_rows, normalization=column_normalization
            )

        return Line_region

    def generate_column(self, size, normalization):
        """Generate a column of non-uniform charge, including row non-uniformity.

        The pixel-numbering used to generate non-uniformity across the line rows runs from 1 -> size

        Parameters
        -----------
        size : int
            The size of the non-uniform column of charge
        normalization : float
            The input normalization of the column's charge e.g. the level of charge injected.

        """
        return normalization * (np.arange(1, size + 1)) ** self.row_slope


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
