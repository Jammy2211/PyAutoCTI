"""
File: python/VIS_CTICalibrate/LinePattern.py

Created on: 02/14/18
Author: James Nightingale
"""

from copy import deepcopy

import numpy as np
from autocti import exc
from autoarray.structures.arrays.one_d import array_1d
from autoarray.structures import region

from typing import List, Tuple


class PatternLine(object):
    def __init__(self, normalization: float, regions: List[region.Region1D]):
        """ 
        Class for the pattern of a `Line` data structure, which defines the 1D regions the line charge appears
        on a CTI line dataset, e input normalization and other properties.

        Parameters
        -----------
        normalization
            The normalization of the line.
        regions:
            A list of the integer coordinates specifying the (x0, x1) left and right pixel coordinates of each line 
            region.
        """
        self.normalization = normalization
        self.regions = list(map(region.Region2D, regions))

    def check_pattern_is_within_image_dimensions(self, dimensions: Tuple[int]):

        for region in self.regions:

            if region.x1 > dimensions[1]:
                raise exc.PatternLineException(
                    "The line pattern_line regions are bigger than the image image_shape"
                )

    @property
    def total_pixels_min(self) -> int:
        return np.min(list(map(lambda region: region.total_pixels, self.regions)))

    @property
    def pixels_between_regions(self) -> List[int]:
        return [
            self.regions[i + 1].x0 - self.regions[i].x1
            for i in range(len(self.regions) - 1)
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

        for region in self.regions:
            pre_cti_line[region.slice] += self.normalization

        return array_1d.Array1D(
            array=pre_cti_line, pattern_line=self, pixel_scales=pixel_scales
        )


class PatternLineNonUniform(AbstractPatternLine):
    def __init__(
        self,
        normalization,
        regions,
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
        regions : [(int,)]
            A list of the integer coordinates specifying the corners of each line region
            (top-row, bottom-row, left-column, right-column).
        row_slope : float
            The power-law slope of non-uniformity in the row line profile.
        """
        super(PatternLineNonUniform, self).__init__(normalization, regions)
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
