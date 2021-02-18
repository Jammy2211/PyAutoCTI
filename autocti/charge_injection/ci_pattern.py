"""
File: python/VIS_CTICalibrate/ChargeInjectPattern.py

Created on: 02/14/18
Author: James Nightingale
"""

from copy import deepcopy

import numpy as np
from autocti import exc
from autoarray.structures import region
from autoarray.structures.frames import frame_util
from autocti.charge_injection import ci_frame


class AbstractCIPattern(object):
    def __init__(self, normalization, regions):
        """ Abstract base class for a charge injection ci_pattern, which defines the regions charge injections appears \
         on a charge-injection ci_frame, the input normalization and other properties.

        Parameters
        -----------
        normalization : float
            The normalization of the charge injection lines.
        regions: [(int,)]
            A list of the integer coordinates specifying the corners of each charge injection region \
            (top-row, bottom-row, left-column, right-column).
        """
        self.normalization = normalization
        self.regions = list(map(region.Region2D, regions))

    def with_extracted_regions(self, extraction_region):

        ci_pattern = deepcopy(self)

        extracted_regions = list(
            map(
                lambda region: frame_util.region_after_extraction(
                    original_region=region, extraction_region=extraction_region
                ),
                self.regions,
            )
        )
        extracted_regions = list(filter(None, extracted_regions))
        if not extracted_regions:
            extracted_regions = None

        ci_pattern.regions = extracted_regions
        return ci_pattern

    def check_pattern_is_within_image_dimensions(self, dimensions):

        for region in self.regions:

            if region.y1 > dimensions[0] or region.x1 > dimensions[1]:
                raise exc.CIPatternException(
                    "The charge injection ci_pattern regions are bigger than the image image_shape"
                )

    @property
    def total_rows_min(self):
        return np.min(list(map(lambda region: region.total_rows, self.regions)))

    @property
    def total_columns_min(self):
        return np.min(list(map(lambda region: region.total_columns, self.regions)))

    @property
    def rows_between_regions(self):
        return [
            self.regions[i + 1].y0 - self.regions[i].y1
            for i in range(len(self.regions) - 1)
        ]


class CIPatternUniform(AbstractCIPattern):
    """ A uniform charge injection ci_pattern, which is defined by the regions it appears on the charge injection \
        ci_frame and its normalization.

    """

    def ci_pre_cti_from(self, shape_native, pixel_scales):
        """Use this charge injection ci_pattern to generate a pre-cti charge injection image. This is performed by \
        going to its charge injection regions and adding the charge injection normalization value.

        Parameters
        -----------
        shape_native : (int, int)
            The image_shape of the ci_pre_ctis to be created.
        """

        self.check_pattern_is_within_image_dimensions(shape_native)

        ci_pre_cti = np.zeros(shape_native)

        for region in self.regions:
            ci_pre_cti[region.slice] += self.normalization

        return ci_frame.CIFrame.manual(
            array=ci_pre_cti, ci_pattern=self, pixel_scales=pixel_scales
        )


class CIPatternNonUniform(AbstractCIPattern):
    def __init__(
        self,
        normalization,
        regions,
        row_slope,
        column_sigma=None,
        maximum_normalization=np.inf,
    ):
        """A non-uniform charge injection ci_pattern, which is defined by the regions it appears on a charge injection
        ci_frame and its average normalization.

        Non-uniformity across the columns of a charge injection ci_pattern is due to spikes / drops in the current that
        injects the charge. This is a noisy process, leading to non-uniformity with no regularity / smoothness. Thus,
        it cannot be modeled with an analytic profile, and must be assumed as prior-knowledge about the charge
        injection electronics or estimated from the observed charge injection ci_data.

        Non-uniformity across the rows of a charge injection ci_pattern is due to a drop-off in voltage in the current.
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
        super(CIPatternNonUniform, self).__init__(normalization, regions)
        self.row_slope = row_slope
        self.column_sigma = column_sigma
        self.maximum_normalization = maximum_normalization

    def ci_pre_cti_from(self, shape_native, pixel_scales, ci_seed=-1):
        """Use this charge injection ci_pattern to generate a pre-cti charge injection image. This is performed by going \
        to its charge injection regions and adding its non-uniform charge distribution.

        For one column of a non-uniform charge injection ci_pre_ctis, it is assumed that each non-uniform charge \
        injection region has the same overall normalization value (after drawing this value randomly from a Gaussian \
        distribution). Physically, this is true provided the spikes / troughs in the current that cause \
        non-uniformity occur in an identical fashion for the generation of each charge injection region.

        Parameters
        -----------
        column_sigma
        shape_native
            The image_shape of the ci_pre_ctis to be created.
        maximum_normalization

        ci_seed : int
            Input ci_seed for the random number generator to give reproducible results. A new ci_seed is always used for each \
            ci_pre_ctis, ensuring each non-uniform ci_region has the same column non-uniformity ci_pattern.
        """

        self.check_pattern_is_within_image_dimensions(shape_native)

        ci_pre_cti = np.zeros(shape_native)

        if ci_seed == -1:
            ci_seed = np.random.randint(
                0, int(1e9)
            )  # Use one ci_seed, so all regions have identical column
            # non-uniformity.

        for region in self.regions:
            ci_pre_cti[region.slice] += self.ci_region_from_region(
                region_dimensions=region.shape, ci_seed=ci_seed
            )

        return ci_frame.CIFrame.manual(
            array=ci_pre_cti, ci_pattern=self, pixel_scales=pixel_scales
        )

    def ci_region_from_region(self, region_dimensions, ci_seed):
        """Generate the non-uniform charge distribution of a charge injection region. This includes non-uniformity \
        across both the rows and columns of the charge injection region.

        Before adding non-uniformity to the rows and columns, we assume an input charge injection level \
        (e.g. the average current being injected). We then simulator non-uniformity in this region.

        Non-uniformity in the columns is caused by sharp peaks and troughs in the input charge current. To simulator  \
        this, we change the normalization of each column by drawing its normalization value from a Gaussian \
        distribution which has a mean of the input normalization and standard deviation *column_sigma*. The seed \
        of the random number generator ensures that the non-uniform charge injection update_via_regions of each ci_pre_ctis \
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


def ci_regions_from(injection_on : int, injection_off : int, injection_total : int, serial_size : int, serial_prescan_size : int, serial_overscan_size : int):

    ci_regions = []

    injection_start_count = 0

    for index in range(injection_total):

        ci_region = (injection_start_count, injection_start_count + injection_on, serial_prescan_size, serial_size - serial_overscan_size)

        ci_regions.append(ci_region)

        injection_start_count += injection_on + injection_off

    return ci_regions

