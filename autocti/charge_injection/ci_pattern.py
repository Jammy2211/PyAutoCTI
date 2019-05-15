"""
File: python/VIS_CTICalibrate/ChargeInjectPattern.py

Created on: 02/14/18
Author: James Nightingale
"""

import numpy as np

from autocti import exc
from autocti.charge_injection import ci_frame


def uniform_from_lists(normalizations, regions):
    """Setup the collection of patterns from lists of uniform ci_pattern properties

    Params
    -----------
    normalizations : list
        The normalization in each charge injection ci_pattern.
    regions : [(int, int, int, int)]
        The regions each charge injection ci_pattern appears. This is identical across all images.
    """
    return list(map(lambda n: CIPatternUniform(n, regions), normalizations))


def non_uniform_from_lists(normalizations, regions, row_slopes):
    """Setup the collection of patterns from lists of non-uniform ci_pattern properties

    Params
    -----------
    normalizations : [float]
        The normalization in each charge injection ci_pattern.
    regions : [(int, int, int, int)]
        The regions each charge injection ci_pattern appears. This is identical across all images.
    row_slopes : [float]
        The power-law slopes of non-uniformity in the rows of each charge injection profile.
    """
    return list(map(lambda n, s: CIPatternNonUniform(n, regions, s), normalizations, row_slopes))


class CIPattern(object):

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
        self.regions = list(map(ci_frame.Region, regions))

    def check_pattern_is_within_image_dimensions(self, dimensions):

        for region in self.regions:

            if region.y1 > dimensions[0] or region.x1 > dimensions[1]:
                raise exc.CIPatternException(
                    'The charge injection ci_pattern regions are bigger than the image image_shape')

    @property
    def total_rows_min(self):
        return np.min(list(map(lambda region : region.total_rows, self.regions)))

    @property
    def total_columns_min(self):
        return np.min(list(map(lambda region : region.total_columns, self.regions)))

    @property
    def rows_between_regions(self):
        return [self.regions[i+1].y0 - self.regions[i].y1 for i in range(len(self.regions)-1)]

class CIPatternUniform(CIPattern):
    """ A uniform charge injection ci_pattern, which is defined by the regions it appears on the charge injection \
        ci_frame and its normalization.

    """

    def ci_pre_cti_from_shape(self, shape):
        """Compute the pre-cti image of the uniform charge injection ci_pattern.

        This is performed by going to each charge injection region and adding the charge injection normalization value.

        Parameters
        -----------
        shape : (int,)
            The image_shape of the ci_pre_ctis to be created.
        """

        self.check_pattern_is_within_image_dimensions(shape)

        ci_pre_cti = np.zeros(shape)

        for region in self.regions:
            ci_pre_cti[region.slice] += self.normalization

        return ci_pre_cti

    """ Class to simulate a charge injection image with a uniform charge injection ci_pattern."""

    def simulate_ci_pre_cti(self, shape):
        """Use this charge injection ci_pattern to generate a pre-cti charge injection image. This is performed by \
        going to its charge injection regions and adding the charge injection normalization value.

        Parameters
        -----------
        shape : (int, int)
            The image_shape of the ci_pre_ctis to be created.
        """

        self.check_pattern_is_within_image_dimensions(shape)

        ci_pre_cti = np.zeros(shape)

        for region in self.regions:
            ci_pre_cti[region.slice] += self.normalization

        return ci_pre_cti


class CIPatternNonUniform(CIPattern):

    def __init__(self, normalization, regions, row_slope):
        """ A non-uniform charge injection ci_pattern, which is defined by the regions it appears on a charge injection
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

    def ci_pre_cti_from_ci_image_and_mask(self, ci_image, mask):
        """Compute the pre-cti image of this non-uniform charge injection ci_pattern.

        This is performed by estimating the charge injection of each column, by taking the average value of all charge \
        regions in that column (including non-uniformity across the rows). The mask is used to remove cosmic rays from \
        this averaging.

        Parameters
        -----------
        ci_image : ndarray
            2D array of ci_pre_ctis ci_data the column non-uniformity is estimated from.
        mask : ndarray
            2D array of masked ci_pre_ctis pixels, used to mask cosmic rays.
        """

        dimensions = ci_image.shape

        self.check_pattern_is_within_image_dimensions(dimensions)

        ci_pre_cti = np.zeros(dimensions)

        for column_number in range(dimensions[1]):

            means_of_columns = self.mean_charge_in_all_image_columns(column=ci_image[:, column_number],
                                                                     column_mask=mask[:, column_number])

            filtered = list(filter(None, means_of_columns))
            if len(filtered) == 0:
                raise exc.CIPatternException(
                    'All Pixels in a charge injection region were flagged as masked - code does'
                    'not currently handle such a circumstance')

            overall_mean = np.mean(filtered)

            for region in self.regions:
                ci_pre_cti[region.y_slice, column_number] = overall_mean

        return ci_pre_cti

    def mean_charge_in_all_image_columns(self, column, column_mask):
        """For one column of a charge injection image, measure the mean charge in each column of each charge injection \
        region.

        Parameters
        -----------
        column : ndarray
            1D array of the ci_pre_ctis column.
        column_mask : ndarray
            1D array of the column's masked pixels, to mask cosmic rays.
         """

        means_of_columns = []

        for region in self.regions:
            means_of_columns.append(self.mean_charge_in_column(column[region.y_slice].flatten(),
                                                               column_mask[region.y_slice].flatten()))

        return means_of_columns

    def mean_charge_in_column(self, column, column_mask):
        """Measure the mean charge in a column of a non-uniform charge injection region (accounting for row \
        non-uniformity if present). Cosmic rays can be masked.

        Parameters
        -----------
        column : ndarray
            1D array of the column of the charge injection region.
        column_mask : ndarray
            1D array of the column's masked pixels, to mask cosmic rays.
        """
        column_size = len(column)

        if sum(column_mask) == column_size:
            return None

        model_column = self.generate_column(column_size, self.normalization)
        return np.mean(np.ma.masked_array(column - model_column + self.normalization, column_mask))

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

    def simulate_ci_pre_cti(self, shape, ci_seed=-1, column_deviation=0.0,
                            maximum_normalization=np.inf):
        """Use this charge injection ci_pattern to generate a pre-cti charge injection image. This is performed by going \
        to its charge injection regions and adding its non-uniform charge distribution.

        For one column of a non-uniform charge injection ci_pre_ctis, it is assumed that each non-uniform charge \
        injection region has the same overall normalization value (after drawing this value randomly from a Gaussian \
        distribution). Physically, this is true provided the spikes / troughs in the current that cause \
        non-uniformity occur in an identical fashion for the generation of each charge injection region.

        Parameters
        -----------
        column_deviation
        shape
            The image_shape of the ci_pre_ctis to be created.
        maximum_normalization

        ci_seed : int
            Input ci_seed for the random number generator to give reproducible results. A new ci_seed is always used for each \
            ci_pre_ctis, ensuring each non-uniform ci_region has the same column non-uniformity ci_pattern.
        """

        self.check_pattern_is_within_image_dimensions(shape)

        ci_pre_cti = np.zeros(shape)

        if ci_seed == -1:
            ci_seed = np.random.randint(0, int(1e9))  # Use one ci_seed, so all regions have identical column
            # non-uniformity.

        for region in self.regions:
            ci_pre_cti[region.slice] += \
                self.simulate_region(region_dimensions=region.shape, ci_seed=ci_seed, column_deviation=column_deviation,
                                     maximum_normalization=maximum_normalization)

        return ci_pre_cti

    def simulate_region(self, region_dimensions, ci_seed, column_deviation=0.0, maximum_normalization=np.inf):
        """Generate the non-uniform charge distribution of a charge injection region. This includes non-uniformity \
        across both the rows and columns of the charge injection region.

        Before adding non-uniformity to the rows and columns, we assume an input charge injection level \
        (e.g. the average current being injected). We then simulate non-uniformity in this region.

        Non-uniformity in the columns is caused by sharp peaks and troughs in the input charge current. To simulate  \
        this, we change the normalization of each column by drawing its normalization value from a Gaussian \
        distribution which has a mean of the input normalization and standard deviation *column_deviation*. The seed \
        of the random number generator ensures that the non-uniform charge injection update_via_regions of each ci_pre_ctis \
        are identical.

        Non-uniformity in the rows is caused by the charge smoothly decreasing as the injection is switched off. To \
        simulate this, we assume the charge level as a function of row number is not flat but defined by a \
        power-law with slope *row_slope*.

        Non-uniform charge injection images are generated using the function *simulate_pre_cti*, which uses this \
        function.

        Parameters
        -----------
        maximum_normalization
        column_deviation
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
            while column_normalization <= 0 or column_normalization >= maximum_normalization:
                column_normalization = np.random.normal(self.normalization, column_deviation)

            ci_region[0:ci_rows, column_number] = self.generate_column(size=ci_rows, normalization=column_normalization)

        return ci_region
