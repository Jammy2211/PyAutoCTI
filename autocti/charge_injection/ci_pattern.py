"""
File: python/VIS_CTICalibrate/ChargeInjectPattern.py

Created on: 02/14/18
Author: James Nightingale
"""

from __future__ import division, print_function
import sys

if sys.version_info[0] < 3:
    from future_builtins import *

import numpy as np

from autocti.image import cti_image
from autocti.tools import infoio
from autocti import exc


def create_uniform_via_lists(normalizations, regions):
    """Setup the collection of patterns from lists of uniform ci_pattern properties

    Params
    -----------
    normalizations : list
        The normalization in each charge injection ci_pattern.
    regions : [(int, int, int, int)]
        The regions each charge injection ci_pattern appears. This is identical across all images.
    """
    return list(map(lambda n: CIPatternUniform(n, regions), normalizations))

def create_non_uniform_via_lists(normalizations, regions, row_slopes):
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

def create_uniform_fast_via_lists(normalizations, regions):
    """Setup the collection of fast patterns from lists of uniform ci_pattern properties

    Params
    -----------
    normalizations : list
        The normalization in each charge injection ci_pattern.
    regions : [(int, int, int, int)]
        The regions each charge injection ci_pattern appears. This is identical across all images.
    """
    return list(map(lambda n: CIPatternUniformFast(n, regions), normalizations))

def create_uniform_simulate_via_lists(normalizations, regions):
    """Setup the collection of simulation patterns from lists of uniform ci_pattern properties

    Params
    -----------
    normalizations : list
        The normalization in each charge injection ci_pattern.
    regions : [(int, int, int, int)]
        The regions each charge injection ci_pattern appears. This is identical across all images.
    """
    return list(map(lambda n: CIPatternUniformSimulate(n, regions), normalizations))

def create_non_uniform_simulate_via_lists(normalizations, regions, column_deviations, row_slopes,
                                          maximum_normalization):
    """Setup the collection of simulation patterns from lists of non-uniform ci_pattern properties

    Params
    -----------
    normalizations : list
        The normalization in each charge injection ci_pattern.
    regions : [(int, int, int, int)]
        The regions each charge injection ci_pattern appears. This is identical across all images.
    column_deviation : [float]
        The level of charge deviation across the columns of the region e.g. the standard deviation of the \
        Gaussian distribution each charge level is drawn from.
    row_slopes : [float]
        The power-law slopes of non-uniformity in the rows of each charge injection profile.
    maximum_normalization : float
        The maximum normalization of a charge injection column (e.g. the full well capacity)
    """
    return list(map(lambda n, c, s: CIPatternNonUniformSimulate(n, regions, c, s, maximum_normalization),
                    normalizations, column_deviations, row_slopes))


class CIPattern(object):

    def __init__(self, normalization, regions):
        """ Abstract base class for a charge injection ci_pattern, which defines the regions charge injections appears \
         on a charge-injection ci_frame, the input normalization and other properties.

        Parameters
        -----------
        normalization : float
            The normalization of the charge injection lines.
        ci_regions : [(int, int, int, int)]
            A list of the integer coordinates specifying the corners of each charge injection region \
            (top-row, bottom-row, lelf-column, right-column).
        """
        self.normalization = normalization
        self.regions = list(map(lambda region : cti_image.Region(region), regions))

    def generate_info(self):
        """Generate string containing information on the charge injection ci_pattern."""
        return infoio.generate_class_info(self, prefix='ci_pattern_', include_types=[int, float, list])

    def output_info_file(self, path, filename='CIPattern'):
        """Output information on the charge injection ci_pattern to a text file.

        Params
        ----------
        path : str
            The output nlo path of the ci_data
        """
        infoio.output_class_info(self, path, filename)

    def compute_ci_pre_cti(self, *values):
        raise AssertionError("CIPattern - compute_ci_pre_cti should be overriden by child class")

    def check_pattern_is_within_image_dimensions(self, dimensions):

        for region in self.regions:

            if region.y1 > dimensions[0] or region.x1 > dimensions[1]:
                raise exc.CIPatternException('The charge injection ci_pattern regions are bigger than the image image_shape')


class CIPatternUniform(CIPattern):

    def __init__(self, normalization, regions):
        """ A uniform charge injection ci_pattern, which is defined by the regions it appears on the charge injection \
        ci_frame and its normalization.

        Parameters
        -----------
        normalization : float
            The normalization of the uniform charge injection region.
        ci_regions : [(int, int, int, int)]
            A list of the integer coordinates specifying the corners of each charge injection region. This is \
            defined as in a NumPy array, e.g. (top-row, bottom-row, lelf-column, right-column).
        """
        super(CIPatternUniform, self).__init__(normalization, regions)

    def compute_ci_pre_cti(self, shape):
        """Compute the pre-cti image of the uniform charge injection ci_pattern.

        This is performed by going to each charge injection region and adding the charge injection normalization value.

        Parameters
        -----------
        image_shape : (int, int)
            The image_shape of the ci_pre_ctis to be created.
        """

        self.check_pattern_is_within_image_dimensions(shape)

        ci_pre_cti = np.zeros(shape)

        for region in self.regions:

            ci_pre_cti[region.y0:region.y1, region.x0:region.x1] += self.normalization

        return ci_pre_cti


class CIPatternNonUniform(CIPattern):

    def __init__(self, normalization, regions, row_slope):
        """ A non-uniform charge injection ci_pattern, which is defined by the regions it appears on a charge injection \
        ci_frame and its average normalization.

        Non-uniformity across the columns of a charge injection ci_pattern is due to spikes / drops in the current that \
        injects the charge. This is a noisy process, leading to non-uniformity with no regularity / smoothness. Thus, \
        it cannot be modeled with an analytic profile, and must be assumed as prior-knowledge about the charge \
        injection electronics or estimated from the observed charge injection ci_data.

        Non-uniformity across the rows of a charge injection ci_pattern is due to a drop-off in voltage in the current. \
        Therefore, it appears smooth and be modeled as an analytic function, which this code assumes is a  \
        power-law with slope row_slope.

        Parameters
        -----------
        normalization : float
            The normalization of the charge injection region.
        regions : [(int, int, int, int)]
            A list of the integer coordinates specifying the corners of each charge injection region \
            (top-row, bottom-row, lelf-column, right-column).
        row_slope : float
            The power-law slope of non-uniformity in the row charge injection profile.
        """
        super(CIPatternNonUniform, self).__init__(normalization, regions)
        self.row_slope = row_slope

    def compute_ci_pre_cti(self, image, mask):
        """Compute the pre-cti image of this non-uniform charge injection ci_pattern.

        This is performed by estimating the charge injection of each column, by taking the average value of all charge \
        regions in that column (including non-uniformity across the rows). The mask is used to remove cosmic rays from \
        this averaging.

        Parameters
        -----------
        image : ndarray
            2D array of ci_pre_ctis ci_data the column non-uniformity is estimated from.
        mask : ndarray
            2D array of masked ci_pre_ctis pixels, used to mask cosmic rays.
        """

        dimensions = image.shape

        self.check_pattern_is_within_image_dimensions(dimensions)

        ci_pre_cti = np.zeros(dimensions)

        for column_number in range(dimensions[1]):

            means_of_columns = self.mean_charge_in_all_image_columns(column=image[:, column_number],
                                                                     column_mask=mask[:, column_number])

            if None in means_of_columns:
                means_of_columns.remove(None)
                if means_of_columns == []:
                    raise exc.CIPatternException(
                        'All Pixels in a charge injection region were flagged as masked - code does'
                        'not currently handle such a circumstance')

            overall_mean = np.mean(means_of_columns)

            for region in self.regions:
                ci_pre_cti[region.y0:region.y1, column_number] = overall_mean

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

            means_of_columns.append(self.mean_charge_in_column(column[region.y0:region.y1].flatten(),
                                                               column_mask[region.y0:region.y1].flatten()))

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
        return normalization * (np.arange(1, size+1))**self.row_slope


class CIPatternUniformFast(CIPatternUniform):

    def __init__(self, normalization, regions):
        """ A fast uniform charge injection ci_pattern, which is defined by the regions it appears on a charge injection \
         ci_frame and its normalization.
         
         This is used for performing fast CTI addition in CTI calibration (see *CIPreCTIFast*).

        Parameters
        -----------
        normalization : float
            The normalization of the charge injection region.
        ci_regions : [(int, int, int, int)]
            A list of the integer coordinates specifying the corners of each charge injection region. This is \
            defined as in a NumPy array, e.g. (top-row, bottom-row, lelf-column, right-column).
        """
        super(CIPatternUniformFast, self).__init__(normalization, regions)

    def compute_fast_column(self, number_rows):
        """Compute a uniform fast column, which represents one column of charge in a uniform charge injection immage \
        (and therefore every column of charge in that pre-cti image).

        This is performed by using the charge injection ci_pattern's regions to determine the rows which contain \
        charge and adding its normalization to those rows.

        The fast columns is output as a 2D NumPy array where the second dimension is of size 1. This is performed \
        so that the *ci_image.FrameGeometry* routines can be applied to the output fast_column.

        Parameters
        -----------
        number_rows : int
            The number of rows in the fast column, thus defining its size and image_shape.
        """

        fast_column = np.zeros((number_rows, 1))

        for region in self.regions:

            fast_column[region.y0:region.y1, 0] += self.normalization

        return fast_column

    def compute_fast_row(self, number_columns):
        """Compute a uniform fast row, which represents one row of charge in a uniform charge injection image \
        (and therefore every row of charge in that ci_pre_ctis).

        This is performed by using the charge injection ci_pattern's regions to determine the rows which contain \
        charge and adding its normalization to those rows.

        Unlike the fast column above, which assumes there may be multiple regions corresponing to the charge injection \
        going on and off, all rows are assumed to be identical. This is consistent with the charge injection occuring \
        perpendicular to serial clocking.

        The fast rows is output as a 2D NumPy array where the second dimension is of size 1. This is performed \
        so that the *ci_image.FrameGeometry* routines can be applied to the output fast_row.

        Parameters
        -----------
        number_columns : int
            The number of columns in the fast row, thus defining its size and image_shape.
        """

        fast_row = np.zeros((1, number_columns))

        fast_row[0, self.regions[0].x0:self.regions[0].x1] += self.normalization

        return fast_row


class CIPatternUniformSimulate(CIPatternUniform):

    def __init__(self, normalization, regions):
        """ Class to simulate a charge injection image with a uniform charge injection ci_pattern.

        Parameters
        -----------
        normalization : float
            The normalization of the charge injection region.
        ci_regions : [(int, int, int, int)]
            A list of the integer coordinates specifying the corners of each charge injection region. This is \
            defined as in a NumPy array, e.g. (top-row, bottom-row, lelf-column, right-column).
        """
        super(CIPatternUniformSimulate, self).__init__(normalization, regions)

    def simulate_ci_pre_cti(self, dimensions, *values):
        """Use this charge injection ci_pattern to generate a pre-cti charge injection image. This is performed by going \
        to its charge injection regions and adding the charge injection normalization value.

        Parameters
        -----------
        image_shape : (int, int)
            The image_shape of the ci_pre_ctis to be created.
        """

        self.check_pattern_is_within_image_dimensions(dimensions)

        ci_pre_cti = np.zeros(dimensions)

        for region in self.regions:
            ci_pre_cti[region.y0:region.y1, region.x0:region.x1] += self.normalization

        return ci_pre_cti

    def create_pattern(self):
        return CIPatternUniformFast(normalization=self.normalization, regions=self.regions)


class CIPatternNonUniformSimulate(CIPatternNonUniform):

    def __init__(self, normalization, regions, column_deviation=0.0, row_slope=0.0, maximum_normalization=np.inf):
        """ Class to simulate a charge injection image with a non-uniform charge injection ci_pattern.

        Non-uniformity across columns is simulated by drawing each normalization value from a Gaussian with \
        standard deviation *column_deviation*. Non-uniformity across rows is simulated by drawing their normalization \
        as a function of pixel distance from a power-law. See *CIPatternNonUniform* for more details.

        Parameters
        -----------
        normalization : float
            The normalization of the charge injection region.
        ci_regions : [(int, int, int, int)]
            A list of the integer coordinates specifying the corners of each charge injection region. This is defined as \
            in a NumPy array, e.g. (top-row, bottom-row, lelf-column, right-column).
        column_deviation : float
            The level of charge deviation across the columns of the region e.g. the standard deviation of the \
             Gaussian distribution each charge level is drawn from.
        row_slope : float
            The power-law slope of non-uniformity in the rows of the charge injection profile.
        maximum_normalization : float
            The maximum normalization of a charge injection column (e.g. the full well capacity)
        """
        super(CIPatternNonUniformSimulate, self).__init__(normalization, regions, row_slope)
        self.column_deviation = column_deviation
        self.maximum_normalization = maximum_normalization

    def simulate_ci_pre_cti(self, dimensions, ci_seed=-1):
        """Use this charge injection ci_pattern to generate a pre-cti charge injection image. This is performed by going \
        to its charge injection regions and adding its non-uniform charge distribution.

        For one column of a non-uniform charge injection ci_pre_ctis, it is assumed that each non-uniform charge \
        injection region has the same overall normalization value (after drawing this value randomly from a Gaussian \
        distribution). Physically, this is true provided the spikes / troughs in the current that cause \
        non-uniformity occur in an identical fashion for the generation of each charge injection region.

        Parameters
        -----------
        image_shape : (int, int)
            The image_shape of the ci_pre_ctis to be created.
        ci_seed : int
            Input ci_seed for the random number generator to give reproducible results. A new ci_seed is always used for each \
            ci_pre_ctis, ensuring each non-uniform ci_region has the same column non-uniformity ci_pattern.
        """

        self.check_pattern_is_within_image_dimensions(dimensions)

        ci_pre_cti = np.zeros(dimensions)

        if ci_seed == -1:
            ci_seed = np.random.randint(0, 1e9) # Use one ci_seed, so all regions have identical column non-uniformity.

        for region in self.regions:
            ci_pre_cti[region.y0:region.y1, region.x0:region.x1] += \
                self.simulate_region(region_dimensions=(region.y1 - region.y0, region.x1 - region.x0), ci_seed=ci_seed)

        return ci_pre_cti

    def simulate_region(self, region_dimensions, ci_seed):
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
            while column_normalization <= 0 or column_normalization >= self.maximum_normalization:
                column_normalization = np.random.normal(self.normalization, self.column_deviation)

            ci_region[0:ci_rows, column_number] = self.simulate_column(size=ci_rows, normalization=column_normalization)

        return ci_region

    def simulate_column(self, size, normalization):
        """Simulate a column of non-uniform charge, including row non-uniformity.

        The pixel-numbering used to generate non-uniformity across the charge injection rows runs from 1 -> size

        Parameters
        -----------
        size : int
            The size of the non-uniform column of charge
        normalization : float
            The input normalization of the column's charge e.g. the level of charge injected.

        """
        return self.generate_column(size, normalization)

    def create_pattern(self):
        return CIPatternNonUniform(normalization=self.normalization, regions=self.regions, row_slope=self.row_slope)



