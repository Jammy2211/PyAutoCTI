import numpy as np
from typing import List, Tuple, Union

import autoarray as aa


class Extractor2D:
    def __init__(self, region_list: aa.type.Region2DList):
        """
        Abstract class containing methods for extracting regions from a 2D charge injection image.

        This uses the `region_list`, which contains the charge injection regions in pixel coordinates.

        Parameters
        ----------
        region_list
            Integer pixel coordinates specifying the corners of each charge injection region (top-row, bottom-row,
            left-column, right-column).
        """
        self.region_list = list(map(aa.Region2D, region_list))

    @property
    def total_rows_min(self) -> int:
        """
        The number of rows between the read-out electronics and the charge injection region closest to them.
        """
        return np.min([region.total_rows for region in self.region_list])

    @property
    def total_columns_min(self) -> int:
        """
        The number of columns between the read-out electronics and the charge injection region closest to them.
        """
        return np.min([region.total_columns for region in self.region_list])

    @property
    def binning_axis(self) -> int:
        """
        The axis over which binning is performed to turn a 2D FPR into a 1D FPR.

        For a parallel extractor `axis=1` such that binning is performed over the rows containing the FPR.
        """
        raise NotImplementedError

    def region_list_from(self, pixels: Tuple[int, int]) -> List[aa.Region2D]:
        raise NotImplementedError

    def array_2d_list_from(
        self, array: aa.Array2D, pixels: Tuple[int, int]
    ) -> List[aa.Array2D]:
        """
        Extract a specific region from every charge injection region on the charge injection image and return as a list
        of 2D arrays.

        For example, this might extract the parallel EPERs of every charge injection region.

        The `region_2d_list_from()` of each `Extractor` class describes the exact extraction performed for each
        extractor when this function is called..
        """
        return [
            np.ma.array(data=array.native[region.slice], mask=array.mask[region.slice])
            for region in self.region_list_from(pixels=pixels)
        ]

    def stacked_array_2d_from(
        self, array: aa.Array2D, pixels: Tuple[int, int]
    ) -> np.ndarray:
        """
        Extract a region (e.g. the parallel FPR) of every charge injection region on the charge injection image and
        stack them by taking their mean.

        This returns the 2D average of the extracted regions (e.g. the parallel FPRs) of all of the charge injection
        regions, which for certain CCD charge injection electronics one may expect to be similar.

        For fits to charge injection data this function is also used to create images like the stacked 2D residuals,
        which therefore quantify the goodness-of-fit of a CTI model.

        Parameters
        ------------
        array
            The 2D array which contains the charge injection image from which the regions (e.g. the parallel FPRs)
            are extracted and stacked.
        pixels
            The row pixel index to extract the FPR between (e.g. `pixels=(0, 3)` extracts the 1st, 2nd and 3rd
            FPR rows)
        """
        fpr_array_list = self.array_2d_list_from(array=array, pixels=pixels)
        return np.ma.mean(np.ma.asarray(fpr_array_list), axis=0)

    def binned_array_1d_from(
        self, array: aa.Array2D, pixels: Tuple[int, int]
    ) -> np.ndarray:
        """
        Extract a region (e.g. the parallel FPR) of every charge injection region on the charge injection image, stack
        them by taking their mean and then bin them up to a 1D region (e.g. the 1D parallel FPR) by taking the mean
        across the direction opposite to clocking (e.g. bin over the serial direction for a parallel FPR).

        This returns the 1D average region (e.g. of the parallel FPR) of all of the charge injection regions. When
        binning a uniform charge injection this binning process removes noise to clearly reveal the FPR or EPER.
        For non-uniform injections this will provide an average FPR or EPER.

        For fits to charge injection data this function is also used to create images like the stacked 1D residuals,
        which therefore quantify the goodness-of-fit of a CTI model.

        Parameters
        ------------
        array
            The 2D array which contains the charge injeciton image from which the parallel FPRs are extracted and
            stacked.
        pixels
            The row pixel index to extract the FPR between (e.g. `pixels=(0, 3)` extracts the 1st, 2nd and 3rd
            FPR rows)
        """
        front_stacked_array = self.stacked_array_2d_from(array=array, pixels=pixels)
        return np.ma.mean(np.ma.asarray(front_stacked_array), axis=self.binning_axis)

    def add_to_array(
        self, new_array: aa.Array2D, array: aa.Array2D, pixels: Tuple[int, int]
    ) -> aa.Array2D:
        """
        Extracts the parallel FPRs from a charge injection image and adds them to a new image.

        Parameters
        ----------
        new_array
            The 2D array which the extracted parallel FPRs are added to.
        array
            The 2D array which contains the charge injeciton image from which the parallel FPRs are extracted.
        pixels
            The row pixel index which determines the region of the FPR (e.g. `pixels=(0, 3)` will compute the region
            corresponding to the 1st, 2nd and 3rd FPR rows).
        """

        region_list = self.region_list_from(pixels=pixels)

        array_2d_list = self.array_2d_list_from(array=array, pixels=pixels)

        for arr, region in zip(array_2d_list, region_list):
            new_array[region.y0 : region.y1, region.x0 : region.x1] += arr

        return new_array
