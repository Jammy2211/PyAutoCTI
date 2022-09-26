import numpy as np
from typing import List, Tuple

import autoarray as aa

from autocti.extract.two_d.parallel.fpr import Extract2DParallelFPR


class Extract2DParallelFPRCI(Extract2DParallelFPR):
    """
    Extends the parallel 2D First Pixel Response (FPR) extractor class with functionality that extracts
    charge injection data's charge injectionr regions and estimates the properties of the charge injection.
    """

    def injection_norm_lists_from(
        self, array: aa.Array2D, pixels: Tuple[int, int]
    ) -> List[List]:
        """
        The inner regions of the FPR of each charge injection line informs us of the injected level of charge for that
        injection.

        By taking the median of values of the charge injection regions (after accounting for those which have had
        electrons captured and relocated due to CTI), we can therefore estimate the input charge injection
        normalization.

        This function does this for every column of every individual charge injection for every charge injection region.

        For example, if there are 3 charge injection regions, this function returns a list of lists where the outer
        list contains 3 lists each of which give estimates of the charge injection normalization in a given charge
        injection region. Thd size of the inner list is therefore the number of charge injection columns.

        The function `injection_norm_list_from` performs the median over all charge injection region in each
        column and thus estimates a single injection normalization per column. Which function one uses depends on the
        properties of the charge injection on the instrumentation.

        Parameters
        ----------
        array
            The charge injection image from which the charge injection normalizations are estimated.
        pixels
            The row pixel index to extract the FPR between (e.g. `pixels=(0, 3)` extracts the 1st, 2nd and 3rd
            FPR rows). To remove the 10 leading pixels which have lost electrons due to CTI, an input such
            as `pixels=(10, 30)` would be used.
        """
        injection_norm_lists = []

        arr_list = [
            np.ma.array(data=array.native[region.slice], mask=array.mask[region.slice])
            for region in self.region_list_from(pixels=pixels)
        ]

        for array_2d in arr_list:

            injection_of_array_list = []

            for column_index in range(array_2d.shape[1]):

                injection_norm = float(np.ma.median(array_2d[:, column_index]))

                injection_of_array_list.append(injection_norm)

            injection_norm_lists.append(injection_of_array_list)

        return injection_norm_lists

    def injection_norm_list_from(
        self, array: aa.Array2D, pixels: Tuple[int, int]
    ) -> List[List]:
        """
        The inner regions of the FPR of each charge injection line informs us of the injected level of charge for that
        injection.

        By taking the median of values of the charge injection regions (after accounting for those which have had
        electrons captured and relocated due to CTI), we can therefore estimate the input charge injection
        normalization.

        This function does this for every column of every charge injection. If multiple charge injections are performed
        per column, the median of all charge regions is taken.

        For example, if there are 3 charge injection regions, this function returns a list where each value
        estimates the charge injection normalization of a given charge injection region. The size of this list
        is therefore the number of charge injection columns.

        The function `injection_norm_lists_from` performs the median over each individual charge injection
        region in each column. It therefore estimates multiple injection normalizations per column. for each individual
        charge region. Which function one uses depends on the properties of the charge injection on the instrumentation.

        Parameters
        ----------
        array
            The charge injection image from which the charge injection normalizations are estimated.
        pixels
            The row pixel index to extract the FPR between (e.g. `pixels=(0, 3)` extracts the 1st, 2nd and 3rd
            FPR rows). To remove the 10 leading pixels which have lost electrons due to CTI, an input such
            as `pixels=(10, 30)` would be used.
        """
        injection_norm_list = []

        arr_list = [
            np.ma.array(data=array.native[region.slice], mask=array.mask[region.slice])
            for region in self.region_list_from(pixels=pixels)
        ]

        for column_index in range(arr_list[0].shape[1]):

            for i, array_2d in enumerate(arr_list):

                if i == 0:
                    arr = array_2d[:, column_index]
                else:
                    arr = np.concatenate((arr[:], array_2d[:, column_index]))

            injection_norm_list.append(float(np.ma.median(arr)))

        return injection_norm_list
