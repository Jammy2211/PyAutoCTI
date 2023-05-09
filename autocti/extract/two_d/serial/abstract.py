import numpy as np
from typing import List, Optional, Tuple

import autoarray as aa

from autocti.extract.two_d.abstract import Extract2D
from autocti.extract.settings import SettingsExtract


class Extract2DSerial(Extract2D):
    @property
    def binning_axis(self) -> int:
        """
        The axis over which binning is performed to turn 2D serial data (E.g. an EPER) into 1D data.

        For a serial extract `axis=0` such that binning is performed over the rows containing the EPER.
        """
        return 0

    def _value_list_from(
        self, array: aa.Array2D, value_str: str, settings: SettingsExtract
    ):
        median_list = []

        arr_list = [
            np.ma.array(data=array.native[region.slice], mask=array.mask[region.slice])
            for region in self.region_list_from(settings=settings)
        ]

        arr_stack = np.ma.stack(arr_list)

        for row_index in range(arr_list[0].shape[0]):
            if value_str == "median":
                median_list.append(float(np.ma.median(arr_stack[:, row_index, :])))

        return median_list

    def median_list_from(
        self, array: aa.Array2D, settings: SettingsExtract
    ) -> List[float]:
        """
        Returns the median values of the `Extract2D` object's corresponding region, where the median is taken over
        all rows of the region(s).

        To describe this function we will use the example of estimating the median of the charge lines in charge
        injection data, by taking the median over the rows of charge injection (e.g. each median is anti-aligned with
        the FPR).

        By taking the median of values of the charge injection regions (after accounting for those which have had
        electrons captured and relocated due to CTI), we can therefore estimate whether there is row-to-row variation
        in the charge injection.

        This function does this for every rows of every charge injection. If multiple charge injections are performed,
        the median of all charge regions is taken.

        For example, if there are 3 charge injection regions, this function returns a list where each value
        estimates the charge injection normalization of a given charge injection region. The size of this list
        is therefore the number of charge injection rows.

        The function `median_list_of_lists_from` performs the median over each individual charge injection
        region in each row. It therefore estimates multiple injection normalizations per row. for each individual
        charge region. Which function one uses depends on the properties of the charge injection on the instrumentation.

        Parameters
        ----------
        settings
           The settings used to extract the serial region (e.g. the EPERs), which for example include the `pixels`
           tuple specifying the range of pixel columns they are extracted between.
        """
        return self._value_list_from(
            array=array,
            value_str="median",
            settings=settings,
        )

    def _value_list_of_lists_from(
        self, array: aa.Array2D, value_str: str, settings: SettingsExtract
    ):
        median_lists = []

        arr_list = [
            np.ma.array(data=array.native[region.slice], mask=array.mask[region.slice])
            for region in self.region_list_from(settings=settings)
        ]

        for array_2d in arr_list:
            value_list = []

            for row_index in range(array_2d.shape[0]):
                value = float(np.ma.median(array_2d[row_index, :]))

                if value_str == "median":
                    value_list.append(value)

            median_lists.append(value_list)

        return median_lists

    def median_list_of_lists_from(
        self, array: aa.Array2D, settings: SettingsExtract
    ) -> List[List]:
        """
        Returns the median values of the `Extract2D` object's corresponding region, where the median is taken over
        all row(s) of every individual region.

        To describe this function we will use the example of estimating the median of the charge lines in charge
        injection data, by taking the median over every individual row of charge injection (e.g. aligned with the
        FPR).

        The inner regions of the FPR of each charge injection line informs us of the injected level of
        charge for that injection.

        By taking the median of values of the charge injection regions (after accounting for those which have had
        electrons captured and relocated due to CTI), we can therefore estimate the input charge injection
        normalization.

        This function does this for every row of every individual charge injection for every charge injection region.

        For example, if there are 3 charge injection regions, this function returns a list of lists where the outer
        list contains 3 lists each of which give estimates of the charge injection normalization in a given charge
        injection region. Thd size of the inner list is therefore the number of charge injection rows.

        The function `median_list_from` performs the median over all charge injection region in each
        row and thus estimates a single injection normalization per row. Which function one uses depends on the
        properties of the charge injection on the instrumentation.

        Parameters
        ----------
        settings
           The settings used to extract the serial region (e.g. the EPERs), which for example include the `pixels`
           tuple specifying the range of pixel columns they are extracted between.
        """
        return self._value_list_of_lists_from(
            array=array,
            value_str="median",
            settings=settings,
        )
