import numpy as np
from typing import List, Optional, Tuple

import autoarray as aa

from autocti.extract.two_d.abstract import Extract2D
from autocti.extract.settings import SettingsExtract

from autocti import exc


class Extract2DParallel(Extract2D):
    @property
    def binning_axis(self) -> int:
        """
        The axis over which binning is performed to turn 2D parallel data (e.g. an EPER) into 1D data.

        For a parallel extract `axis=1` such that binning is performed over the rows containing the EPER.
        """
        return 1

    def row_column_index_list_of_lists_from(self, settings: SettingsExtract):
        """
        The function `_value_list_of_lists_from` extracts a list of lists of the mean of values from a 2D array.

        This function returns the row pixel indexes of the data used to create this list of lists. For example, if
        two FPR regions are extracted from rows 100 -> 300 and 400 -> 600, the returned list of lists would be
        two lists, one running from 100 -> 300 and the other 400 -> 600.

        Parameters
        ----------
        settings
            The settings used to extract the parallel data, for example specifying which pixels are used to extract
            the parallel data.
        """
        return [
            list(range(region.y0, region.y1))
            for region in self.region_list_from(settings=settings)
        ]

    def _value_list_from(
        self, array: aa.Array2D, value_str: str, settings: SettingsExtract
    ):
        value_list = []

        arr_list = [
            array.native[region.slice]
            for region in self.region_list_from(settings=settings)
        ]

        mask_list = [
            array.mask[region.slice]
            for region in self.region_list_from(settings=settings)
        ]

        arr_stack = np.stack(arr_list)
        mask_stack = np.stack(mask_list)

        for column_index in range(arr_list[0].shape[1]):
            arr_extract = arr_stack[:, :, column_index]
            mask_extract = mask_stack[:, :, column_index]

            arr_unmasked = arr_extract[np.invert(mask_extract)]

            if value_str == "median":
                value_list.append(float(np.median(arr_unmasked)))
            elif value_str == "mean":
                value_list.append(float(np.mean(arr_unmasked)))
            elif value_str == "std":
                value_list.append(float(np.std(arr_unmasked)))

        return value_list

    def median_list_from(
        self, array: aa.Array2D, settings: SettingsExtract
    ) -> List[float]:
        """
        Returns the median values of the `Extract2D` object's corresponding region, where the median is taken over
        all columns of the region(s).

        To describe this function we will use the example of estimating the median of the charge lines in charge
        injection data, by taking the median over every individual column of charge injection (e.g. aligned with the
        FPR).

        The inner regions of the FPR of each charge injection line informs us of the injected level of charge for
        that injection.

        By taking the median of values of the charge injection regions (after accounting for those which have had
        electrons captured and relocated due to CTI), we can therefore estimate the input charge injection
        normalization.

        This function does this for every column of every charge injection. If multiple charge injections are performed
        per column, the median of all charge regions is taken.

        For example, if there are 3 charge injection regions, this function returns a list where each value
        estimates the charge injection normalization of a given charge injection region. The size of this list
        is therefore the number of charge injection columns.

        The function `median_list_of_lists_from` performs the median over each individual charge injection
        region in each column. It therefore estimates multiple injection normalizations per column. for each individual
        charge region. Which function one uses depends on the properties of the charge injection on the instrumentation.

        Parameters
        ----------
        settings
           The settings used to extract the parallel region (e.g. the EPERs), which for example include the `pixels`
           tuple specifying the range of pixel rows they are extracted between.
        """
        return self._value_list_from(
            array=array,
            settings=settings,
            value_str="median",
        )

    def std_list_from(
        self, array: aa.Array2D, settings: SettingsExtract
    ) -> List[float]:
        """
        Returns the standard deviation (sigma) values of the `Extract2D` object's corresponding region, where the
        standard deviation is taken over all columns of the region(s).

        To describe this function we will use the example of estimating the noise of the charge lines in
        charge injection data, by taking the standard deviation over every individual column of charge
        injection (e.g. aligned with the FPR).

        The inner regions of the FPR of each charge injection line informs us of the noise in the injected level of
        charge for that injection.

        By taking the standard deviation of values of the charge injection regions (after accounting for those which
        have had electrons captured and relocated due to CTI), we can therefore estimate the input charge injection
        normalization.

        This function does this for every column of every charge injection. If multiple charge injections are performed
        per column, the standard deviation of all charge regions is taken.

        For example, if there are 3 charge injection regions, this function returns a list where each value
        estimates the charge injection noise of a given charge injection region. The size of this list
        is therefore the number of charge injection columns.

        The function `sigma_list_of_lists_from` performs the standard deviation over each individual charge injection
        region in each column. It therefore estimates multiple injection noise per column. for each individual
        charge region. Which function one uses depends on the properties of the charge injection on the instrumentation.

        Parameters
        ----------
        settings
           The settings used to extract the parallel region (e.g. the EPERs), which for example include the `pixels`
           tuple specifying the range of pixel rows they are extracted between.
        """
        return self._value_list_from(array=array, settings=settings, value_str="std")

    def _value_list_of_lists_from(
        self,
        array: aa.Array2D,
        settings: SettingsExtract,
        value_str: str,
    ):
        value_lists = []

        arr_list = [
            array.native[region.slice]
            for region in self.region_list_from(settings=settings)
        ]

        mask_list = [
            array.mask[region.slice]
            for region in self.region_list_from(settings=settings)
        ]

        for array_2d, mask in zip(arr_list, mask_list):
            value_list = []

            for column_index in range(array_2d.shape[1]):
                arr_extract = array_2d[:, column_index]
                mask_extract = mask[:, column_index]

                arr_unmasked = arr_extract[np.invert(mask_extract)]

                if value_str == "median":
                    value = float(np.median(arr_unmasked))
                elif value_str == "mean":
                    value = float(np.mean(arr_unmasked))
                elif value_str == "std":
                    value = float(np.std(arr_unmasked))

                value_list.append(value)

            value_lists.append(value_list)

        return value_lists

    def median_list_of_lists_from(
        self, array: aa.Array2D, settings: SettingsExtract
    ) -> List[List]:
        """
        Returns the median values of the `Extract2D` object's corresponding region, where the median is taken over
        all column(s) of every individual region.

        To describe this function we will use the example of estimating the median of the charge lines in charge
        injection data, by taking the median over every individual column of charge injection (e.g. aligned with the
        FPR).

        The inner regions of the FPR of each charge injection line informs us of the injected level of
        charge for that injection.

        By taking the median of values of the charge injection regions (after accounting for those which have had
        electrons captured and relocated due to CTI), we can therefore estimate the input charge injection
        normalization.

        This function does this for every column of every individual charge injection for every charge injection region.

        For example, if there are 3 charge injection regions, this function returns a list of lists where the outer
        list contains 3 lists each of which give estimates of the charge injection normalization in a given charge
        injection region. Thd size of the inner list is therefore the number of charge injection columns.

        The function `median_list_from` performs the median over all charge injection region in each
        column and thus estimates a single injection normalization per column. Which function one uses depends on the
        properties of the charge injection on the instrumentation.

        Parameters
        ----------
        settings
           The settings used to extract the parallel region (e.g. the EPERs), which for example include the `pixels`
           tuple specifying the range of pixel rows they are extracted between.
        """
        return self._value_list_of_lists_from(
            array=array,
            settings=settings,
            value_str="median",
        )

    def std_list_of_lists_from(
        self, array: aa.Array2D, settings: SettingsExtract
    ) -> List[List]:
        """
        Returns the standard deviation (sigma) values of the `Extract2D` object's corresponding region, where the
        standard deviation is taken over all column(s) of every individual region.

        To describe this function we will use the example of estimating the noise of the charge lines in charge
        injection data, by taking the standard deviation over every individual column of charge injection (e.g.
        aligned with the FPR).

        The inner regions of the FPR of each charge injection line informs us of the noise level of
        charge for that injection.

        By taking the standard deviation of values of the charge injection regions (after accounting for those which have had
        electrons captured and relocated due to CTI), we can therefore estimate the input charge injection
        noise level.

        This function does this for every column of every individual charge injection for every charge injection region.

        For example, if there are 3 charge injection regions, this function returns a list of lists where the outer
        list contains 3 lists each of which give estimates of the charge injection noise in a given charge
        injection region. Thd size of the inner list is therefore the number of charge injection columns.

        The function `std_list_from` performs the standard deviation over all charge injection region in each
        column and thus estimates a single injection noise level per column. Which function one uses depends on the
        properties of the charge injection on the instrumentation.

        Parameters
        ----------
        settings
           The settings used to extract the parallel region (e.g. the EPERs), which for example include the `pixels`
           tuple specifying the range of pixel rows they are extracted between.
        """
        return self._value_list_of_lists_from(
            array=array, settings=settings, value_str="std"
        )

    def _value_list_of_lists_via_array_list_from(
        self,
        array_list: List[aa.Array2D],
        settings: SettingsExtract,
        value_str: str,
    ):
        value_lists = [
            self._value_list_of_lists_from(
                array=array, settings=settings, value_str=value_str
            )
            for array in array_list
        ]

        value_lists = np.asarray(value_lists)

        if value_str == "median":
            return np.median(value_lists, axis=0).tolist()
        elif value_str == "std":
            return np.std(value_lists, axis=0).tolist()

    def median_list_of_lists_via_array_list_from(
        self, array_list: List[aa.Array2D], settings: SettingsExtract
    ) -> List[List]:
        """
        Returns the median values of the `Extract2D` object's corresponding region, where the median is taken over
        all column(s) of every individual region.

        To describe this function we will use the example of estimating the median of the charge lines in charge
        injection data, by taking the median over every individual column of charge injection (e.g. aligned with the
        FPR).

        The inner regions of the FPR of each charge injection line informs us of the injected level of
        charge for that injection.

        By taking the median of values of the charge injection regions (after accounting for those which have had
        electrons captured and relocated due to CTI), we can therefore estimate the input charge injection
        normalization.

        This function does this for every column of every individual charge injection for every charge injection region.

        For example, if there are 3 charge injection regions, this function returns a list of lists where the outer
        list contains 3 lists each of which give estimates of the charge injection normalization in a given charge
        injection region. Thd size of the inner list is therefore the number of charge injection columns.

        The function `median_list_from` performs the median over all charge injection region in each
        column and thus estimates a single injection normalization per column. Which function one uses depends on the
        properties of the charge injection on the instrumentation.

        Parameters
        ----------
        settings
           The settings used to extract the parallel region (e.g. the EPERs), which for example include the `pixels`
           tuple specifying the range of pixel rows they are extracted between.
        """
        return self._value_list_of_lists_via_array_list_from(
            array_list=array_list,
            settings=settings,
            value_str="median",
        )
