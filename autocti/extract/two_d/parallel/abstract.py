import numpy as np
from typing import List, Optional, Tuple

import autoarray as aa


from autocti.extract.two_d.abstract import Extract2D

from autocti import exc


class Extract2DParallel(Extract2D):
    @property
    def binning_axis(self) -> int:
        """
        The axis over which binning is performed to turn 2D parallel data (e.g. an EPER) into 1D data.

        For a parallel extract `axis=1` such that binning is performed over the rows containing the EPER.
        """
        return 1

    def _value_list_from(
        self,
        array: aa.Array2D,
        value_str: str,
        pixels: Optional[Tuple[int, int]] = None,
        pixels_from_end: Optional[int] = None,
    ):

        value_list = []

        if pixels_from_end is None:

            arr_list = [
                np.ma.array(
                    data=array.native[region.slice], mask=array.mask[region.slice]
                )
                for region in self.region_list_from(pixels=pixels)
            ]

        else:

            arr_list = [
                np.ma.array(
                    data=array.native[region.slice], mask=array.mask[region.slice]
                )
                for region in self.region_list_from(pixels_from_end=pixels_from_end)
            ]

        arr_stack = np.ma.stack(arr_list)

        for column_index in range(arr_list[0].shape[1]):

            if value_str == "median":
                value_list.append(float(np.ma.median(arr_stack[:, :, column_index])))
            elif value_str == "std":
                value_list.append(float(np.ma.std(arr_stack[:, :, column_index])))

        return value_list

    def median_list_from(
        self,
        array: aa.Array2D,
        pixels: Optional[Tuple[int, int]] = None,
        pixels_from_end: Optional[int] = None,
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
        array
            The charge injection image from which the charge injection normalizations are estimated.
        pixels
            The row pixel index to extract the region (e.g. FPR / EPER) between. For example, `pixels=(0, 3)`
            extracts the 1st, 2nd and 3rd FPR / EPER rows). To remove the 10 leading pixels which have lost electrons
            due to CTI, `pixels=(10, 30)` would be used.
        pixels_from_end
            Alternative row pixex index specification, which extracts this number of pixels from the end of
            each region (e.g. FPR / EPER). For example, if each FPR is 100 pixels and `pixels_from_end=10`, the
            last 10 pixels of each FPR (pixels (90, 100) are extracted.
        """
        return self._value_list_from(
            array=array,
            pixels=pixels,
            pixels_from_end=pixels_from_end,
            value_str="median",
        )

    def std_list_from(
        self,
        array: aa.Array2D,
        pixels: Optional[Tuple[int, int]] = None,
        pixels_from_end: Optional[int] = None,
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
        array
            The charge injection image from which the charge injection standard deviations / noise are estimated.
        pixels
            The row pixel index to extract the FPR between (e.g. `pixels=(0, 3)` extracts the 1st, 2nd and 3rd
            FPR rows). To remove the 10 leading pixels which have lost electrons due to CTI, an input such
            as `pixels=(10, 30)` would be used.
        """
        return self._value_list_from(
            array=array, pixels=pixels, pixels_from_end=pixels_from_end, value_str="std"
        )

    def _value_list_of_lists_from(
        self,
        array: aa.Array2D,
        value_str: str,
        pixels: Optional[Tuple[int, int]],
        pixels_from_end: Optional[int],
    ):
        value_lists = []

        if pixels_from_end is None:

            arr_list = [
                np.ma.array(
                    data=array.native[region.slice], mask=array.mask[region.slice]
                )
                for region in self.region_list_from(pixels=pixels)
            ]

        else:

            arr_list = [
                np.ma.array(
                    data=array.native[region.slice], mask=array.mask[region.slice]
                )
                for region in self.region_list_from(pixels_from_end=pixels_from_end)
            ]

        for array_2d in arr_list:

            value_list = []

            for column_index in range(array_2d.shape[1]):

                if value_str == "median":
                    value = float(np.ma.median(array_2d[:, column_index]))
                elif value_str == "std":
                    value = float(np.ma.std(array_2d[:, column_index]))

                value_list.append(value)

            value_lists.append(value_list)

        return value_lists

    def median_list_of_lists_from(
        self,
        array: aa.Array2D,
        pixels: Optional[Tuple[int, int]] = None,
        pixels_from_end: Optional[int] = None,
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
        array
            The data from which the median values are estimated.
        pixels
            The row pixel index to extract the region between (e.g. `pixels=(0, 3)` extracts the 1st, 2nd and 3rd
            rows of the region).
        """
        return self._value_list_of_lists_from(
            array=array,
            pixels=pixels,
            pixels_from_end=pixels_from_end,
            value_str="median",
        )

    def std_list_of_lists_from(
        self,
        array: aa.Array2D,
        pixels: Optional[Tuple[int, int]] = None,
        pixels_from_end: Optional[int] = None,
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
        array
            The data from which the standard deviation values are estimated.
        pixels
            The row pixel index to extract the region between (e.g. `pixels=(0, 3)` extracts the 1st, 2nd and 3rd
            rows of the region).
        """
        return self._value_list_of_lists_from(
            array=array, pixels=pixels, pixels_from_end=pixels_from_end, value_str="std"
        )
