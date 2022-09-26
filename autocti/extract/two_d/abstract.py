import numpy as np
from typing import Optional, List, Tuple, Union

import autoarray as aa

from autocti.charge_injection.imaging.imaging import ImagingCI
from autocti.dataset_1d.dataset_1d.dataset_1d import Dataset1D
from autocti.layout.one_d import Layout1D


class Extract2D:
    def __init__(
        self,
        region_list: Optional[aa.type.Region2DList] = None,
        parallel_overscan: Optional[aa.type.Region2DLike] = None,
        serial_prescan: Optional[aa.type.Region2DLike] = None,
        serial_overscan: Optional[aa.type.Region2DLike] = None,
    ):
        """
        Abstract class containing methods for extracting regions from a 2D charge injection image.

        This uses the `region_list`, which contains the charge injection regions in pixel coordinates.

        Parameters
        ----------
        region_list
            Integer pixel coordinates specifying the corners of each charge injection region (top-row, bottom-row,
            left-column, right-column).
        """

        self.region_list = (
            list(map(aa.Region2D, region_list)) if region_list is not None else None
        )

        self.parallel_overscan = (
            aa.Region2D(region=parallel_overscan)
            if isinstance(parallel_overscan, tuple)
            else parallel_overscan
        )
        self.serial_prescan = (
            aa.Region2D(region=serial_prescan)
            if isinstance(serial_prescan, tuple)
            else serial_prescan
        )
        self.serial_overscan = (
            aa.Region2D(region=serial_overscan)
            if isinstance(serial_overscan, tuple)
            else serial_overscan
        )

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

        For a parallel extract `axis=1` such that binning is performed over the rows containing the FPR.
        """
        raise NotImplementedError

    def region_list_from(self, pixels: Tuple[int, int]) -> List[aa.Region2D]:
        raise NotImplementedError

    def array_2d_list_from(
        self, array: aa.Array2D, pixels: Tuple[int, int]
    ) -> List[aa.Array2D]:
        """
        Extract a specific region from every signal region (e.g. the charge injection region of charge injection data) on the CTI calibration data and return as a list
        of 2D arrays.

        For example, this might extract the parallel EPERs of every charge injection region.

        The `region_2d_list_from()` of each `Extract2D` class describes the exact extraction performed for each
        extract when this function is called.

        Parameters
        ----------
        array
            The array from which the regions are extracted and put into the returned list of arrays.
        pixels
            The integer range of pixels between which the extraction is performed.
        """
        arr_list = [
            array.native[region.slice]
            for region in self.region_list_from(pixels=pixels)
        ]

        mask_2d_list = [
            array.mask[region.slice] for region in self.region_list_from(pixels=pixels)
        ]

        return [
            aa.Array2D.manual_mask(array=arr, mask=mask_2d).native
            for arr, mask_2d in zip(arr_list, mask_2d_list)
        ]

    def stacked_array_2d_from(
        self, array: aa.Array2D, pixels: Tuple[int, int]
    ) -> np.ndarray:
        """
        Extract a region (e.g. the parallel FPR) of every signal region (e.g. the charge injection region of charge injection data) on the CTI calibration data and
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

        arr_list = [
            np.ma.array(data=array.native[region.slice], mask=array.mask[region.slice])
            for region in self.region_list_from(pixels=pixels)
        ]

        stacked_array_2d = np.ma.mean(np.ma.asarray(arr_list), axis=0)

        mask_2d_list = [
            array.mask[region.slice] for region in self.region_list_from(pixels=pixels)
        ]

        return aa.Array2D.manual_mask(
            array=np.asarray(stacked_array_2d.data),
            mask=sum(mask_2d_list) == len(mask_2d_list),
        ).native

    def stacked_array_2d_total_pixels_from(
        self, array: aa.Array2D, pixels: Tuple[int, int]
    ) -> np.ndarray:
        """
        The function `stacked_array_2d_from` extracts a region (e.g. the parallel FPR) of every charge injection
        region on the charge injection image and stacks them by taking their mean.

        If the data being stacked is a noise-map, we need to know how many pixels were used in the stacking of every
        final pixel on the stacked 2d array in order to compute the new noise map via quadrature.

        Parameters
        ------------
        array
            The 2D array which contains the charge injection image from which the regions (e.g. the parallel FPRs)
            are extracted and stacked.
        pixels
            The row pixel index to extract the FPR between (e.g. `pixels=(0, 3)` extracts the 1st, 2nd and 3rd
            FPR rows)
        """

        mask_2d_list = [
            array.mask[region.slice] for region in self.region_list_from(pixels=pixels)
        ]

        arr_total_pixels = sum([np.invert(mask_2d) for mask_2d in mask_2d_list])

        return aa.Array2D.manual_mask(
            array=np.asarray(arr_total_pixels),
            mask=sum(mask_2d_list) == len(mask_2d_list),
        ).native

    def binned_array_1d_from(
        self, array: aa.Array2D, pixels: Tuple[int, int]
    ) -> aa.Array1D:
        """
        Extract a region (e.g. the parallel FPR) of every signal region (e.g. the charge injection region of charge injection data) on the CTI calibration data, stack
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
            The column / row pixel index to extract the region (e.g. FPR, EPER) between (e.g. `pixels=(0, 3)` extracts
            the 1st, 2nd and 3rd columns / rows)
        """

        arr_list = [
            np.ma.array(data=array.native[region.slice], mask=array.mask[region.slice])
            for region in self.region_list_from(pixels=pixels)
        ]
        stacked_array_2d = np.ma.mean(np.ma.asarray(arr_list), axis=0)
        binned_array_1d = np.ma.mean(
            np.ma.asarray(stacked_array_2d), axis=self.binning_axis
        )
        return aa.Array1D.manual_native(
            array=binned_array_1d, pixel_scales=array.pixel_scale
        )

    def binned_array_1d_total_pixels_from(
        self, array: aa.Array2D, pixels: Tuple[int, int]
    ) -> aa.Array1D:
        """
        The function `binned_array_1d_from` extracts a region (e.g. the parallel FPR) of every charge injection region
        on the charge injection image, stacks them by taking their mean and then bin them up to a 1D region
        (e.g. the 1D parallel FPR) by taking the mean across the direction opposite to clocking (e.g. bin over the
        serial direction for a parallel FPR).

        If the data being stacked is a noise-map, we need to know how many pixels were used in the stacking of every
        final pixel on the binned 1D array in order to compute the new noise map via quadrature.

        Parameters
        ------------
        array
            The 2D array which contains the charge injeciton image from which the parallel FPRs are extracted and
            stacked.
        pixels
            The column / row pixel index to extract the region (e.g. FPR, EPER) between (e.g. `pixels=(0, 3)` extracts
            the 1st, 2nd and 3rd columns / rows)
        """

        mask_2d_list = [
            array.mask[region.slice] for region in self.region_list_from(pixels=pixels)
        ]

        arr_total_pixels = sum([np.invert(mask_2d) for mask_2d in mask_2d_list])

        binned_total_pixels = np.ma.sum(
            np.ma.asarray(arr_total_pixels), axis=self.binning_axis
        )

        return aa.Array1D.manual_native(
            array=binned_total_pixels, pixel_scales=array.pixel_scale
        )

    def binned_region_1d_from(self, pixels: Tuple[int, int]) -> aa.Region1D:
        raise NotImplementedError

    def add_to_array(
        self, new_array: aa.Array2D, array: aa.Array2D, pixels: Tuple[int, int]
    ) -> aa.Array2D:
        """
        Extracts the region (e.g. the parallel FPRs) from a charge injection image and adds them to a new image.

        Parameters
        ----------
        new_array
            The 2D array which the extracted parallel FPRs are added to.
        array
            The 2D array which contains the charge injection image from which the parallel FPRs are extracted.
        pixels
            The row pixel index which determines the region of the FPR (e.g. `pixels=(0, 3)` will compute the region
            corresponding to the 1st, 2nd and 3rd FPR rows).
        """

        region_list = self.region_list_from(pixels=pixels)

        array_2d_list = self.array_2d_list_from(array=array, pixels=pixels)

        for arr, region in zip(array_2d_list, region_list):
            new_array[region.y0 : region.y1, region.x0 : region.x1] += arr

        return new_array

    def dataset_1d_from(
        self, dataset_2d: ImagingCI, pixels: Tuple[int, int]
    ) -> Dataset1D:

        binned_data_1d = self.binned_array_1d_from(array=dataset_2d.data, pixels=pixels)

        binned_noise_map_1d = self.binned_array_1d_from(
            array=dataset_2d.noise_map, pixels=pixels
        )

        binned_noise_map_1d_total_pixels = self.binned_array_1d_total_pixels_from(
            array=dataset_2d.noise_map, pixels=pixels
        )

        binned_noise_map_1d /= np.sqrt(binned_noise_map_1d_total_pixels)

        binned_pre_cti_data_1d = self.binned_array_1d_from(
            array=dataset_2d.pre_cti_data, pixels=pixels
        )

        binned_region_1d = self.binned_region_1d_from(pixels=pixels)

        layout_1d = Layout1D(
            shape_1d=binned_data_1d.shape_native, region_list=[binned_region_1d]
        )

        return Dataset1D(
            data=binned_data_1d,
            noise_map=binned_noise_map_1d,
            pre_cti_data=binned_pre_cti_data_1d,
            layout=layout_1d,
        )
