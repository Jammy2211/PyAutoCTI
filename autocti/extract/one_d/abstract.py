import numpy as np
from typing import List, Optional, Tuple

import autoarray as aa


class Extract1D:
    def __init__(
        self,
        region_list: Optional[aa.type.Region1DList] = None,
        prescan: Optional[aa.type.Region1DLike] = None,
        overscan: Optional[aa.type.Region1DLike] = None,
    ):
        """
        Abstract class containing methods for extracting regions from a 1D line dataset which contains some sort of
        original signal whose profile before CTI is known (e.g. warm pixel, charge injection).

        This uses the `region_list`, which contains the signal's regions in pixel coordinates (x0, x1).

        Parameters
        ----------
        region_list
            Integer pixel coordinates specifying the corners of signal (x0, x1).
        """
        self.region_list = (
            list(map(aa.Region1D, region_list)) if region_list is not None else None
        )

        if isinstance(prescan, tuple):
            prescan = aa.Region1D(region=prescan)

        if isinstance(overscan, tuple):
            overscan = aa.Region1D(region=overscan)

        self.prescan = prescan
        self.overscan = overscan

    @property
    def total_pixels_min(self) -> int:
        """
        The number of rows between the read-out electronics and the signal closest to them.
        """
        return np.min([region.total_pixels for region in self.region_list])

    def region_list_from(
        self,
        pixels: Optional[Tuple[int, int]] = None,
        pixels_from_end: Optional[int] = None,
    ) -> List[aa.Region2D]:
        raise NotImplementedError

    def array_1d_list_from(
        self,
        array: aa.Array1D,
        pixels: Optional[Tuple[int, int]] = None,
        pixels_from_end: Optional[int] = None,
    ) -> List[aa.Array1D]:
        """
        Extract a specific region from every region on the line dataset and return as a list of 1D arrays.

        For example, this might extract the EPERs trailing every signal.

        The `region_1d_list_from()` of each `Extract1D` class describes the exact extraction performed for each
        extract when this function is called.

        Parameters
        ----------
        array
            The array from which the regions are extracted and put into the returned list of rrays.
        pixels
            The integer range of pixels between which the extraction is performed.
        """

        arr_list = [
            array.native[region.slice]
            for region in self.region_list_from(
                pixels=pixels, pixels_from_end=pixels_from_end
            )
        ]

        mask_1d_list = [
            array.mask[region.slice]
            for region in self.region_list_from(
                pixels=pixels, pixels_from_end=pixels_from_end
            )
        ]

        return [
            aa.Array1D(values=arr, mask=mask_1d).native
            for arr, mask_1d in zip(arr_list, mask_1d_list)
        ]

    def stacked_array_1d_from(
        self, array: aa.Array1D, pixels: Tuple[int, int]
    ) -> aa.Array1D:
        """
        Extract a region (e.g. the FPR) of every region on the line dataset and stack them by taking their mean.

        This returns the 1D average of the extracted regions (e.g. the FPRs) of all of the regions, which for certain
        calibration datasets one may expect to be similar.

        For fits to a line dataset this function is also used to create images like the stacked 1D residuals,
        which therefore quantify the goodness-of-fit of a CTI model.

        Parameters
        ----------
        array
            The 1D array which contains the line dataset from which the regions (e.g. the FPRs) are extracted and
            stacked.
        pixels
            The pixel index to extract the region between (e.g. `pixels=(0, 3)` extracts the 1st, 2nd and 3rd
            FPR rows)
        """

        arr_list = [
            np.ma.array(data=array.native[region.slice], mask=array.mask[region.slice])
            for region in self.region_list_from(pixels=pixels)
        ]

        stacked_array_1d = np.ma.mean(np.ma.asarray(arr_list), axis=0)

        mask_1d_list = [
            array.mask[region.slice] for region in self.region_list_from(pixels=pixels)
        ]

        return aa.Array1D(
            values=np.asarray(stacked_array_1d.data),
            mask=sum(mask_1d_list) == len(mask_1d_list),
        ).native

    def add_to_array(
        self, new_array: aa.Array1D, array: aa.Array1D, pixels: Tuple[int, int]
    ) -> aa.Array1D:
        """
        Extracts the region (e.g. the FPRs) from the line dataset and adds them to a line dataset.

        Parameters
        ----------
        new_array
            The 1D array which the extracted region (e.g. the FPRs) are added to.
        array
            The 1D array which contains the 1D line dataset from which the region (e.g. the FPRs) are extracted.
        pixels
            The row pixel index which determines the region extracted (e.g. `pixels=(0, 3)` will compute the region
            corresponding to the 1st, 2nd and 3rd pixels).
        """
        region_list = self.region_list_from(pixels=pixels)

        array_1d_list = self.array_1d_list_from(array=array, pixels=pixels)

        for arr, region in zip(array_1d_list, region_list):
            new_array[region.x0 : region.x1] += arr

        return new_array
