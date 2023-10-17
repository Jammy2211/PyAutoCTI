import copy
import numpy as np
from typing import List, Optional, Tuple

import autoarray as aa

from autocti.extract.settings import SettingsExtract


class Extract1D:
    def __init__(
        self,
        shape_1d: Optional[Tuple[int]] = None,
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

        self.shape_1d = shape_1d

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

    @property
    def parallel_rows_between_regions(self):
        return [
            self.region_list[i + 1].x0 - self.region_list[i].x1
            for i in range(len(self.region_list) - 1)
        ]

    @property
    def trail_size_to_array_edge(self):
        return self.shape_1d[0] - np.max([region.x1 for region in self.region_list])

    def region_list_from(self, settings: SettingsExtract) -> List[aa.Region2D]:
        raise NotImplementedError

    def array_1d_list_from(
        self, array: aa.Array1D, settings: SettingsExtract
    ) -> List[aa.Array1D]:
        """
        Extract a specific region from every region on the line dataset and return as a list of 1D arrays.

        For example, this might extract the EPERs trailing every signal.

        The `region_list_from()` of each `Extract1D` class describes the exact extraction performed for each
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
            for region in self.region_list_from(settings=settings)
        ]

        mask_1d_list = [
            array.mask[region.slice]
            for region in self.region_list_from(settings=settings)
        ]

        return [
            aa.Array1D(values=arr, mask=mask_1d).native
            for arr, mask_1d in zip(arr_list, mask_1d_list)
        ]

    def stacked_array_1d_from(
        self, array: aa.Array1D, settings: SettingsExtract
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

        arr_list = np.asarray(
            [
                np.array(array.native[region.slice])
                for region in self.region_list_from(settings=settings)
            ]
        )

        mask_1d_list = [
            array.mask[region.slice]
            for region in self.region_list_from(settings=settings)
        ]

        stacked_array_1d = np.mean(
            np.asarray(arr_list), axis=0, where=np.invert(np.asarray(mask_1d_list))
        )

        return aa.Array1D(
            values=np.asarray(stacked_array_1d.data),
            mask=sum(mask_1d_list) == len(mask_1d_list),
        ).native

    def add_to_array(
        self, new_array: aa.Array1D, array: aa.Array1D, settings: SettingsExtract
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
        region_list = self.region_list_from(settings=settings)

        array_1d_list = self.array_1d_list_from(array=array, settings=settings)

        for arr, region in zip(array_1d_list, region_list):
            new_array[region.x0 : region.x1] += arr

        return new_array

    def add_gaussian_noise_to(
        self,
        array: aa.Array1D,
        settings: SettingsExtract,
        noise_sigma: float,
        noise_seed: int = -1,
    ) -> aa.Array2D:
        """
        Adds Gaussian noise of an input sigma value to the regions of the `Extract` object and returns the overall
        input array with this noise added.

        Parameters
        ----------
        array
            The 1D array which contains the charge injection where regions of Gaussian noise is added.
        noise_sigma
            The sigma value (standard deviation) of the Gaussian from which noise values are drann.
        noise_seed
            The seed of the random number generator, used for the random noises maps.
        pixels
            The row pixel index which determines the region of the FPR (e.g. `pixels=(0, 3)` will compute the region
            corresponding to the 1st, 2nd and 3rd FPR rows).
        pixels_from_end
            Alternative row pixex index specification, which extracts this number of pixels from the end of
            each region (e.g. FPR / EPER). For example, if each FPR is 100 pixels and `pixels_from_end=10`, the
            last 10 pixels of each FPR (pixels (90, 100)) are extracted.
        """

        region_list = self.region_list_from(settings=settings)

        array_1d_list = self.array_1d_list_from(array=array, settings=settings)

        array = copy.copy(array.native)

        for arr, region in zip(array_1d_list, region_list):
            array[region.x0 : region.x1] = aa.preprocess.data_with_gaussian_noise_added(
                data=arr, sigma=noise_sigma, seed=noise_seed
            )

        return array
