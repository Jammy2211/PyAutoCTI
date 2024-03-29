from typing import List

import numpy as np

import autoarray as aa

from autocti.extract.two_d.parallel.abstract import Extract2DParallel
from autocti.extract.settings import SettingsExtract

from autocti.extract.two_d import extract_2d_util


@aa.numba_util.jit()
def capture_estimate_from(
    array, mask, x0, x1, y0, y1, pixels_from_start, pixels_from_end
):
    value_list = []

    for x in range(x0, x1):
        fpr = array[y0:y1, x]
        mask_ = mask[y0:y1, x]

        if np.sum(mask_):
            injection_estimate = np.median(
                fpr[-pixels_from_end:][mask_[-pixels_from_end:]]
            )
            capture_estimate = np.sum(
                injection_estimate
                - fpr[0:pixels_from_start][mask_[0:pixels_from_start]]
            )

            value_list.append(capture_estimate)

    return value_list


class Extract2DParallelFPR(Extract2DParallel):
    def region_list_from(self, settings: SettingsExtract) -> List[aa.Region2D]:
        """
        Returns a list of the 2D parallel FPR regions from the `region_list` containing signal  (e.g. the charge
        injection regions of charge injection data), between two input `pixels` indexes.

        Negative pixel values can be input into the `pixels` tuple, whereby columns in front of the parallel FPRs
        are extracted.

        A 2D region is defined following the convention:

        (top-row, bottom-row, left-column, right-column) = (y0, y1, x0, x1)

        For parallel FPR's the charge spans all columns of the charge injection region, thus the coordinates x0 and x1
        do not change. y0 and y1 are updated based on the `pixels` input.

        Negative pixel values can be input into the `pixels` tuple, whereby rows in front of the parallel FPRs are
        also extracted.

        The diagram below illustrates the extraction for `pixels=(0, 1)`:

        [] = read-out electronics
        [==========] = read-out register
        [..........] = serial prescan
        [pppppppppp] = parallel overscan
        [ssssssssss] = serial overscan
        [f#ff#f#f#f] = signal region (FPR) (0 / 1 indicate the region index)
        [tttttttttt] = parallel / serial charge injection region trail

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
           [...][ttttttttttttttttttttt][sss]
           [...][c1c1cc1c1cc1cc1ccc1cc][sss]
        |  [...][1c1c1cc1c1cc1ccc1cc1][sss]    |
        |  [...][ttttttttttttttttttttt][sss]    | Direction
        Par [...][ttttttttttttttttttttt][sss]    | of
        |  [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
        |/  [...][cc0ccc0cccc0cccc0cccc][sss]    \/

        []     [=====================]
               <---------Ser--------

        The extracted regions correspond to the first parallel FPR all charge injection regions:

        region_list[0] = [0, 1, 3, 21] (serial prescan is 3 pixels)
        region_list[1] = [3, 4, 3, 21] (serial prescan is 3 pixels)

        For `pixels=(0,1)` the extracted arrays returned via the `array_2d_list_from()` function keep the first
        parallel FPR of each charge injection region:

        array_2d_list[0] = [c0c0c0cc0c0c0c0c0c0c0]
        array_2d_list[1] = [1c1c1c1c1c1c1c1c1c1c1]

        Parameters
        ----------
        settings
           The settings used to extract the parallel FPRs, which for example include the `pixels` tuple specifying the
           range of pixel rows they are extracted between.
        """
        return [
            region.parallel_front_region_from(
                pixels=settings.pixels, pixels_from_end=settings.pixels_from_end
            )
            for region in self.region_list
        ]

    def binned_region_1d_from(self, settings: SettingsExtract) -> aa.Region1D:
        """
        The `Extract` objects allow one to extract a `Dataset1D` from a 2D CTI dataset, which is used to perform
        CTI modeling in 1D.

        This is performed by binning up the data via the `binned_array_1d_from` function.

        In order to create the 1D dataset a `Layout1D` is required, which requires the `region_list` containing the
        charge regions on the 1D dataset (e.g. where the FPR appears in 1D after binning).

        The function returns the this region if the 1D dataset is extracted from the parallel FPRs. This is
        the full range of the `pixels` tuple, unless negative entries are included, meaning that pixels
        before the FPRs are extracted.

        Parameters
        ----------
        settings
           The settings used to extract the parallel FPR, which for example include the `pixels` tuple specifying the
           range of pixel rows they are extracted between.
        """
        return extract_2d_util.binned_region_1d_fpr_from(pixels=settings.pixels)

    def capture_estimate_from(
        self, array: aa.Array2D, pixels_from_start, pixels_from_end
    ):
        """
        Estimates the total number of electrons captured in each column of the parallel FPRs.

        This is performed by the following steps:

        1) Go to each individual parallel FPR.
        2) Estimate the injection level by taking the median of the final N pixels (N = `pixels_from_end`).
        3) Subtracted this estimate from all N pixels at the start of the parallel FPR which will have had electrons
           captured (N = `pixels_from_start`).
        4) Sum the difference thereby estimating of the total number of electrons captured in the parallel FPR.
        5) Return the median of all individual parallel FPR capture estimates.

        This routine acts on every individual parallel FPR because charge injection non-uniformity means that using
        averages over the full charge injection region will not be accurate.

        The function also omits masked pixels, which are typically due to cosmic rays.

        Parameters
        ----------
        array
            The array contain the charge injection data whose parallel FPRs are used to estimate the capture.
        pixels_from_start
            The number of pixels at the start of the parallel FPR which are used to estimate the capture.
        pixels_from_end
            The number of pixels at the end of the parallel FPR which are used to estimate the injection level.

        Returns
        -------
        An estimate of the total number of electrons captured in each column of the parallel FPRs.
        """
        capture_list = []

        for region in self.region_list:
            capture_list_ = capture_estimate_from(
                array=np.array(array.native),
                mask=np.invert(array.mask),
                x0=int(region.x0),
                x1=int(region.x1),
                y0=int(region.y0),
                y1=int(region.y1),
                pixels_from_start=pixels_from_start,
                pixels_from_end=pixels_from_end,
            )

            capture_list = capture_list + capture_list_

        return np.median(capture_list)
