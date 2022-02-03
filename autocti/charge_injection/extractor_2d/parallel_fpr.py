import numpy as np
from typing import List, Tuple

import autoarray as aa

from autocti.charge_injection.extractor_2d.abstract import Extractor2D


class Extractor2DParallelFPR(Extractor2D):
    def region_list_from(self, pixels: Tuple[int, int]) -> List[aa.Region2D]:
        """
        Returns a list of the 2D parallel FPR regions given the `Extractor`'s list of charge injection regions, where
        a 2D region is defined following the conventio:

        (top-row, bottom-row, left-column, right-column) = (y0, y1, x0, x1)

        For parallel FPR's the charge spans all columns of the charge injection region, thus the coordinates x0 and x1
        do not change. y0 and y1 are updated based on the `pixels` input.

         scans of a charge injection array.

        The diagram below illustrates the extraction for `pixels=(0, 1)`:

        [] = read-out electronics
        [==========] = read-out register
        [..........] = serial prescan
        [pppppppppp] = parallel overscan
        [ssssssssss] = serial overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicate the region index)
        [tttttttttt] = parallel / serial charge injection region trail

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
           [...][ttttttttttttttttttttt][sss]
           [...][c1c1cc1c1cc1cc1ccc1cc][sss]
        |  [...][1c1c1cc1c1cc1ccc1cc1][sss]    |
        |  [...][ttttttttttttttttttttt][sss]    | Direction
       Par [...][ttttttttttttttttttttt][sss]    | of
        |  [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
       \/  [...][cc0ccc0cccc0cccc0cccc][sss]    \/

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
        pixels
            The row pixel index which determines the region of the FPR (e.g. `pixels=(0, 3)` will compute the region
            corresponding to the 1st, 2nd and 3rd FPR rows).
        """
        return [
            region.parallel_front_region_from(pixels=pixels)
            for region in self.region_list
        ]

    def stacked_array_2d_from(
        self, array: aa.Array2D, pixels: Tuple[int, int]
    ) -> np.ndarray:
        """
        Extract the parallel FPR of every charge injection region on the charge injection image and stack them by
        taking their mean.

        This returns the 2D average FPR of all of the charge injection regions, which for certain CCD charge injection
        electronics one may expect to be similar.

        For fits to charge injection data this function is also used to create images like the stacked 2D residuals,
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
        fpr_array_list = self.array_2d_list_from(array=array, pixels=pixels)
        return np.ma.mean(np.ma.asarray(fpr_array_list), axis=0)

    def binned_array_1d_from(
        self, array: aa.Array2D, pixels: Tuple[int, int]
    ) -> np.ndarray:
        """
        Extract the parallel FPR of every charge injection region on the charge injection image, stack them by taking
        their mean and then bin them up to a 1D FPR by taking the mean across the serial direction..

        This returns the 1D average FPR of all of the charge injection regions, which for a perfectly uniform CCD
        charge injection electronics therefore bins up to remove noise. For non-uniform injections this will provide
        an average of the FPR.

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
        return np.ma.mean(np.ma.asarray(front_stacked_array), axis=1)

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

        region_list = [
            region.parallel_front_region_from(pixels=pixels)
            for region in self.region_list
        ]

        array_2d_list = self.array_2d_list_from(array=array, pixels=pixels)

        for arr, region in zip(array_2d_list, region_list):
            new_array[region.y0 : region.y1, region.x0 : region.x1] += arr

        return new_array
