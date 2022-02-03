import numpy as np
from typing import List, Tuple

import autoarray as aa

from autocti.charge_injection.extractor_2d.abstract import Extractor2D


class Extractor2DSerialEPER(Extractor2D):
    def region_list_from(self, pixels: Tuple[int, int]):
        """
        Extract the serial EPERs of every charge injection region on the charge injection image and return as a list
        of 2D arrays.

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
           [...][c1c1cc1c1cc1cc1ccc1cc][st1]
        |  [...][1c1c1cc1c1cc1ccc1cc1c][ts0]    |
        |  [...][ttttttttttttttttttttt][sss]    | Direction
       Par [...][ttttttttttttttttttttt][sss]    | of
        |  [...][0ccc0cccc0cccc0cccc0c][st1]    | clocking
       \/  [...][cc0ccc0cccc0cccc0cccc][ts0]   \/

        []     [=====================]
               <---------Ser--------

        The extracted regions correspond to the first serial EPER all charge injection regions:

        region_list[0] = [0, 2, 22, 225 (serial prescan is 3 pixels)
        region_list[1] = [4, 6, 22, 25] (serial prescan is 3 pixels)

        For `pixels=(0,1)` the extracted arrays returned via the `array_2d_list_from()` function keep the first
        serial EPER of each charge injection region:

        array_2d_list[0] = [st0]
        array_2d_list[1] = [st1]

        Parameters
        ----------
        pixels
            The column indexes to extract the trails between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd columns)
        """

        return list(
            map(
                lambda ci_region: ci_region.serial_trailing_region_from(pixels),
                self.region_list,
            )
        )

    def stacked_array_2d_from(
        self, array: aa.Array2D, pixels: Tuple[int, int]
    ) -> np.ndarray:
        front_arrays = self.array_2d_list_from(array=array, pixels=pixels)
        return np.ma.mean(np.ma.asarray(front_arrays), axis=0)

    def binned_array_1d_from(
        self, array: aa.Array2D, pixels: Tuple[int, int]
    ) -> np.ndarray:
        trails_stacked_array = self.stacked_array_2d_from(array=array, pixels=pixels)
        return np.ma.mean(np.ma.asarray(trails_stacked_array), axis=0)

    def add_to_array(
        self, new_array: aa.Array2D, array: aa.Array2D, pixels: Tuple[int, int]
    ):

        region_list = [
            region.serial_trailing_region_from(pixels=pixels)
            for region in self.region_list
        ]

        array_2d_list = self.array_2d_list_from(array=array, pixels=pixels)

        for arr, region in zip(array_2d_list, region_list):
            new_array[region.y0 : region.y1, region.x0 : region.x1] += arr

        return new_array
