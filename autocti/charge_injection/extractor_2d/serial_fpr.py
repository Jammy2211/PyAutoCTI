import numpy as np
from typing import List, Tuple

import autoarray as aa

from autocti.charge_injection.extractor_2d.abstract import Extractor2D


class Extractor2DSerialFPR(Extractor2D):
    def region_list_from(self, pixels: Tuple[int, int]):
        """
        Extract the serial FPR of every charge injection region on the charge injection image and return as a list
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
           [...][c1c1cc1c1cc1cc1ccc1cc][sss]
        |  [...][1c1c1cc1c1cc1ccc1cc1c][sss]    |
        |  [...][ttttttttttttttttttttt][sss]    | Direction
       Par [...][ttttttttttttttttttttt][sss]    | of
        |  [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
       \/  [...][cc0ccc0cccc0cccc0cccc][sss]    \/

        []     [=====================]
               <---------Ser--------

        The extracted regions correspond to the first serial FPR all charge injection regions:

        region_list[0] = [0, 2, 3, 21] (serial prescan is 3 pixels)
        region_list[1] = [4, 6, 3, 21] (serial prescan is 3 pixels)

        For `pixels=(0,1)` the extracted arrays returned via the `array_2d_list_from()` function keep the first
        serial FPR of each charge injection region:

        array_2d_list[0] =[c0c0]
        array_2d_list[1] =[1c1c]

        Parameters
        ------------
        arrays
        pixels
            The column indexes to extract the front edge between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd
            columns)
        """
        return list(
            map(
                lambda ci_region: ci_region.serial_front_region_from(pixels),
                self.region_list,
            )
        )

    def stacked_array_2d_from(self, array: aa.Array2D, pixels: Tuple[int, int]):
        front_arrays = self.array_2d_list_from(array=array, pixels=pixels)
        return np.ma.mean(np.ma.asarray(front_arrays), axis=0)

    def binned_array_1d_from(self, array: aa.Array2D, pixels: Tuple[int, int]):
        front_stacked_array = self.stacked_array_2d_from(array=array, pixels=pixels)
        return np.ma.mean(np.ma.asarray(front_stacked_array), axis=0)

    def add_to_array(
        self, new_array: aa.Array2D, array: aa.Array2D, pixels: Tuple[int, int]
    ):

        region_list = [
            region.serial_front_region_from(pixels=pixels)
            for region in self.region_list
        ]

        array_2d_list = self.array_2d_list_from(array=array, pixels=pixels)

        for arr, region in zip(array_2d_list, region_list):
            new_array[region.y0 : region.y1, region.x0 : region.x1] += arr

        return new_array
