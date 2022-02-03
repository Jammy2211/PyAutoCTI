import numpy as np
from typing import List, Tuple

import autoarray as aa

from autocti.charge_injection.extractor_2d.abstract import Extractor2D


class Extractor2DSerialFPR(Extractor2D):
    def array_2d_list_from(self, array: aa.Array2D, columns):
        """
        Extract a list of the serial front edge structures of a charge injection array.

        The diagram below illustrates the arrays that is extracted from a array for columnss=(0, 3):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCDPhase panel    [pppppppppp] = parallel overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicates ci_region index)
        [xxxxxxxxxx]
        [tttttttttt] = parallel / serial charge injection region trail (0 / 1 indicates ci_region index)

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][ttttttttttttttttttttt][sss]
          [...][c1c1cc1c1cc1cc1ccc1cc][sss]
        | [...][1c1c1cc1c1cc1ccc1cc1c][sss]    |
        | [...][ttttttttttttttttttttt][sss]    | Direction
        P [...][ttttttttttttttttttttt][sss]    | of
        | [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
          [...][cc0ccc0cccc0cccc0cccc][sss]    |

        []     [=====================]
               <---------S----------

        The extracted array keeps just the serial front edges of all charge injection scans.

        list index 0:

        [c0c0]

        list index 1:

        [1c1c]

        Parameters
        ----------
        columns
            The column indexes to extract the front edge between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd
            columns)
        """
        front_region_list = self.region_list_from(columns=columns)
        front_arrays = list(
            map(lambda region: array.native[region.slice], front_region_list)
        )
        front_masks = list(
            map(lambda region: array.mask[region.slice], front_region_list)
        )
        front_arrays = list(
            map(
                lambda front_array, front_mask: np.ma.array(
                    front_array, mask=front_mask
                ),
                front_arrays,
                front_masks,
            )
        )
        return front_arrays

    def stacked_array_2d_from(self, array: aa.Array2D, columns):
        front_arrays = self.array_2d_list_from(array=array, columns=columns)
        return np.ma.mean(np.ma.asarray(front_arrays), axis=0)

    def binned_array_1d_from(self, array: aa.Array2D, columns):
        front_stacked_array = self.stacked_array_2d_from(array=array, columns=columns)
        return np.ma.mean(np.ma.asarray(front_stacked_array), axis=0)

    def region_list_from(self, columns):
        """
        Returns a list of the serial front edges scans of a charge injection array.

        The diagram below illustrates the region that is calculated from a array for columns=(0, 4):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCDPhase panel    [pppppppppp] = parallel overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicates ci_region index)
        [xxxxxxxxxx]
        [tttttttttt] = parallel / serial charge injection region trail ((0 / 1 indicates ci_region index)

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][ttttttttttttttttttttt][sss]
          [...][c1c1cc1c1cc1cc1ccc1cc][sss]
        | [...][1c1c1cc1c1cc1ccc1cc1c][sss]    |
        | [...][ttttttttttttttttttttt][sss]    | Direction
        P [...][ttttttttttttttttttttt][sss]    | of
        | [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
          [...][cc0ccc0cccc0cccc0cccc][sss]    |

        []     [=====================]
               <---------S----------

        The extracted array keeps just the serial front edges of all charge injection scans.

        list index 0:

        [0, 2, 3, 21] (serial prescan is 3 pixels)

        list index 1:

        [4, 6, 3, 21] (serial prescan is 3 pixels)

        Parameters
        ------------
        arrays
        columns
            The column indexes to extract the front edge between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd
            columns)
        """
        return list(
            map(
                lambda ci_region: ci_region.serial_front_region_from(columns),
                self.region_list,
            )
        )

    def add_to_array(self, new_array, array, columns):

        region_list = [
            region.serial_front_region_from(pixels=columns)
            for region in self.region_list
        ]

        array_2d_list = self.array_2d_list_from(array=array, columns=columns)

        for arr, region in zip(array_2d_list, region_list):
            new_array[region.y0 : region.y1, region.x0 : region.x1] += arr

        return new_array
