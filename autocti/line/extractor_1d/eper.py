import numpy as np
from typing import List, Tuple

import autoarray as aa

from autocti.line.extractor_1d.abstract import Extractor1D


class Extractor1DEPER(Extractor1D):
    def array_1d_list_from(
        self, array: aa.Array1D, pixels: Tuple[int, int]
    ) -> List[aa.Array1D]:
        """
        Extract the parallel trails of a charge injection array.


        The diagram below illustrates the arrays that is extracted from a array for pixels=(0, 1):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCDPhase panel    [pppppppppp] = parallel overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicates ci region index)
        [xxxxxxxxxx]
        [t#t#t#t#t#] = parallel / serial charge injection region trail (0 / 1 indicates ci region index)

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][t1t1t1t1t1t1t1t1t1t1t][sss]
          [...][c1c1cc1c1cc1cc1ccc1cc][sss]
        | [...][1c1c1cc1c1cc1ccc1cc1][sss]    |
        | [...][t0t0t0t0t0t0t0t0t0t0t][sss]    | Direction
        P [...][0t0t0t0t0t0t0t0t0t0t0][sss]    | of
        | [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
          [...][cc0ccc0cccc0cccc0cccc][sss]    |

        []     [=====================]
               <---------S----------

        The extracted array keeps just the trails following all charge injection scans:

        list index 0:

        [t0t0t0tt0t0t0t0t0t0t0]

        list index 1:

        [1t1t1t1t1t1t1t1t1t1t1]

        Parameters
        ------------
        array
        pixels
            The row indexes to extract the trails between (e.g. pixels(0, 3) extracts the 1st, 2nd and 3rd pixels)
        """

        trails_region_list = self.region_list_from(pixels=pixels)
        trails_arrays = list(
            map(lambda region: array.native[region.slice], trails_region_list)
        )
        trails_masks = list(
            map(lambda region: array.mask[region.slice], trails_region_list)
        )
        trails_arrays = list(
            map(
                lambda trails_array, front_mask: np.ma.array(
                    trails_array, mask=front_mask
                ),
                trails_arrays,
                trails_masks,
            )
        )
        return trails_arrays

    def stacked_array_1d_from(
        self, array: aa.Array1D, pixels: Tuple[int, int]
    ) -> aa.Array1D:
        trails_arrays = self.array_1d_list_from(array=array, pixels=pixels)
        return np.ma.mean(np.ma.asarray(trails_arrays), axis=0)

    def region_list_from(self, pixels: Tuple[int, int]) -> List[aa.Region1D]:
        """
        Returns the parallel scans of a charge injection array.

            The diagram below illustrates the region that is calculated from a array for pixels=(0, 1):

            ---KEY---
            ---------

            [] = read-out electronics   [==========] = read-out register

            [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
            [xxxxxxxxxx] = CCDPhase panel    [pppppppppp] = parallel overscan
            [c#cc#c#c#c] = charge injection region (0 / 1 indicates ci region index)
            [xxxxxxxxxx]
            [t#t#t#t#t#] = parallel / serial charge injection region trail (0 / 1 indicates ci region index)

            P = Parallel Direction      S = Serial Direction

                   [ppppppppppppppppppppp]
                   [ppppppppppppppppppppp]
              [...][t1t1t1t1t1t1t1t1t1t1t][sss]
              [...][c1c1cc1c1cc1cc1ccc1cc][sss]
            | [...][1c1c1cc1c1cc1ccc1cc1][sss]    |
            | [...][t0t0t0t0t0t0t0t0t0t0t][sss]    | Direction
            P [...][0t0t0t0t0t0t0t0t0t0t0][sss]    | of
            | [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
              [...][cc0ccc0cccc0cccc0cccc][sss]    |

            []     [=====================]
                   <---------S----------

            The extracted array keeps just the trails following all charge injection scans:

            list index 0:

            [2, 4, 3, 21] (serial prescan is 3 pixels)

            list index 1:

            [6, 7, 3, 21] (serial prescan is 3 pixels)

            Parameters
            ------------
            arrays
            pixels
                The row indexes to extract the trails between (e.g. pixels(0, 3) extracts the 1st, 2nd and 3rd pixels)
        """

        return list(
            map(
                lambda region: region.trailing_region_from(pixels=pixels),
                self.region_list,
            )
        )

    def add_to_array(
        self, new_array: aa.Array1D, array: aa.Array1D, pixels: Tuple[int, int]
    ):

        region_list = [
            region.trailing_region_from(pixels=pixels) for region in self.region_list
        ]

        array_1d_list = self.array_1d_list_from(array=array, pixels=pixels)

        for arr, region in zip(array_1d_list, region_list):
            new_array[region.x0 : region.x1] += arr

        return new_array
