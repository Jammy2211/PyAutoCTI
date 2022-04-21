from typing import Tuple

import autoarray as aa

from autocti.extract.two_d.abstract import Extract2D


class Extract2DSerialFPR(Extract2D):
    @property
    def binning_axis(self) -> int:
        """
        The axis over which binning is performed to turn a 2D serial FPR into a 1D FPR.

        For a serial extract `axis=0` such that binning is performed over the columns containing the FPR.
        """
        return 0

    def region_list_from(self, pixels: Tuple[int, int]):
        """
        Extract the serial FPR of every charge injection region on the charge injection image and return as a list
        of 2D arrays.

        Negative pixel values are supported to the `pixels` tuple, whereby columns in front of the serial FPRs (e.g.
        the serial prescan) are also extracted.

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
        ----------
        pixels
            The column indexes to extract the front edge between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd
            columns)
        """
        return [
            region.serial_front_region_from(pixels=pixels)
            for region in self.region_list
        ]

    def binned_region_1d_from(self, pixels: Tuple[int, int]) -> aa.Region1D:
        """
        The `Extract` objects allow one to extract a `Dataset1D` from a 2D CTI dataset, which is used to perform
        CTI modeling in 1D.

        This is performed by binning up the data via the `binned_array_1d_from` function.

        In order to create the 1D dataset a `Layout1D` is required, which requires the `region_list` containing the
        charge regions on the 1D dataset (e.g. where the FPR appears in 1D after binning).

        The function returns the this region list if the 1D dataset is extracted from the serial FPRs. This is
        the full range of the `pixels` tuple, unless negative entries are included, meaning that pixels
        before the FPRs are also extracted.

        Parameters
        ----------
        pixels
            The column pixel index to extract the FPRs between (e.g. `pixels=(0, 3)` extracts the 1st, 2nd and 3rd FPR
            columns)
        """
        if pixels[0] <= 0 and pixels[1] <= 0:
            return None
        elif pixels[0] >= 0:
            return aa.Region1D(region=(0, pixels[1]))
        return aa.Region1D(region=(abs(pixels[0]), pixels[1] + abs(pixels[0])))
