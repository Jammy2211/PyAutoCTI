from typing import Optional, Tuple

import autoarray as aa

from autocti.extract.two_d.serial.abstract import Extract2DSerial
from autocti.extract.settings import SettingsExtract

from autocti.extract.two_d import extract_2d_util


class Extract2DSerialFPR(Extract2DSerial):
    def region_list_from(self, settings: SettingsExtract):
        """
        Returns a list of the 2D serial FPR regions from the `region_list` containing signal  (e.g. the charge
        injection regions of charge injection data), between two input `pixels` indexes.

        Negative pixel values can be input into the `pixels` tuple, whereby columns in front of the serial FPRs (e.g.
        the serial prescan) are extracted.

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
        |  [...][1c1c1cc1c1cc1ccc1cc1c][sss]    |
        |  [...][ttttttttttttttttttttt][sss]    | Direction
        Par [...][ttttttttttttttttttttt][sss]    | of
        |  [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
        |/  [...][cc0ccc0cccc0cccc0cccc][sss]    \/

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
        settings
            The settings used to extract the serial region FPRs from, which for example include the `pixels`
            tuple specifying the range of pixel columns they are extracted between.
        """
        return [
            region.serial_front_region_from(
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

        The function returns the this region if the 1D dataset is extracted from the serial FPRs. This is
        the full range of the `pixels` tuple, unless negative entries are included, meaning that pixels
        before the FPRs are extracted.

        Parameters
        ----------
        settings
            The settings used to extract the serial region FPRs from, which for example include the `pixels`
            tuple specifying the range of pixel columns they are extracted between.
        """
        return extract_2d_util.binned_region_1d_fpr_from(pixels=settings.pixels)
