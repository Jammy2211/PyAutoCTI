from typing import List, Optional, Tuple

import autoarray as aa

from autocti.extract.two_d.serial.abstract import Extract2DSerial
from autocti.extract.settings import SettingsExtract


class Extract2DSerialPrescan(Extract2DSerial):
    def region_list_from(self, settings: SettingsExtract) -> List[aa.Region2D]:
        """
        Returns a list of the 2D serial prescan region, which is the parallel overscan input to the
        object, between two input `pixels` indexes (this is somewhat redundant information, but mimicks
        the `Extract` object API across all other `Extract` objects).

        (top-row, bottom-row, left-column, right-column) = (y0, y1, x0, x1)

        The serial prescan spans all columns of the image, thus the coordinates x0 and x1 do not change. y0 and y1
        are updated based on the `pixels` input.

        Negative pixel values can be input into the `pixels` tuple, whereby columns in front of the serial prescan are
        also extracted.

        The diagram below illustrates the extraction for `pixels=(0, 1)`:

        [] = read-out electronics
        [==========] = read-out register
        [..........] = serial prescan
        [pppppppppp] = parallel overscan
        [ssssssssss] = serial overscan
        [f#ff#f#f#f] = signal region (FPR) (0 / 1 indicate the region index)
        [tttttttttt] = serial / serial charge injection region trail

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
           [...][ttttttttttttttttttttt][sss]
           [...][f1f1ff1f1ff1ff1fff1ff][sss]
        |  [...][1f1f1ff1f1ff1fff1ff1][sss]    |
        |  [...][ttttttttttttttttttttt][sss]    | Direction
        Par [...][ttttttttttttttttttttt][sss]    | of
        |  [...][0fff0ffff0ffff0ffff0f][sss]    | clocking
        |/  [...][ff0fff0ffff0ffff0ffff][sss]    \/

        []     [=====================]
               <---------Ser--------

        The extracted regions correspond to the serial prescan [...] regions.

        For `pixels=(0,1)` the extracted arrays returned via the `array_2d_list_from()` function keep the first
        serial pixels across the columns of the prescan:

        array_2d_list[0] = [s]

        Parameters
        ----------
        settings
           The settings used to extract the serial region overscan from, which for example include the `pixels`
           tuple specifying the range of pixel columns they are extracted between.
        """

        pixels = settings.pixels

        if settings.pixels_from_end is not None:
            pixels = (
                self.serial_prescan.total_columns - settings.pixels_from_end,
                self.serial_prescan.total_columns,
            )

        serial_prescan_extract = aa.Region2D(
            (
                self.serial_prescan.y0,
                self.serial_prescan.y1,
                self.serial_prescan.x0 + pixels[0],
                self.serial_prescan.x0 + pixels[1],
            )
        )

        return [serial_prescan_extract]
