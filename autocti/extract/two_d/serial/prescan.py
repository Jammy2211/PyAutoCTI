from typing import List, Tuple

import autoarray as aa

from autocti.extract.two_d.serial.abstract import Extract2DSerial


class Extract2DSerialPrescan(Extract2DSerial):
    def region_list_from(self, pixels: Tuple[int, int]) -> List[aa.Region2D]:
        """
         Returns a list containing the 2D serial prescan region, which is simply the parallel overscan input to the
         object, extracted between two input `pixels` indexes (this is somewhat redundant information, but mimicks
         the `Extract` object API across all other `Extract` objects).

         (top-row, bottom-row, left-column, right-column) = (y0, y1, x0, x1)

         The serial prescan spans all columns of the image, thus the coordinates x0 and x1 do not change. y0 and y1
         are updated based on the `pixels` input.

         Negative pixel values are supported to the `pixels` tuple, whereby columns in front of the serial prescan are
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
         pixels
             The column pixel index which determines the region of the prescan (e.g. `pixels=(0, 3)` will compute the
             region corresponding to the 1st, 2nd and 3rd prescan columns).
        """
        serial_prescan_extract = aa.Region2D(
            (
                self.serial_prescan.y0,
                self.serial_prescan.y1,
                self.serial_prescan.x0 + pixels[0],
                self.serial_prescan.x0 + pixels[1],
            )
        )

        return [serial_prescan_extract]
