from typing import List, Optional, Tuple

import autoarray as aa

from autocti.extract.two_d.parallel.eper import Extract2DParallelEPER
from autocti.extract.two_d.serial.abstract import Extract2DSerial
from autocti.extract.settings import SettingsExtract


class Extract2DSerialOverscanNoEPER(Extract2DSerial):
    def region_list_from(self, settings: SettingsExtract) -> List[aa.Region2D]:
        """
        Returns a list of the 2D serial overscan regions without EPER trails, between two input `pixels` indexes.

        (top-row, bottom-row, left-column, right-column) = (y0, y1, x0, x1)

        The serial overscan spans all columns of the image, thus the coordinates x0 and x1 do not change. y0 and y1
        are updated based on where serial EPER trails are not and the `pixels` input.

        Negative pixel values can be input into the `pixels` tuple, whereby columns in front of the serial overscan are
        also extracted.

        The diagram below illustrates the extraction for `pixels=(0, 1)`:

        [] = read-out electronics
        [==========] = read-out register
        [..........] = serial prescan
        [pppppppppp] = serial overscan
        [ssssssssss] = serial overscan
        [snsnsnsnsn] = serial overscan with no serial EPERs
        [f#ff#f#f#f] = signal region (FPR) (0 / 1 indicate the region index)
        [tttttttttt] = serial / serial charge injection region trail

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
           [...][ttttttttttttttttttttt][sns]
           [...][c1c1cc1c1cc1cc1ccc1cc][sns]
        |  [...][1c1c1cc1c1cc1ccc1cc1][sss]    |
        |  [...][ttttttttttttttttttttt][sns]    | Direction
        Par [...][ttttttttttttttttttttt][sns]    | of
        |  [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
        |/  [...][cc0ccc0cccc0cccc0cccc][sss]    \/

        []     [=====================]
               <---------Ser--------

        The extracted regions correspond to the serial overscan with no serial EPERs [sns] regions.

        For `pixels=(0,1)` the extracted arrays returned via the `array_2d_list_from()` function keep the first
        serial pixels across the columns of the overscan:

        array_2d_list[0] = [sns]

        Parameters
        ----------
        settings
            The settings used to extract the serial overscan region without EPERs from, which for example include
            the `pixels`  tuple specifying the range of pixel columns they are extracted between.
        """

        pixels = settings.pixels

        if settings.pixels_from_end is not None:
            pixels = (
                self.serial_overscan.total_columns - settings.pixels_from_end,
                self.serial_overscan.total_columns,
            )

        extract_parallel_eper = Extract2DParallelEPER(
            shape_2d=self.shape_2d,
            region_list=self.region_list,
            parallel_overscan=self.parallel_overscan,
            serial_overscan=self.serial_overscan,
            serial_prescan=self.serial_prescan,
        )

        parallel_eper_region_list = extract_parallel_eper.region_list_from(
            settings=SettingsExtract(pixels_from_end=-1)
        )

        region_list = []

        for i, region in enumerate(self.region_list):

            region_list.append(
                aa.Region2D(
                    (
                        parallel_eper_region_list[i].y0,
                        parallel_eper_region_list[i].y1,
                        self.serial_overscan.x0 + pixels[0],
                        self.serial_overscan.x0 + pixels[1],
                    )
                )
            )

        return region_list
