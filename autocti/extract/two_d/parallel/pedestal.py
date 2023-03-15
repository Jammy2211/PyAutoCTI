from typing import List

import autoarray as aa

from autocti.extract.two_d.parallel.abstract import Extract2DParallel
from autocti.extract.settings import SettingsExtract


class Extract2DParallelPedestal(Extract2DParallel):
    def region_list_from(self, settings: SettingsExtract) -> List[aa.Region2D]:
        """
        Returns a list of the 2D pedestral region, which is in the corner of the CCD and extracted using
        the coordinates of the parallel and serial overscan regions.

        (top-row, bottom-row, left-column, right-column) = (y0, y1, x0, x1)

        The parallel pedestal spans all rows of the parallel overscan and columns of the serial overscan. Unlike
        the parallel and serial overscans, the pedestal is not directional. Separatel parallel and serial pedestal
        extractor objects are used to extract the pedestal in theses directions.

        Negative pixel values can be input into the `pixels` tuple, whereby rows in front of the pedestal are
        extracted.

        The diagram below illustrates the extraction for `pixels=(0, 1)`:

        [] = read-out electronics
        [==========] = read-out register
        [..........] = serial prescan
        [pppppppppp] = parallel overscan
        [ssssssssss] = serial overscan
        [pepepepepe] = pedestal
        [f#ff#f#f#f] = signal region (FPR) (0 / 1 indicate the region index)
        [tttttttttt] = parallel / serial charge injection region trail

               [ppppppppppppppppppppp][pepe]
               [ppppppppppppppppppppp][pepe]
           [...][ttttttttttttttttttttt][sss]
           [...][c1c1cc1c1cc1cc1ccc1cc][sss]
        |  [...][1c1c1cc1c1cc1ccc1cc1][sss]    |
        |  [...][ttttttttttttttttttttt][sss]    | Direction
        Par [...][ttttttttttttttttttttt][sss]    | of
        |  [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
        \/  [...][cc0ccc0cccc0cccc0cccc][sss]    \/

         []     [=====================]
                <---------Ser--------

         The extracted regions correspond to the parallel overscan [pepe] regions.

         For `pixels=(0,1)` the extracted arrays returned via the `array_2d_list_from()` function keep the first
         pedestal pixels across the rows of the overscan:

         array_2d_list[0] = [pepe]

         Parameters
         ----------
        settings
           The settings used to extract the parallel pedestal, which for example include the `pixels` tuple specifying the
           range of pixel rows they are extracted between.
        """
        pixels = settings.pixels

        if settings.pixels_from_end is not None:
            pixels = (
                self.pedestal.total_rows - settings.pixels_from_end,
                self.pedestal.total_rows,
            )

        pedestal_extract = aa.Region2D(
            (
                self.pedestal.y0 + pixels[0],
                self.pedestal.y0 + pixels[1],
                self.pedestal.x0,
                self.pedestal.x1,
            )
        )

        return [pedestal_extract]

    def array_2d_from(self, array: aa.Array2D) -> aa.Array2D:
        """
        Extract from an `Array2D` the pedestal and return it as a new `Array2D` where everything else (e.g.
        the charge injection regions, serial EPERs) are zeros.

        The dimensions of the input array therefore do not change (unlike other ``Extract`` methods).

        Negative pixel values can be input into the `pixels` tuple, whereby rows in front of the parallel pedestal (e.g.
        the serial EPERs) are extracted.

        The diagram below illustrates the extraction:

        [] = read-out electronics
        [==========] = read-out register
        [..........] = serial prescan
        [pppppppppp] = parallel overscan
        [ssssssssss] = serial overscan
        [pepepepepe] = pedestal
        [f#ff#f#f#f] = signal region (FPR) (0 / 1 indicate the region index)
        [tttttttttt] = parallel / serial charge injection region trail

             [tptpptptptpptpptpptpt][pepe]
             [tptptptpptpttptptptpt][pepe]
        [...][ttttttttttttttttttttt][sss]
        [...][ccccccccccccccccccccc][sss]
        | [...][ccccccccccccccccccccc][sss]    |
        | [...][ttttttttttttttttttttt][sss]    | Direction
        Par[...][ttttttttttttttttttttt][sss]    | of
        | [...][ccccccccccccccccccccc][sss]    | clocking
        |/ [...][ccccccccccccccccccccc][sss]   \/

        []     [=====================]
             <--------Ser---------

        The extracted array keeps only the pedestal, everything else become 0s:

              [0000000000000000000000][pepe]
              [0000000000000000000000][pepe]
          [000][000000000000000000000][000]
          [000][000000000000000000000][000]
        | [000][000000000000000000000][000]    |
        | [000][000000000000000000000][000]    | Direction
        Par[000][00000000000000000000][000]    | of
        | [000][000000000000000000000][000]    | clocking
        |/ [000][000000000000000000000][000]   \/

        []     [=====================]
             <--------Ser---------
        """

        array_2d = array.native.copy() * 0.0

        return self.add_to_array(
            new_array=array_2d,
            array=array,
            settings=SettingsExtract(pixels=(0, self.pedestal.total_rows)),
        )
