from typing import List

import autoarray as aa

from autocti.extract.two_d.parallel.abstract import Extract2DParallel
from autocti.extract.settings import SettingsExtract

from autocti.extract.two_d import extract_2d_util


class Extract2DParallelOverscan(Extract2DParallel):
    def region_list_from(self, settings: SettingsExtract) -> List[aa.Region2D]:
        """
        Returns a list of the 2D parallel overscan region, which is the parallel overscan input to the
        object, between two input `pixels` indexes (this is somewhat redundant information, but mimicks
        the `Extract` object API across all other `Extract` objects).

        (top-row, bottom-row, left-column, right-column) = (y0, y1, x0, x1)

        The parallel overscan spans all columns of the image, thus the coordinates x0 and x1 do not change. y0 and y1
        are updated based on the `pixels` input.

        Negative pixel values can be input into the `pixels` tuple, whereby rows in front of the parallel overscan are
        also extracted.

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
        |  [...][1c1c1cc1c1cc1ccc1cc1][sss]    |
        |  [...][ttttttttttttttttttttt][sss]    | Direction
        Par [...][ttttttttttttttttttttt][sss]    | of
        |  [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
        |/  [...][cc0ccc0cccc0cccc0cccc][sss]    \/

        []     [=====================]
               <---------Ser--------

        The extracted regions correspond to the parallel overscan [ppppppppppppppp] regions.

        For `pixels=(0,1)` the extracted arrays returned via the `array_2d_list_from()` function keep the first
        parallel pixels across the rows of the overscan:

        array_2d_list[0] = [ppppppppppppppppppppp]

        Parameters
        ----------
        settings
           The settings used to extract the parallel overscan, which for example include the `pixels` tuple specifying the
           range of pixel rows they are extracted between.
        """

        pixels = settings.pixels

        if settings.pixels_from_end is not None:
            pixels = (
                self.parallel_overscan.total_rows - settings.pixels_from_end,
                self.parallel_overscan.total_rows,
            )

        parallel_overscan_extract = aa.Region2D(
            (
                self.parallel_overscan.y0 + pixels[0],
                self.parallel_overscan.y0 + pixels[1],
                self.parallel_overscan.x0,
                self.parallel_overscan.x1,
            )
        )

        return [parallel_overscan_extract]

    def binned_region_1d_from(self, settings: SettingsExtract) -> aa.Region1D:
        """
        The `Extract` objects allow one to extract a `Dataset1D` from a 2D CTI dataset, which is used to perform
        CTI modeling in 1D.

        This is performed by binning up the data via the `binned_array_1d_from` function.

        In order to create the 1D dataset a `Layout1D` is required, which requires the `region_list` containing the
        charge regions on the 1D dataset (e.g. where the FPR appears in 1D after binning).

        The function returns the this region if the 1D dataset is extracted from the parallel overscan. This
        assumes the overscan contains EPER trails and therefore all pixels in front of the overscan act as the
        charge region and therefore FPR. This is the case when science imaging flat field data is used.

        The charge region is therefore only included if there are negative entries in the `pixels` tuple, meaning that
        pixels before the overscan (e.g. the FPR) are extracted.

        Parameters
        ----------
        settings
           The settings used to extract the parallel overscan, which for example include the `pixels` tuple specifying the
           range of pixel rows they are extracted between.
        """
        return extract_2d_util.binned_region_1d_eper_from(pixels=settings.pixels)
