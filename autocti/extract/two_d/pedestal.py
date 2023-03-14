from typing import List, Optional, Tuple

import autoarray as aa

from autocti.extract.two_d.parallel.abstract import Extract2DParallel

from autocti.extract.two_d import extract_2d_util


class Extract2DParallelPedestal(Extract2DParallel):
    def region_list_from(
        self,
        pixels: Optional[Tuple[int, int]] = None,
        pixels_from_end: Optional[int] = None,
    ) -> List[aa.Region2D]:
        """
        Returns a list containing the 2D pedestral region, which is in the corner of the CCD and extracted using
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
        [pepepepepe[ = pedestal
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
         pixels
             The row pixel index which determines the region of the overscan (e.g. `pixels=(0, 3)` will compute the
             region corresponding to the 1st, 2nd and 3rd overscan rows).
         pixels_from_end
             Alternative row pixex index specification, which extracts this number of pixels from the end of
             the overscan. For example, if the overscan is 100 pixels and `pixels_from_end=10`, the
             last 10 pixels of the overscan (pixels (90, 100)) are extracted.
        """

        if pixels_from_end is not None:
            pixels = (
                self.parallel_overscan.total_rows - pixels_from_end,
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

    def binned_region_1d_from(self, pixels: Tuple[int, int]) -> aa.Region1D:
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
        pixels
            The row pixel index to extract the EPERs between (e.g. `pixels=(0, 3)` extracts the 1st, 2nd and 3rd EPER
            rows)
        """
        return extract_2d_util.binned_region_1d_eper_from(pixels=pixels)
