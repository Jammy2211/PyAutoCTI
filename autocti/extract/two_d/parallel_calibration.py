from copy import deepcopy
from typing import Tuple

import autoarray as aa

from autocti.charge_injection.mask_2d import Mask2DCI


class Extract2DParallelCalibration:
    def __init__(self, shape_2d: Tuple[int, int], region_list: aa.type.Region2DList):
        """
        Class containing methods for extracting a parallel calibration dataset from a 2D CTI calibration dataset.

        The parallel calibration region is the region of a dataset that is necessary for fitting a parallel-only CTI
        model. For example, for charge injection imaging, parallel EPERs form only in columns of the CCD where charge
        is injected, the serial prescan and overscan have no signal. The parallel calibration dataset therefore
        extracts only these columns.

        A subset of the parallel calibration data may also be extracted (e.g. only the first row of every charge region)
        for fast initial CTI modeling.

        This uses the `region_list`, which contains the regions with charge (e.g. the charge injection regions) in
        pixel coordinates.

        Parameters
        ----------
        shape_2d
            The two dimensional shape of the charge injection imaging, corresponding to the number of rows (pixels
            in parallel direction) and columns (pixels in serial direction).
        region_list
            Integer pixel coordinates specifying the corners of each charge region (top-row, bottom-row,
            left-column, right-column).
        """
        self.shape_2d = shape_2d
        self.region_list = list(map(aa.Region2D, region_list))

    def extraction_region_from(self, columns: Tuple[int, int]) -> aa.type.Region2DLike:
        """
        Extract a parallel calibration array from an input array, where this array contains a sub-set of the input
        array which is specifically used for only parallel CTI calibration. This array is simply a specified number
        of columns that are closest to the read-out electronics.

        The diagram below illustrates the arrays that is extracted from a array with columns=(0, 3):

        [] = read-out electronics
        [==========] = read-out register
        [..........] = serial prescan
        [pppppppppp] = parallel overscan
        [ssssssssss] = serial overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicate the region index)
        [tttttttttt] = parallel / serial charge injection region trail

               [ptptptptptptptptptptp]
               [tptptptptptptptptptpt]
          [...][xxxxxxxxxxxxxxxxxxxxx][sss]
          [...][ccccccccccccccccccccc][sss]
        | [...][ccccccccccccccccccccc][sts]    |
        | [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | Direction
        P [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | of
        | [...][ccccccccccccccccccccc][sss]    | clocking
        \/[...][ccccccccccccccccccccc][sss]   \/

        []     [=====================]
               <--------Ser---------

        The extracted array keeps just the trails following all charge injection scans and replaces all other
        values with 0s:

               [ptp]
               [tpt]
               [xxx]
               [ccc]
        |      [ccc]                           |
        |      [xxx]                           | Direction
       Par     [xxx]                           | of
        |      [ccc]                           | clocking
               [ccc]                           |

        []     [=====================]
               <--------Ser---------
        """
        return self.region_list[0].serial_towards_roe_full_region_from(
            shape_2d=self.shape_2d, pixels=columns
        )

    def array_2d_from(self, array: aa.Array2D, columns: Tuple[int, int]) -> aa.Array2D:
        """
        Extract a parallel calibration array from an input array, where this array contains a sub-set of the input
        array which is specifically used for only parallel CTI calibration. This array is simply a specified number
        of columns that are closest to the read-out electronics.

        The diagram below illustrates the arrays that is extracted from a array with columns=(0, 3):

        [] = read-out electronics
        [==========] = read-out register
        [..........] = serial prescan
        [pppppppppp] = parallel overscan
        [ssssssssss] = serial overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicate the region index)
        [tttttttttt] = parallel / serial charge injection region trail

               [ptptptptptptptptptptp]
               [tptptptptptptptptptpt]
          [...][xxxxxxxxxxxxxxxxxxxxx][sss]
          [...][ccccccccccccccccccccc][sss]
        | [...][ccccccccccccccccccccc][sts]    |
        | [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | Direction
        P [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | of
        | [...][ccccccccccccccccccccc][sss]    | clocking
        \/[...][ccccccccccccccccccccc][sss]   \/

        []     [=====================]
               <--------Ser---------

        The extracted array keeps just the trails following all charge injection scans and replaces all other
        values with 0s:

               [ptp]
               [tpt]
               [xxx]
               [ccc]
        |      [ccc]                           |
        |      [xxx]                           | Direction
        P      [xxx]                           | of
        |      [ccc]                           | clocking
        \/     [ccc]                          \/

        []     [=====================]
               <--------Ser---------
        """
        extraction_region = self.extraction_region_from(columns=columns)
        return aa.Array2D.manual_native(
            array=array.native[extraction_region.slice],
            header=array.header,
            pixel_scales=array.pixel_scales,
        )

    def extracted_layout_from(self, layout, columns: Tuple[int, int]) -> "Layout2DCI":
        """
        Extract a parallel calibration array from an input array, where this array contains a sub-set of the input
        array which is specifically used for only parallel CTI calibration. This array is simply a specified number
        of columns that are closest to the read-out electronics.

        The diagram below illustrates the arrays that is extracted from a array with columns=(0, 3):

        [] = read-out electronics
        [==========] = read-out register
        [..........] = serial prescan
        [pppppppppp] = parallel overscan
        [ssssssssss] = serial overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicate the region index)
        [tttttttttt] = parallel / serial charge injection region trail

               [ptptptptptptptptptptp]
               [tptptptptptptptptptpt]
          [...][xxxxxxxxxxxxxxxxxxxxx][sss]
          [...][ccccccccccccccccccccc][sss]
        | [...][ccccccccccccccccccccc][sts]    |
        | [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | Direction
        P [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | of
        | [...][ccccccccccccccccccccc][sss]    | clocking
        \/[...][ccccccccccccccccccccc][sss]    \/

        []     [=====================]
               <--------Ser---------

        The extracted array keeps just the trails following all charge injection scans and replaces all other
        values with 0s:

               [ptp]
               [tpt]
               [xxx]
               [ccc]
        |      [ccc]                           |
        |      [xxx]                           | Direction
        P      [xxx]                           | of
        |      [ccc]                           | clocking
        \/     [ccc]                           \/

        []     [=====================]
               <--------Ser---------
        """

        extraction_region = self.extraction_region_from(columns=columns)

        return self.with_extracted_regions(
            layout=layout, extraction_region=extraction_region
        )

    def with_extracted_regions(
        self, layout, extraction_region: aa.type.Region2DLike
    ) -> "Layout2DCI":

        layout = deepcopy(layout)

        extracted_region_list = list(
            map(
                lambda region: aa.util.layout.region_after_extraction(
                    original_region=region, extraction_region=extraction_region
                ),
                self.region_list,
            )
        )
        extracted_region_list = list(filter(None, extracted_region_list))
        if not extracted_region_list:
            extracted_region_list = None

        layout.region_list = extracted_region_list
        return layout

    def mask_2d_from(self, mask: aa.Mask2D, columns: Tuple[int, int]) -> "Mask2DCI":
        """
        Extract a mask to go with a parallel calibration array from an input mask.

        The parallel calibration array is described in the function `array_2d_from()`.
        """
        extraction_region = self.region_list[0].serial_towards_roe_full_region_from(
            shape_2d=self.shape_2d, pixels=columns
        )
        return Mask2DCI(
            mask=mask[extraction_region.slice], pixel_scales=mask.pixel_scales
        )
