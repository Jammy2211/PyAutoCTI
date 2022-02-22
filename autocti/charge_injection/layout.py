from copy import deepcopy
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import autoarray as aa

from autocti import exc
from autocti.charge_injection.extractor_2d.parallel_fpr import Extractor2DParallelFPR
from autocti.charge_injection.extractor_2d.parallel_eper import Extractor2DParallelEPER
from autocti.charge_injection.extractor_2d.serial_fpr import Extractor2DSerialFPR
from autocti.charge_injection.extractor_2d.serial_eper import Extractor2DSerialEPER
from autocti.charge_injection.mask_2d import Mask2DCI


class Layout2DCI(aa.Layout2D):
    def __init__(
        self,
        shape_2d: Tuple[int, int],
        region_list: aa.type.Region2DList,
        original_roe_corner: Tuple[int, int] = (1, 0),
        parallel_overscan: Optional[aa.type.Region2DLike] = None,
        serial_prescan: Optional[aa.type.Region2DLike] = None,
        serial_overscan: Optional[aa.type.Region2DLike] = None,
        electronics: Optional["ElectronicsCI"] = None,
    ):
        """
        A charge injection layout, which defines the regions charge injections appear on a charge injection image.

        It also contains over regions of the image, for example the serial prescan, overscan and paralle overscan.

        Parameters
        -----------
        shape_2d
            The two dimensional shape of the charge injection imaging, corresponding to the number of rows (pixels
            in parallel direction) and columns (pixels in serial direction).
        region_list
            Integer pixel coordinates specifying the corners of each charge injection region (top-row, bottom-row,
            left-column, right-column).
        original_roe_corner
            The original read-out electronics corner of the charge injeciton imaging, which is internally rotated to a
            common orientation in **PyAutoCTI**.
        parallel_overscan
            Integer pixel coordinates specifying the corners of the parallel overscan (top-row, bottom-row,
            left-column, right-column).
        serial_prescan
            Integer pixel coordinates specifying the corners of the serial prescan (top-row, bottom-row,
            left-column, right-column).
        serial_overscan
            Integer pixel coordinates specifying the corners of the serial overscan (top-row, bottom-row,
            left-column, right-column).
        electronics
            The charge injection electronics parameters of the image (e.g. the IG1 and IG2 voltages).
        """

        self.region_list = list(map(aa.Region2D, region_list))

        for region in self.region_list:

            if region.y1 > shape_2d[0] or region.x1 > shape_2d[1]:
                raise exc.LayoutException(
                    "The charge injection layout_ci regions are bigger than the image image_shape"
                )

        self.extractor_parallel_fpr = Extractor2DParallelFPR(region_list=region_list)
        self.extractor_parallel_eper = Extractor2DParallelEPER(region_list=region_list)
        self.extractor_serial_fpr = Extractor2DSerialFPR(region_list=region_list)
        self.extractor_serial_eper = Extractor2DSerialEPER(region_list=region_list)

        self.electronics = electronics

        super().__init__(
            shape_2d=shape_2d,
            original_roe_corner=original_roe_corner,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

    def layout_extracted_from(
        self, extraction_region: aa.type.Region2DLike
    ) -> "Layout2DCI":
        """
        The charge injection layout after an extraction is performed on its associated charge injection image, where
        the extraction is defined by a region of pixel coordinates 
        
        (top-row, bottom-row, left-column, right-column) = (y0, y1, x0, x1)

        For example, if a charge injection region occupies the pixels (0, 10, 0, 5) on a 100 x 100 charge injection
        image, and the first 5 columns of this charge injection image are extracted to create a 100 x 5 image, the
        new charge injection region will be (0, 20, 0, 5).

        Parameters
        ----------
        extraction_region
            The (y0, y1, x0, x1) pixel coordinates defining the region which the layout is extracted from.
        """

        layout = super().layout_extracted_from(extraction_region=extraction_region)

        region_list = [
            aa.util.layout.region_after_extraction(
                original_region=region, extraction_region=extraction_region
            )
            for region in self.region_list
        ]

        return self.__class__(
            original_roe_corner=self.original_roe_corner,
            shape_2d=self.shape_2d,
            region_list=region_list,
            parallel_overscan=layout.parallel_overscan,
            serial_prescan=layout.serial_prescan,
            serial_overscan=layout.serial_overscan,
        )

    @property
    def pixels_between_regions(self) -> List[int]:
        """
        Returns a list where each entry is the number of pixels a charge injection region and its neighboring
        charge injection region.
        """
        return [
            self.region_list[i + 1].y0 - self.region_list[i].y1
            for i in range(len(self.region_list) - 1)
        ]

    @property
    def parallel_rows_to_array_edge(self) -> int:
        """
        The number of pixels from the edge of the parallel EPERs to the edge of the array.

        This is the number of pixels from the last charge injection FPR edge to the read out register and electronics
        and will include the parallel overscan if the CCD has one.
        """
        return self.shape_2d[0] - np.max([region.y1 for region in self.region_list])

    @property
    def smallest_parallel_rows_between_ci_regions(self) -> int:
        """
        The smallest number of pixels between any two charge injection regions, or the distance of the last
        charge injection region to the edge of the charge injeciton image (e.g. in the direction away from the
        readout register and electronics).
        """
        pixels_between_regions = self.pixels_between_regions
        pixels_between_regions.append(self.parallel_rows_to_array_edge)
        return np.min(pixels_between_regions)

    def regions_array_2d_from(self, array: aa.Array2D) -> aa.Array2D:
        """
        Extract all of the charge-injection regions from an input `Array2D` object and returns them as a new `Array2D`
        where these extracted regions are included and all other entries are zeros.

        The dimensions of the input array therefore do not change (unlike other `Layout2DCI` methods).

        The diagram below illustrates the extraction:

        [] = read-out electronics
        [==========] = read-out register
        [..........] = serial prescan
        [pppppppppp] = parallel overscan
        [ssssssssss] = serial overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicate the region index)
        [tttttttttt] = parallel / serial charge injection region trail

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][xxxxxxxxxxxxxxxxxxxxx][sss]
          [...][ccccccccccccccccccccc][sss]
        | [...][ccccccccccccccccccccc][sss]    |
        | [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | Direction
      Par [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | of
        | [...][ccccccccccccccccccccc][sss]    | clocking
       \/  [...][ccccccccccccccccccccc][sss]   \/

        []     [=====================]
               <--------Ser---------

        The extracted array keeps just the charge injection region, all other values become 0:

               [000000000000000000000]
               [000000000000000000000]
          [000][000000000000000000000][000]
          [000][ccccccccccccccccccccc][000]
        | [000][ccccccccccccccccccccc][000]    |
        | [000][000000000000000000000][000]    | Direction
       Par[000][000000000000000000000][000]    | of
        | [000][ccccccccccccccccccccc][000]    | clocking
       \/ [000][ccccccccccccccccccccc][000]   \/

        []     [=====================]
               <--------Ser---------
        """

        new_array = array.native.copy() * 0.0

        for region in self.region_list:
            new_array[region.slice] += array.native[region.slice]

        return new_array

    def non_regions_array_2d_from(self, array: aa.Array2D) -> aa.Array2D:
        """
        Extract all of the areas of an `Array2D` that are not within any of the layout's charge-injection regions
        and return them as a new `Array2D` where these extracted regions are included and the charge injection regions
        are zeros

        The extracted array therefore includes all EPER trails and other regions of the image which may contain
        signal but are not in the FPR.

        The dimensions of the input array therefore do not change (unlike other `Layout2DCI` methods).

        The diagram below illustrates the extraction:

        [] = read-out electronics
        [==========] = read-out register
        [..........] = serial prescan
        [pppppppppp] = parallel overscan
        [ssssssssss] = serial overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicate the region index)
        [tttttttttt] = parallel / serial charge injection region trail

               [tptpptptptpptpptpptpt]
               [tptptptpptpttptptptpt]
          [...][ttttttttttttttttttttt][sss]
          [...][ccccccccccccccccccccc][sss]
        | [...][ccccccccccccccccccccc][sss]    |
        | [...][ttttttttttttttttttttt][sss]    | Direction
       Par[...][ttttttttttttttttttttt][sss]    | of
        | [...][ccccccccccccccccccccc][sss]    | clocking
        \/ [...][ccccccccccccccccccccc][sss]   \/

        []     [=====================]
               <--------Ser---------

        The extracted array keeps everything except the charge injection  region,which become 0s:

               [tptpptptptpptpptpptpt]
               [tptptptpptpttptptptpt]
          [000][ttttttttttttttttttttt][000]
          [000][000000000000000000000][000]
        | [000][000000000000000000000][000]    |
        | [000][ttttttttttttttttttttt][000]    | Direction
       Par[000][ttttttttttttttttttttt][000]    | of
        | [000][000000000000000000000][000]    | clocking
          [000][000000000000000000000][000]   \/

        []     [=====================]
               <--------Ser---------
        """

        non_regions_ci_array = array.native.copy()

        for region in self.region_list:
            non_regions_ci_array[region.slice] = 0.0

        return non_regions_ci_array

    def parallel_epers_array_2d_from(self, array: aa.Array2D) -> aa.Array2D:
        """
        Extract all of the areas of an `Array2D` that contain the parallel EPERs and return them as a new `Array2D`
        where these extracted regions are included and everything else (e.g. the charge injection regions, serial
        EPERs) are zeros.

        The dimensions of the input array therefore do not change (unlike other `Layout2DCI` methods).

        The diagram below illustrates the extraction:

        [] = read-out electronics
        [==========] = read-out register
        [..........] = serial prescan
        [pppppppppp] = parallel overscan
        [ssssssssss] = serial overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicate the region index)
        [tttttttttt] = parallel / serial charge injection region trail

               [tptpptptptpptpptpptpt]
               [tptptptpptpttptptptpt]
          [...][ttttttttttttttttttttt][sss]
          [...][ccccccccccccccccccccc][sss]
        | [...][ccccccccccccccccccccc][sss]    |
        | [...][ttttttttttttttttttttt][sss]    | Direction
       Par[...][ttttttttttttttttttttt][sss]    | of
        | [...][ccccccccccccccccccccc][sss]    | clocking
       \/ [...][ccccccccccccccccccccc][sss]   \/

        []     [=====================]
               <--------Ser---------

        The extracted array keeps only the parallel EPERs, everything else become 0s:

               [tptpptptptpptpptpptpt]
               [tptptptpptpttptptptpt]
          [000][ttttttttttttttttttttt][000]
          [000][000000000000000000000][000]
        | [000][000000000000000000000][000]    |
        | [000][ttttttttttttttttttttt][000]    | Direction
       Par[000][ttttttttttttttttttttt][000]    | of
        | [000][000000000000000000000][000]    | clocking
       \/ [000][000000000000000000000][000]   \/

        []     [=====================]
               <--------Ser---------
        """

        parallel_array = self.non_regions_array_2d_from(array=array)

        parallel_array.native[self.serial_prescan.slice] = 0.0
        parallel_array.native[self.serial_overscan.slice] = 0.0

        return parallel_array

    def parallel_fprs_and_epers_array_2d_from(
        self,
        array: aa.Array2D,
        fpr_pixels: Tuple[int, int] = None,
        trails_pixels: Tuple[int, int] = None,
    ) -> aa.Array2D:
        """
        Extract all of the data values in an input `array2D` corresponding to the parallel front edges and trails of
        each the charge-injection region.

        One can specify the range of rows that are extracted, for example:

        fpr_pixels = (0, 1) will extract just the first leading front edge row.
        fpr_pixels = (0, 2) will extract the leading two front edge rows.
        trails_pixels = (0, 1) will extract the first row of trails closest to the charge injection region.

        The diagram below illustrates the arrays that are extracted from the input array for `fpr_pixels=(0,1)`
        and `trails_pixels=(0,1)`:

        The diagram below illustrates the extraction:

        [] = read-out electronics
        [==========] = read-out register
        [..........] = serial prescan
        [pppppppppp] = parallel overscan
        [ssssssssss] = serial overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicate the region index)
        [tttttttttt] = parallel / serial charge injection region trail

               [tptpptptptpptpptpptpt]
               [tptptptpptpttptptptpt]
          [...][ttttttttttttttttttttt][sss]
          [...][ccccccccccccccccccccc][sss]
        | [...][ccccccccccccccccccccc][sss]    |
        | [...][ttttttttttttttttttttt][sss]    | Direction
        P [...][ttttttttttttttttttttt][sss]    | of
        | [...][ccccccccccccccccccccc][sss]    | clocking
          [...][ccccccccccccccccccccc][sss]    |

        []     [=====================]
               <--------Ser---------

        The extracted array keeps just the leading edges and trails following all charge injection scans and
        replaces all other values with 0s:

               [000000000000000000000]
               [000000000000000000000]
          [000][ttttttttttttttttttttt][000]
          [000][000000000000000000000][000]
        | [000][ccccccccccccccccccccc][000]    |
        | [000][000000000000000000000][000]    | Direction
        P [000][ttttttttttttttttttttt][000]    | of
        | [000][000000000000000000000][000]    | clocking
          [000][ccccccccccccccccccccc][000]    |

        []     [=====================]
               <--------Ser---------

        Parameters
        ------------
        fpr_pixels
            The row indexes to extract the front edge between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows).
        trails_pixels
            The row indexes to extract the trails between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows)
        """
        new_array = array.native.copy() * 0.0

        if fpr_pixels is not None:

            new_array = self.extractor_parallel_fpr.add_to_array(
                new_array=new_array, array=array, pixels=fpr_pixels
            )

        if trails_pixels is not None:

            new_array = self.extractor_parallel_eper.add_to_array(
                new_array=new_array, array=array, pixels=trails_pixels
            )

        return new_array

    def parallel_calibration_extraction_region_from(
        self, columns: Tuple[int, int]
    ) -> aa.type.Region2DLike:
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

    def parallel_calibration_array_2d_from(
        self, array: aa.Array2D, columns: Tuple[int, int]
    ) -> aa.Array2D:
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
        extraction_region = self.parallel_calibration_extraction_region_from(
            columns=columns
        )
        return aa.Array2D.manual_native(
            array=array.native[extraction_region.slice],
            header=array.header,
            pixel_scales=array.pixel_scales,
        )

    def parallel_calibration_extracted_layout_from(
        self, columns: Tuple[int, int]
    ) -> "Layout2DCI":
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

        extraction_region = self.parallel_calibration_extraction_region_from(
            columns=columns
        )

        return self.with_extracted_regions(extraction_region=extraction_region)

    def parallel_calibration_mask_from(
        self, mask: aa.Mask2D, columns: Tuple[int, int]
    ) -> "Mask2DCI":
        """
        Extract a mask to go with a parallel calibration array from an input mask.

        The parallel calibration array is described in the function `parallel_calibration_array_2d_from()`.
        """
        extraction_region = self.region_list[0].serial_towards_roe_full_region_from(
            shape_2d=self.shape_2d, pixels=columns
        )
        return Mask2DCI(
            mask=mask[extraction_region.slice], pixel_scales=mask.pixel_scales
        )

    def serial_epers_array_2d_from(self, array: aa.Array2D) -> aa.Array2D:
        """
        Extract an arrays of all of the serial EPERs in the serial overscan region, that are to the side of a
        charge-injection scans from a charge injection array.

        The diagram below illustrates the arrays that is extracted from a array:

        [] = read-out electronics
        [==========] = read-out register
        [..........] = serial prescan
        [pppppppppp] = parallel overscan
        [ssssssssss] = serial overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicate the region index)
        [tttttttttt] = parallel / serial charge injection region trail

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][xxxxxxxxxxxxxxxxxxxxx][sss]
          [...][ccccccccccccccccccccc][tst]
        | [...][ccccccccccccccccccccc][sts]    |
        | [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | Direction
        P [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | of
        | [...][ccccccccccccccccccccc][tst]    | clocking
        \/[...][ccccccccccccccccccccc][sts]    \/

        []     [=====================]
               <--------Ser---------

        The extracted array keeps just the trails following all charge injection scans and replaces all other
        values with 0s:

               [000000000000000000000]
               [000000000000000000000]
          [000][000000000000000000000][000]
          [000][000000000000000000000][tst]
        | [000][000000000000000000000][sts]    |
        | [000][000000000000000000000][000]    | Direction
        P [000][000000000000000000000][000]    | of
        | [000][000000000000000000000][tst]    | clocking
          [000][000000000000000000000][sts]    |

        []     [=====================]
               <--------Ser---------
        """
        array = self.serial_fprs_and_epers_array_2d_from(
            array=array, trails_pixels=(0, self.serial_overscan.total_columns)
        )
        return array

    def serial_overscan_above_epers_array_2d_from(
        self, array: aa.Array2D
    ) -> aa.Array2D:
        """
        Extract an array of the region above the EPER trails of the serial overscan.

        This region does not contain signal from either parallel or serial EPERs, however it may have a faint signal
        from charge that is trailed from the parallel EPER's in the serial direction during serial clocking.

        [] = read-out electronics
        [==========] = read-out register
        [..........] = serial prescan
        [pppppppppp] = parallel overscan
        [ssssssssss] = serial overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicate the region index)
        [tttttttttt] = parallel / serial charge injection region trail

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][xxxxxxxxxxxxxxxxxxxxx][sss]
          [...][ccccccccccccccccccccc][tst]
        | [...][ccccccccccccccccccccc][sts]    |
        | [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | Direction
        P [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | of
        | [...][ccccccccccccccccccccc][tst]    | clocking
          [...][ccccccccccccccccccccc][sts]    |

        []     [=====================]
               <--------Ser---------

        The extracted array keeps just the trails following all charge injection scans and replaces all other
        values with 0s:

               [000000000000000000000]
               [000000000000000000000]
          [000][000000000000000000000][sss]
          [000][000000000000000000000][000]
        | [000][000000000000000000000][000]    |
        | [000][000000000000000000000][sss]    | Direction
        P [000][000000000000000000000][sss]    | of
        | [000][000000000000000000000][000]    | clocking
          [000][000000000000000000000][000]    |

        []     [=====================]
               <--------Ser---------
        """
        new_array = array.native.copy() * 0.0

        new_array[self.serial_overscan.slice] = array.native[self.serial_overscan.slice]

        trails_region_list = list(
            map(
                lambda ci_region: ci_region.serial_trailing_region_from(
                    (0, self.serial_overscan.total_columns)
                ),
                self.region_list,
            )
        )

        for region in trails_region_list:
            new_array[region.slice] = 0

        return new_array

    def serial_fprs_and_epers_array_2d_from(
        self, array: aa.Array2D, fpr_pixels=None, trails_pixels=None
    ) -> aa.Array2D:
        """
        Extract an array of all of the serial FPRs and EPERs of each the charge-injection scans from a charge
        injection array.

        One can specify the range of columns that are extracted, for example:

        fpr_pixels = (0, 1) will extract just the leading front edge column.
        fpr_pixels = (0, 2) will extract the leading two front edge columns.
        trails_pixels = (0, 1) will extract the first column of trails closest to the charge injection region.

        The diagram below illustrates the arrays that is extracted from a array for `fpr_pixels=(0,2)` and
        `trails_pixels=(0,2)`:

        [] = read-out electronics
        [==========] = read-out register
        [..........] = serial prescan
        [pppppppppp] = parallel overscan
        [ssssssssss] = serial overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicate the region index)
        [tttttttttt] = parallel / serial charge injection region trail

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][xxxxxxxxxxxxxxxxxxxxx][sss]
          [...][ccccccccccccccccccccc][sts]
        | [...][ccccccccccccccccccccc][tst]    |
        | [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | Direction
        P [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | of
        | [...][ccccccccccccccccccccc][sts]    | clocking
          [...][ccccccccccccccccccccc][tst]    |

        []     [=====================]
               <--------Ser---------

        The extracted array keeps just the leading edge and trails following all charge injection scans and
        replaces all other values with 0s:

               [000000000000000000000]
               [000000000000000000000]
          [000][000000000000000000000][000]
          [000][cc0000000000000000000][st0]
        | [000][cc0000000000000000000][ts0]    |
        | [000][000000000000000000000][000]    | Direction
        P [000][000000000000000000000][000]    | of
        | [000][cc0000000000000000000][st0]    | clocking
          [000][cc0000000000000000000][st0]    |

        []     [=====================]
               <--------Ser---------

        Parameters
        ------------
        array
        fpr_pixels
            The column indexes to extract the front edge between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd rows)
        trails_pixels
            The column indexes to extract the trails between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd rows)
        """
        new_array = array.native.copy() * 0.0

        if fpr_pixels is not None:

            new_array = self.extractor_serial_fpr.add_to_array(
                new_array=new_array, array=array, pixels=fpr_pixels
            )

        if trails_pixels is not None:

            new_array = self.extractor_serial_eper.add_to_array(
                new_array=new_array, array=array, pixels=trails_pixels
            )

        return new_array

    def serial_calibration_array_2d_list_from(self, array: aa.Array2D):
        """
        Extract each charge injection region image for the serial calibration arrays when creating the
        """

        calibration_region_list = list(
            map(
                lambda ci_region: ci_region.parallel_full_region_from(
                    shape_2d=self.shape_2d
                ),
                self.region_list,
            )
        )
        return list(
            map(lambda region: array.native[region.slice], calibration_region_list)
        )

    def serial_calibration_extracted_layout_from(self, new_shape_2d, rows):

        serial_prescan = (
            (0, new_shape_2d[0], self.serial_prescan[2], self.serial_prescan[3])
            if self.serial_prescan is not None
            else None
        )
        serial_overscan = (
            (0, new_shape_2d[0], self.serial_overscan[2], self.serial_overscan[3])
            if self.serial_overscan is not None
            else None
        )

        x0 = self.region_list[0][2]
        x1 = self.region_list[0][3]
        offset = 0

        new_pattern_region_list_ci = []

        for region in self.region_list:

            labelsize = rows[1] - rows[0]
            new_pattern_region_list_ci.append(
                aa.Region2D(region=(offset, offset + labelsize, x0, x1))
            )
            offset += labelsize

        new_layout_ci = deepcopy(self)
        new_layout_ci.region_list = new_pattern_region_list_ci
        new_layout_ci.serial_prescan = serial_prescan
        new_layout_ci.serial_overscan = serial_overscan

        return new_layout_ci

    def serial_calibration_array_2d_from(
        self, array: aa.Array2D, rows: Tuple[int, int]
    ) -> aa.Array2D:
        """
        Extract a serial calibration array from a charge injection array, where this arrays is a sub-set of the
        array which can be used for serial-only calibration. Specifically, this array is all charge injection
        scans and their serial over-scan trails.

        The diagram below illustrates the arrays that is extracted from a array with column=5:

        [] = read-out electronics
        [==========] = read-out register
        [..........] = serial prescan
        [pppppppppp] = parallel overscan
        [ssssssssss] = serial overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicate the region index)
        [tttttttttt] = parallel / serial charge injection region trail

               [ppppppppppppppppppppp]
               [pppppppppppppppppppp ]
          [...][xxxxxxxxxxxxxxxxxxxxx][sss]
          [...][ccccccccccccccccccccc][tst]
        | [...][ccccccccccccccccccccc][sts]    |
        | [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | Direction
        P [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | of
        | [...][ccccccccccccccccccccc][tst]    | clocking
          [...][ccccccccccccccccccccc][sts]    |

        []     [=====================]
               <--------Ser---------

        The extracted array keeps just the trails following all charge injection scans and replaces all other
        values with 0s:

        |                                      |
        |      [cccccccccccccccc][tst]         | Direction
        P      [cccccccccccccccc][tst]         | of
        |      [cccccccccccccccc][tst]         | clocking
               [cccccccccccccccc][tst]         |

        []     [=====================]
               <--------Ser---------
        """
        calibration_images = self.serial_calibration_array_2d_list_from(array=array)
        calibration_images = list(
            map(lambda image: image[rows[0] : rows[1], :], calibration_images)
        )

        new_array = np.concatenate(calibration_images, axis=0)

        return aa.Array2D.manual(
            array=new_array, header=array.header, pixel_scales=array.pixel_scales
        )

    def serial_calibration_mask_from(
        self, mask: aa.Mask2D, rows: Tuple[int, int]
    ) -> Mask2DCI:
        """
        Extract a serial calibration array from a charge injection array, where this arrays is a sub-set of the
        array which can be used for serial-only calibration. Specifically, this array is all charge injection
        scans and their serial over-scan trails.

        The diagram below illustrates the arrays that is extracted from a array with column=5:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCDPhase panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
        [xxxxxxxxxx]                [tttttttttt] = parallel / serial charge injection region trail

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [pppppppppppppppppppp ]
          [...][xxxxxxxxxxxxxxxxxxxxx][sss]
          [...][ccccccccccccccccccccc][tst]
        | [...][ccccccccccccccccccccc][sts]    |
        | [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | Direction
        P [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | of
        | [...][ccccccccccccccccccccc][tst]    | clocking
          [...][ccccccccccccccccccccc][sts]    |

        []     [=====================]
               <--------Ser---------

        The extracted array keeps just the trails following all charge injection scans and replaces all other
        values with 0s:

        |                                      |
        |      [cccccccccccccccc][tst]         | Direction
        P      [cccccccccccccccc][tst]         | of
        |      [cccccccccccccccc][tst]         | clocking
               [cccccccccccccccc][tst]         |

        []     [=====================]
               <--------Ser---------
        """

        calibration_region_list = list(
            map(
                lambda ci_region: ci_region.parallel_full_region_from(
                    shape_2d=self.shape_2d
                ),
                self.region_list,
            )
        )
        calibration_masks = list(
            map(lambda region: mask[region.slice], calibration_region_list)
        )

        calibration_masks = list(
            map(lambda mask: mask[rows[0] : rows[1], :], calibration_masks)
        )
        return Mask2DCI(
            mask=np.concatenate(calibration_masks, axis=0),
            pixel_scales=mask.pixel_scales,
        )

    def with_extracted_regions(
        self, extraction_region: aa.type.Region2DLike
    ) -> "Layout2DCI":

        layout = deepcopy(self)

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

    def extract_line_from(self, array: aa.Array2D, line_region: str) -> aa.Array1D:

        if line_region == "parallel_front_edge":
            return self.extractor_parallel_fpr.binned_array_1d_from(
                array=array, pixels=(0, self.extractor_parallel_fpr.total_rows_min)
            )
        elif line_region == "parallel_epers":
            return self.extractor_parallel_eper.binned_array_1d_from(
                array=array, pixels=(0, self.smallest_parallel_rows_between_ci_regions)
            )
        elif line_region == "serial_front_edge":
            return self.extractor_serial_fpr.binned_array_1d_from(
                array=array, pixels=(0, self.extractor_serial_fpr.total_columns_min)
            )
        elif line_region == "serial_trails":
            return self.extractor_serial_eper.binned_array_1d_from(
                array=array, pixels=(0, self.serial_eper_pixels)
            )
        else:
            raise exc.PlottingException(
                "The line region specified for the plotting of a line was invalid"
            )

    @classmethod
    def from_euclid_fits_header(cls, ext_header, do_rotation):

        serial_overscan_size = ext_header.get("OVRSCANX", default=None)
        serial_prescan_size = ext_header.get("PRESCANX", default=None)
        serial_size = ext_header.get("NAXIS1", default=None)
        parallel_size = ext_header.get("NAXIS2", default=None)

        electronics = ElectronicsCI.from_ext_header(ext_header=ext_header)

        layout = aa.euclid.Layout2DEuclid.from_fits_header(ext_header=ext_header)

        if do_rotation:
            roe_corner = layout.original_roe_corner
        else:
            roe_corner = (1, 0)

        region_ci_list = region_list_ci_from(
            serial_prescan_size=serial_prescan_size,
            serial_overscan_size=serial_overscan_size,
            serial_size=serial_size,
            parallel_size=parallel_size,
            injection_on=electronics.injection_on,
            injection_off=electronics.injection_off,
            injection_total=electronics.injection_total,
            roe_corner=roe_corner,
        )

        return cls(
            shape_2d=(parallel_size, serial_size),
            region_list=region_ci_list,
            original_roe_corner=layout.original_roe_corner,
            parallel_overscan=layout.parallel_overscan,
            serial_prescan=layout.serial_prescan,
            serial_overscan=layout.serial_overscan,
            electronics=electronics,
        )


class ElectronicsCI:
    def __init__(
        self,
        injection_on: Optional[int] = None,
        injection_off: Optional[int] = None,
        injection_start: Optional[int] = None,
        injection_end: Optional[int] = None,
        ig_1: Optional[float] = None,
        ig_2: Optional[float] = None,
    ):
        """
        Stores the electronics parameters contained for a charge injection line image, with this class currently
        specific to those in Euclid.

        These are extracted from the .fits file header of a charge injection image, with the original .fits
        headers included in the `as_ext_header_dict` property.

        The `injection_on` and `injection_off` parameters determine the number of pixels the charge injection is
        held on and then off for. The `v_start` and `v_end` parameters define the pixels where the charge injection
        starts and ends.

        For example, take a CCD which has 1820 rows of pixels, where:

        - `injection_on=100`
        - `injection_off=200`
        - `v_start=10`
        - `v_end`=1810

        Starting from row 10, for every 300 rows of pixels there first 100 pixels will contain charge injection and
        the remaining 200 rows will not (they will contain EPER trails). This pattern will be repeated 6 times over
        the next 1800 pixels of the CCD with the charge injection ending at 1810.

        NOTE: The charge injection electrons have the following four parameters:

        VSTART_CHJ_INJ
        VEND_CHJ_INJ
        VSTART
        VEND

        I do not yet know which of these maps to which fits header. I am currently assuming all 4 correspond to
        `v_start` and `v_end`, albeit their functionality is not used specifically.

        Parameters
        ----------
        injection_on
            The number of rows of pixels the charge injection is held on for per charge injection region.
        injection_off
            The number of rows of pixels the charge injection is held off for per charge injection region.
        injection_start
            The pixel row where the charge injection begins.
        injection_end
            The pixel row where the charge injection ends.
        ig_1
            The voltage of injection gate 1.
        ig_2
            The voltage of injection gate 2.
        """
        self.injection_on = injection_on
        self.injection_off = injection_off
        self.injection_start = injection_start
        self.injection_end = injection_end
        self.ig_1 = ig_1
        self.ig_2 = ig_2

    @classmethod
    def from_ext_header(cls, ext_header: Dict) -> "ElectronicsCI":
        """
        Creates the charge injection electronics from a Euclid charge injection imaging .fits header.

        Parameters
        ----------
        ext_header
            The .fits header dictionary of a Euclid charge injection image.
        """
        injection_on = ext_header["CI_IJON"]
        injection_off = ext_header["CI_IJOFF"]
        injection_start = ext_header["CI_VSTAR"]
        injection_end = ext_header["CI_VEND"]

        return ElectronicsCI(
            injection_on=injection_on,
            injection_off=injection_off,
            injection_start=injection_start,
            injection_end=injection_end,
        )

    @property
    def as_ext_header_dict(self) -> Dict:
        """
        Returns the charge injection electronics as a dictionary which is representative of the parameter values
        stored in a Euclid charge injection .fits image.
        """
        return {
            "CI_IJON": self.injection_on,
            "CI_IJOFF": self.injection_off,
            "CI_VSTAR": self.injection_start,
            "CI_VEND": self.injection_end,
        }

    @property
    def injection_total(self) -> int:
        """
        The total number of charge injection regions for these electronics settings.
        """
        return math.floor(
            (self.injection_end - self.injection_start)
            / (self.injection_on + self.injection_off)
        )


def region_list_ci_from(
    injection_on: int,
    injection_off: int,
    injection_total: int,
    parallel_size: int,
    serial_size: int,
    serial_prescan_size: int,
    serial_overscan_size: int,
    roe_corner: Tuple[int, int],
):

    region_list_ci = []

    injection_start_count = 0

    for index in range(injection_total):

        if roe_corner == (0, 0):

            ci_region = (
                parallel_size - (injection_start_count + injection_on),
                parallel_size - injection_start_count,
                serial_prescan_size,
                serial_size - serial_overscan_size,
            )

        elif roe_corner == (1, 0):

            ci_region = (
                injection_start_count,
                injection_start_count + injection_on,
                serial_prescan_size,
                serial_size - serial_overscan_size,
            )

        elif roe_corner == (0, 1):

            ci_region = (
                parallel_size - (injection_start_count + injection_on),
                parallel_size - injection_start_count,
                serial_overscan_size,
                serial_size - serial_prescan_size,
            )

        elif roe_corner == (1, 1):

            ci_region = (
                injection_start_count,
                injection_start_count + injection_on,
                serial_overscan_size,
                serial_size - serial_prescan_size,
            )

        region_list_ci.append(ci_region)

        injection_start_count += injection_on + injection_off

    return region_list_ci
