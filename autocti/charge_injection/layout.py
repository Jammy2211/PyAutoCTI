from copy import deepcopy
import numpy as np
from typing import List, Tuple

import autoarray as aa

from autocti import exc
from autocti.charge_injection.extractor import Extractor2DParallelFPR
from autocti.charge_injection.extractor import Extractor2DParallelEPER
from autocti.charge_injection.extractor import Extractor2DSerialFPR
from autocti.charge_injection.extractor import Extractor2DSerialEPER
from autocti.charge_injection.mask_2d import Mask2DCI


class AbstractLayout2DCI(aa.Layout2D):
    def __init__(
        self,
        shape_2d: Tuple[int, int],
        normalization,
        region_list: List[aa.Region2D],
        original_roe_corner: Tuple[int, int] = (1, 0),
        parallel_overscan: Tuple[int, int, int, int] = None,
        serial_prescan: Tuple[int, int, int, int] = None,
        serial_overscan: Tuple[int, int, int, int] = None,
    ):
        """
        Abstract base class for a charge injection layout, which defines the regions charge injections appears
        on charge-injection imaging alongside other properties.

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
        """

        self.region_list = list(map(aa.Region2D, region_list))

        for region in self.region_list:

            if region.y1 > shape_2d[0] or region.x1 > shape_2d[1]:
                raise exc.LayoutException(
                    "The charge injection layout_ci regions are bigger than the image image_shape"
                )

        self.extractor_parallel_front_edge = Extractor2DParallelFPR(
            region_list=region_list
        )
        self.extractor_parallel_epers = Extractor2DParallelEPER(region_list=region_list)
        self.extractor_serial_front_edge = Extractor2DSerialFPR(region_list=region_list)
        self.extractor_serial_trails = Extractor2DSerialEPER(region_list=region_list)

        super().__init__(
            shape_2d=shape_2d,
            original_roe_corner=original_roe_corner,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

        self.normalization = normalization

    def after_extraction_from(
        self, extraction_region: Tuple[int, int, int, int]
    ) -> "AbstractLayout2DCI":
        """
        The charge injection region after an extraction is performed on an associated charge injection image, where
        the extraction is defined by a region of pixel coordinates (top-row, bottom-row, left-column, right-column)

        For example, if a charge injection region occupies the pixels (0, 10, 0, 5) on a 100 x 100 charge injection
        image, and the first 5 columns of this charge injection image are extracted to create a 100 x 5 image, the
         new charge injection region will be (0, 20, 0, 5).

        Parameters
        ----------
        extraction_region

        Returns
        -------

        """

        layout = super().after_extraction_from(extraction_region=extraction_region)

        region_list = [
            aa.util.layout.region_after_extraction(
                original_region=region, extraction_region=extraction_region
            )
            for region in self.region_list
        ]

        return self.__class__(
            original_roe_corner=self.original_roe_corner,
            shape_2d=self.shape_2d,
            normalization=self.normalization,
            region_list=region_list,
            parallel_overscan=layout.parallel_overscan,
            serial_prescan=layout.serial_prescan,
            serial_overscan=layout.serial_overscan,
        )

    @property
    def pixels_between_regions(self) -> List[int]:
        return [
            self.region_list[i + 1].y0 - self.region_list[i].y1
            for i in range(len(self.region_list) - 1)
        ]

    @property
    def parallel_eper_size_to_array_edge(self):
        return self.shape_2d[0] - np.max([region.y1 for region in self.region_list])

    @property
    def smallest_parallel_epers_rows_to_array_edge(self):

        pixels_between_regions = self.pixels_between_regions
        pixels_between_regions.append(self.parallel_eper_size_to_array_edge)
        return np.min(pixels_between_regions)

    def array_2d_of_regions_from(self, array: aa.Array2D) -> aa.Array2D:
        """
        Extract all of the charge-injection regions from an input `array2D` object and returns them as a new `array2D`
        object where these extracted regions are included and all other entries are zeros.

        The diagram below illustrates the regions that are extracted from the input array:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCDPhase panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
        [xxxxxxxxxx]

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][xxxxxxxxxxxxxxxxxxxxx][sss]
          [...][ccccccccccccccccccccc][sss]
        | [...][ccccccccccccccccccccc][sss]    |
        | [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | Direction
        P [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | of
        | [...][ccccccccccccccccccccc][sss]    | clocking
          [...][ccccccccccccccccccccc][sss]    |

        []     [=====================]
               <---------S----------

        The extracted array keeps just the charge injection region and replaces all other values with 0s:

               [000000000000000000000]
               [000000000000000000000]
          [000][000000000000000000000][000]
          [000][ccccccccccccccccccccc][000]
        | [000][ccccccccccccccccccccc][000]    |
        | [000][000000000000000000000][000]    | Direction
        P [000][000000000000000000000][000]    | of
        | [000][ccccccccccccccccccccc][000]    | clocking
          [000][ccccccccccccccccccccc][000]    |

        []     [=====================]
               <---------S----------

        """

        new_array = array.native.copy() * 0.0

        for region in self.region_list:
            new_array[region.slice] += array.native[region.slice]

        return new_array

    def array_2d_of_non_regions_from(self, array: aa.Array2D) -> aa.Array2D:
        """
        Extract all of the data values in an input `array2D` that do not overlap the charge injection regions. This
        includes many areas of the image (e.g. the serial prescan, serial overscan) but is typically used to extract
        a `array2D` that contains the parallel trails that follow the charge-injection regions.

        The diagram below illustrates the `array2D` that is extracted from the input array:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCDPhase panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
        [xxxxxxxxxx]                [tttttttttt] = parallel / serial charge injection region trail

        P = Parallel Direction      S = Serial Direction

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
               <---------S----------

        The extracted array keeps just the trails following all charge injection scans and replaces all other
        values with 0s:

               [tptpptptptpptpptpptpt]
               [tptptptpptpttptptptpt]
          [000][ttttttttttttttttttttt][000]
          [000][000000000000000000000][000]
        | [000][000000000000000000000][000]    |
        | [000][ttttttttttttttttttttt][000]    | Direction
        P [000][ttttttttttttttttttttt][000]    | of
        | [000][000000000000000000000][000]    | clocking
          [000][000000000000000000000][000]    |

        []     [=====================]
               <---------S----------
        """

        non_regions_ci_array = array.native.copy()

        for region in self.region_list:
            non_regions_ci_array[region.slice] = 0.0

        return non_regions_ci_array

    def array_2d_of_parallel_epers_from(self, array: aa.Array2D) -> aa.Array2D:
        """
        Extract all of the data values in an input `array2D` that do not overlap the charge injection regions or the
        serial prescan / serial overscan regions.

        This  extracts a `array2D` that contains only regions of the data where there are parallel trails (e.g. those
        that follow the charge-injection regions).

        The diagram below illustrates the `array2D` that is extracted from the input array:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCDPhase panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
        [xxxxxxxxxx]                [tttttttttt] = parallel / serial charge injection region trail

        P = Parallel Direction      S = Serial Direction

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
               <---------S----------

        The extracted array keeps just the trails following all charge injection scans and replaces all other
        values with 0s:

               [tptpptptptpptpptpptpt]
               [tptptptpptpttptptptpt]
          [000][ttttttttttttttttttttt][000]
          [000][000000000000000000000][000]
        | [000][000000000000000000000][000]    |
        | [000][ttttttttttttttttttttt][000]    | Direction
        P [000][ttttttttttttttttttttt][000]    | of
        | [000][000000000000000000000][000]    | clocking
          [000][000000000000000000000][000]    |

        []     [=====================]
               <---------S----------
        """

        parallel_array = self.array_2d_of_non_regions_from(array=array)

        parallel_array.native[self.serial_prescan.slice] = 0.0
        parallel_array.native[self.serial_overscan.slice] = 0.0

        return parallel_array

    def array_2d_of_parallel_fprs_and_epers_from(
        self,
        array: aa.Array2D,
        fpr_range: Tuple[int, int] = None,
        trails_rows: Tuple[int, int] = None,
    ) -> aa.Array2D:
        """
        Extract all of the data values in an input `array2D` corresponding to the parallel front edges and trails of
        each the charge-injection region.

        One can specify the range of rows that are extracted, for example:

        fpr_range = (0, 1) will extract just the first leading front edge row.
        fpr_range = (0, 2) will extract the leading two front edge rows.
        trails_rows = (0, 1) will extract the first row of trails closest to the charge injection region.

        The diagram below illustrates the arrays that are extracted from the input array for `fpr_range=(0,1)`
        and `trails_rows=(0,1)`:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCDPhase panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
        [xxxxxxxxxx]                [tttttttttt] = parallel / serial charge injection region trail

        P = Parallel Direction      S = Serial Direction

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
               <---------S----------

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
               <---------S----------

        Parameters
        ------------
        fpr_range
            The row indexes to extract the front edge between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows).
        trails_rows
            The row indexes to extract the trails between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows)
        """
        new_array = array.native.copy() * 0.0

        if fpr_range is not None:

            new_array = self.extractor_parallel_front_edge.add_to_array(
                new_array=new_array, array=array, pixels=fpr_range
            )

        if trails_rows is not None:

            new_array = self.extractor_parallel_epers.add_to_array(
                new_array=new_array, array=array, rows=trails_rows
            )

        return new_array

    def extraction_region_for_parallel_calibration_from(self, columns: Tuple[int, int]):
        """
        Extract a parallel calibration array from an input array, where this array contains a sub-set of the input
        array which is specifically used for only parallel CTI calibration. This array is simply a specified number
        of columns that are closest to the read-out electronics.

        The diagram below illustrates the arrays that is extracted from a array with columns=(0, 3):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCDPhase panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
        [xxxxxxxxxx]                [tttttttttt] = parallel / parallel charge injection region trail

        P = Parallel Direction      S = Serial Direction

               [ptptptptptptptptptptp]
               [tptptptptptptptptptpt]
          [...][xxxxxxxxxxxxxxxxxxxxx][sss]
          [...][ccccccccccccccccccccc][sss]
        | [...][ccccccccccccccccccccc][sts]    |
        | [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | Direction
        P [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | of
        | [...][ccccccccccccccccccccc][sss]    | clocking
          [...][ccccccccccccccccccccc][sss]    |

        []     [=====================]
               <---------S----------

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
               [ccc]                           |

        []     [=====================]
               <---------S----------
        """
        return self.region_list[0].serial_towards_roe_full_region_from(
            shape_2d=self.shape_2d, pixels=columns
        )

    def array_2d_for_parallel_calibration_from(
        self, array: aa.Array2D, columns: Tuple[int, int]
    ):
        """
        Extract a parallel calibration array from an input array, where this array contains a sub-set of the input
        array which is specifically used for only parallel CTI calibration. This array is simply a specified number
        of columns that are closest to the read-out electronics.

        The diagram below illustrates the arrays that is extracted from a array with columns=(0, 3):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCDPhase panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
        [xxxxxxxxxx]                [tttttttttt] = parallel / parallel charge injection region trail

        P = Parallel Direction      S = Serial Direction

               [ptptptptptptptptptptp]
               [tptptptptptptptptptpt]
          [...][xxxxxxxxxxxxxxxxxxxxx][sss]
          [...][ccccccccccccccccccccc][sss]
        | [...][ccccccccccccccccccccc][sts]    |
        | [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | Direction
        P [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | of
        | [...][ccccccccccccccccccccc][sss]    | clocking
          [...][ccccccccccccccccccccc][sss]    |

        []     [=====================]
               <---------S----------

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
               [ccc]                           |

        []     [=====================]
               <---------S----------
        """
        extraction_region = self.extraction_region_for_parallel_calibration_from(
            columns=columns
        )
        return aa.Array2D.manual_native(
            array=array.native[extraction_region.slice],
            header=array.header,
            pixel_scales=array.pixel_scales,
        )

    def extracted_layout_for_parallel_calibration_from(self, columns: Tuple[int, int]):
        """
        Extract a parallel calibration array from an input array, where this array contains a sub-set of the input
        array which is specifically used for only parallel CTI calibration. This array is simply a specified number
        of columns that are closest to the read-out electronics.

        The diagram below illustrates the arrays that is extracted from a array with columns=(0, 3):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCDPhase panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
        [xxxxxxxxxx]                [tttttttttt] = parallel / parallel charge injection region trail

        P = Parallel Direction      S = Serial Direction

               [ptptptptptptptptptptp]
               [tptptptptptptptptptpt]
          [...][xxxxxxxxxxxxxxxxxxxxx][sss]
          [...][ccccccccccccccccccccc][sss]
        | [...][ccccccccccccccccccccc][sts]    |
        | [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | Direction
        P [...][xxxxxxxxxxxxxxxxxxxxx][sss]    | of
        | [...][ccccccccccccccccccccc][sss]    | clocking
          [...][ccccccccccccccccccccc][sss]    |

        []     [=====================]
               <---------S----------

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
               [ccc]                           |

        []     [=====================]
               <---------S----------
        """

        extraction_region = self.extraction_region_for_parallel_calibration_from(
            columns=columns
        )

        return self.with_extracted_regions(extraction_region=extraction_region)

    def mask_for_parallel_calibration_from(self, mask, columns):
        extraction_region = self.region_list[0].serial_towards_roe_full_region_from(
            shape_2d=self.shape_2d, pixels=columns
        )
        return Mask2DCI(
            mask=mask[extraction_region.slice], pixel_scales=mask.pixel_scales
        )

    def array_2d_of_serial_trails_from(self, array: aa.Array2D):
        """Extract an arrays of all of the serial trails in the serial overscan region, that are to the side of a
        charge-injection scans from a charge injection array.

        The diagram below illustrates the arrays that is extracted from a array:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCDPhase panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
        [xxxxxxxxxx]                [tttttttttt] = parallel / serial charge injection region trail

        P = Parallel Direction      S = Serial Direction

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
               <---------S----------

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
               <---------S----------
        """
        array = self.array_2d_of_serial_edges_and_epers_array(
            array=array, trails_columns=(0, self.serial_overscan.total_columns)
        )
        return array

    def array_2d_of_serial_overscan_above_trails_from(self, array: aa.Array2D):
        """
        Extract an arrays of all of the scans of the serial overscan that don't contain trails from a
        charge injection region (i.e. are not to the side of one).

        The diagram below illustrates the arrays that is extracted from a array:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCDPhase panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
        [xxxxxxxxxx]                [tttttttttt] = parallel / serial charge injection region trail

        P = Parallel Direction      S = Serial Direction

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
               <---------S----------

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
               <---------S----------
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

    def array_2d_of_serial_edges_and_epers_array(
        self, array: aa.Array2D, front_edge_columns=None, trails_columns=None
    ):
        """
        Extract an arrays of all of the serial front edges and trails of each the charge-injection scans from
        a charge injection array.

        One can specify the range of columns that are extracted, for example:

        front_edge_columns = (0, 1) will extract just the leading front edge column.
        front_edge_columns = (0, 2) will extract the leading two front edge columns.
        trails_columns = (0, 1) will extract the first column of trails closest to the charge injection region.

        The diagram below illustrates the arrays that is extracted from a array for front_edge_columns=(0,2) and
        trails_columns=(0,2):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCDPhase panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
        [xxxxxxxxxx]                [tttttttttt] = parallel / serial charge injection region trail

        P = Parallel Direction      S = Serial Direction

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
               <---------S----------

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
               <---------S----------

        Parameters
        ------------
        array
        front_edge_columns
            The column indexes to extract the front edge between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd rows)
        trails_columns
            The column indexes to extract the trails between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd rows)
        """
        new_array = array.native.copy() * 0.0

        if front_edge_columns is not None:

            new_array = self.extractor_serial_front_edge.add_to_array(
                new_array=new_array, array=array, columns=front_edge_columns
            )

        if trails_columns is not None:

            new_array = self.extractor_serial_trails.add_to_array(
                new_array=new_array, array=array, columns=trails_columns
            )

        return new_array

    def array_2d_list_for_serial_calibration(self, array: aa.Array2D):
        """
        Extract each charge injection region image for the serial calibration arrays above.
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

    def extracted_layout_for_serial_calibration_from(self, new_shape_2d, rows):

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

    def array_2d_for_serial_calibration_from(
        self, array: aa.Array2D, rows: Tuple[int, int]
    ):
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
               <---------S----------

        The extracted array keeps just the trails following all charge injection scans and replaces all other
        values with 0s:

        |                                      |
        |      [cccccccccccccccc][tst]         | Direction
        P      [cccccccccccccccc][tst]         | of
        |      [cccccccccccccccc][tst]         | clocking
               [cccccccccccccccc][tst]         |

        []     [=====================]
               <---------S----------
        """
        calibration_images = self.array_2d_list_for_serial_calibration(array=array)
        calibration_images = list(
            map(lambda image: image[rows[0] : rows[1], :], calibration_images)
        )

        new_array = np.concatenate(calibration_images, axis=0)

        return aa.Array2D.manual(
            array=new_array, header=array.header, pixel_scales=array.pixel_scales
        )

    def mask_for_serial_calibration_from(self, mask, rows: Tuple[int, int]):
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
               <---------S----------

        The extracted array keeps just the trails following all charge injection scans and replaces all other
        values with 0s:

        |                                      |
        |      [cccccccccccccccc][tst]         | Direction
        P      [cccccccccccccccc][tst]         | of
        |      [cccccccccccccccc][tst]         | clocking
               [cccccccccccccccc][tst]         |

        []     [=====================]
               <---------S----------
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

    def with_extracted_regions(self, extraction_region):

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
            return self.extractor_parallel_front_edge.binned_array_1d_from(
                array=array, pixels=(0, self.extractor_parallel_front_edge.total_rows_min)
            )
        elif line_region == "parallel_epers":
            return self.extractor_parallel_epers.binned_array_1d_from(
                array=array, rows=(0, self.smallest_parallel_epers_rows_to_array_edge)
            )
        elif line_region == "serial_front_edge":
            return self.extractor_serial_front_edge.binned_array_1d_from(
                array=array,
                columns=(0, self.extractor_serial_front_edge.total_columns_min),
            )
        elif line_region == "serial_trails":
            return self.extractor_serial_trails.binned_array_1d_from(
                array=array, columns=(0, self.serial_trails_columns)
            )
        else:
            raise exc.PlottingException(
                "The line region specified for the plotting of a line was invalid"
            )


class Layout2DCI(AbstractLayout2DCI):
    """
    A uniform charge injection layout_ci, which is defined by the regions it appears on the charge injection \
    array and its normalization.
    """

    @classmethod
    def from_euclid_fits_header(cls, ext_header, do_rotation):

        serial_overscan_size = ext_header.get("OVRSCANX", default=None)
        serial_prescan_size = ext_header.get("PRESCANX", default=None)
        serial_size = ext_header.get("NAXIS1", default=None)
        parallel_size = ext_header.get("NAXIS2", default=None)

        injection_on = ext_header["CI_IJON"]
        injection_off = ext_header["CI_IJOFF"]

        injection_start = ext_header["CI_VSTAR"]
        injection_end = ext_header["CI_VEND"]

        injection_total = (injection_end - injection_start) / (
            injection_on + injection_off
        )

        import math

        math.floor(injection_total)

        layout = aa.euclid.Layout2DEuclid.from_fits_header(ext_header=ext_header)

        # TODO : Compute via .fits headers without injction_total.

        if do_rotation:
            roe_corner = layout.original_roe_corner
        else:
            roe_corner = (1, 0)

        region_ci_list = region_list_ci_from(
            serial_prescan_size=serial_prescan_size,
            serial_overscan_size=serial_overscan_size,
            serial_size=serial_size,
            parallel_size=parallel_size,
            injection_on=injection_on,
            injection_off=injection_off,
            injection_total=injection_total,
            roe_corner=roe_corner,
        )

        # The header "CI_IG1" is used as a placeholder for the normalization currently.
        normalization = ext_header["CI_IG1"]

        return cls(
            shape_2d=(parallel_size, serial_size),
            normalization=normalization,
            region_list=region_ci_list,
            original_roe_corner=layout.original_roe_corner,
            parallel_overscan=layout.parallel_overscan,
            serial_prescan=layout.serial_prescan,
            serial_overscan=layout.serial_overscan,
        )

    def pre_cti_data_from(self, shape_native, pixel_scales):
        """Use this charge injection layout_ci to generate a pre-cti charge injection image. This is performed by \
        going to its charge injection regions and adding the charge injection normalization value.

        Parameters
        -----------
        shape_native
            The image_shape of the pre_cti_datas to be created.
        """

        pre_cti_data = np.zeros(shape_native)

        for region in self.region_list:
            pre_cti_data[region.slice] += self.normalization

        return aa.Array2D.manual(array=pre_cti_data, pixel_scales=pixel_scales)


class Layout2DCINonUniform(AbstractLayout2DCI):
    def __init__(
        self,
        shape_2d,
        normalization,
        region_list,
        row_slope,
        column_sigma=None,
        maximum_normalization=np.inf,
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
    ):
        """A non-uniform charge injection layout_ci, which is defined by the regions it appears on a charge injection
        array and its average normalization.

        Non-uniformity across the columns of a charge injection layout_ci is due to spikes / drops in the current that
        injects the charge. This is a noisy process, leading to non-uniformity with no regularity / smoothness. Thus,
        it cannot be modeled with an analytic profile, and must be assumed as prior-knowledge about the charge
        injection electronics or estimated from the observed charge injection ci_data.

        Non-uniformity across the rows of a charge injection layout_ci is due to a drop-off in voltage in the current.
        Therefore, it appears smooth and be modeled as an analytic function, which this code assumes is a
        power-law with slope row_slope.

        Parameters
        -----------
        normalization
            The normalization of the charge injection region.
        region_list : [(int,)]
            A list of the integer coordinates specifying the corners of each charge injection region
            (top-row, bottom-row, left-column, right-column).
        row_slope
            The power-law slope of non-uniformity in the row charge injection profile.
        """

        super().__init__(
            shape_2d=shape_2d,
            normalization=normalization,
            region_list=region_list,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

        self.row_slope = row_slope
        self.column_sigma = column_sigma
        self.maximum_normalization = maximum_normalization

    def region_ci_from(self, region_dimensions, ci_seed):
        """Generate the non-uniform charge distribution of a charge injection region. This includes non-uniformity \
        across both the rows and columns of the charge injection region.

        Before adding non-uniformity to the rows and columns, we assume an input charge injection level \
        (e.g. the average current being injected). We then simulator non-uniformity in this region.

        Non-uniformity in the columns is caused by sharp peaks and troughs in the input charge current. To simulator  \
        this, we change the normalization of each column by drawing its normalization value from a Gaussian \
        distribution which has a mean of the input normalization and standard deviation *column_sigma*. The seed \
        of the random number generator ensures that the non-uniform charge injection update_via_regions of each pre_cti_datas \
        are identical.

        Non-uniformity in the rows is caused by the charge smoothly decreasing as the injection is switched off. To \
        simulator this, we assume the charge level as a function of row number is not flat but defined by a \
        power-law with slope *row_slope*.

        Non-uniform charge injection images are generated using the function *simulate_pre_cti*, which uses this \
        function.

        Parameters
        -----------
        maximum_normalization
        column_sigma
        region_dimensions
            The size of the non-uniform charge injection region.
        ci_seed : int
            Input seed for the random number generator to give reproducible results.
        """

        np.random.seed(ci_seed)

        ci_rows = region_dimensions[0]
        ci_columns = region_dimensions[1]
        ci_region = np.zeros(region_dimensions)

        for column_number in range(ci_columns):

            column_normalization = 0
            while (
                column_normalization <= 0
                or column_normalization >= self.maximum_normalization
            ):
                column_normalization = np.random.normal(
                    self.normalization, self.column_sigma
                )

            ci_region[0:ci_rows, column_number] = self.generate_column(
                size=ci_rows, normalization=column_normalization
            )

        return ci_region

    def pre_cti_data_from(self, shape_native, pixel_scales, ci_seed=-1) -> aa.Array2D:
        """Use this charge injection layout_ci to generate a pre-cti charge injection image. This is performed by going \
        to its charge injection regions and adding its non-uniform charge distribution.

        For one column of a non-uniform charge injection pre_cti_datas, it is assumed that each non-uniform charge \
        injection region has the same overall normalization value (after drawing this value randomly from a Gaussian \
        distribution). Physically, this is true provided the spikes / troughs in the current that cause \
        non-uniformity occur in an identical fashion for the generation of each charge injection region.

        Parameters
        -----------
        column_sigma
        shape_native
            The image_shape of the pre_cti_datas to be created.
        maximum_normalization

        ci_seed : int
            Input ci_seed for the random number generator to give reproducible results. A new ci_seed is always used for each \
            pre_cti_datas, ensuring each non-uniform ci_region has the same column non-uniformity layout_ci.
        """

        pre_cti_data = np.zeros(shape_native)

        if ci_seed == -1:
            ci_seed = np.random.randint(
                0, int(1e9)
            )  # Use one ci_seed, so all regions have identical column
            # non-uniformity.

        for region in self.region_list:
            pre_cti_data[region.slice] += self.region_ci_from(
                region_dimensions=region.shape, ci_seed=ci_seed
            )

        try:
            return aa.Array2D.manual(array=pre_cti_data, pixel_scales=pixel_scales)
        except KeyError:
            return pre_cti_data

    def generate_column(self, size, normalization):
        """Generate a column of non-uniform charge, including row non-uniformity.

        The pixel-numbering used to generate non-uniformity across the charge injection rows runs from 1 -> size

        Parameters
        -----------
        size : int
            The size of the non-uniform column of charge
        normalization
            The input normalization of the column's charge e.g. the level of charge injected.

        """
        return normalization * (np.arange(1, size + 1)) ** self.row_slope


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
