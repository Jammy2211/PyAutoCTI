from copy import deepcopy
import math
import numpy as np
from typing import Dict, List, Optional, Tuple

import autoarray as aa

from autocti import exc


class Layout2D(aa.Layout2D):
    def __init__(
        self,
        shape_2d: Tuple[int, int],
        region_list: aa.type.Region2DList,
        original_roe_corner: Tuple[int, int] = (1, 0),
        parallel_overscan: Optional[aa.type.Region2DLike] = None,
        serial_prescan: Optional[aa.type.Region2DLike] = None,
        serial_overscan: Optional[aa.type.Region2DLike] = None,
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

        from autocti.extract.two_d.parallel_fpr import Extract2DParallelFPR
        from autocti.extract.two_d.parallel_eper import Extract2DParallelEPER
        from autocti.extract.two_d.serial_fpr import Extract2DSerialFPR
        from autocti.extract.two_d.serial_eper import Extract2DSerialEPER
        from autocti.extract.two_d.parallel_calibration import (
            Extract2DParallelCalibration,
        )
        from autocti.extract.two_d.serial_calibration import Extract2DSerialCalibration
        from autocti.extract.two_d.misc import Extract2DMisc

        self.region_list = list(map(aa.Region2D, region_list))

        for region in self.region_list:

            if region.y1 > shape_2d[0] or region.x1 > shape_2d[1]:
                raise exc.LayoutException(
                    "The charge injection layout_ci regions are bigger than the image image_shape"
                )

        super().__init__(
            shape_2d=shape_2d,
            original_roe_corner=original_roe_corner,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

        self.extract_parallel_fpr = Extract2DParallelFPR(region_list=region_list)
        self.extract_parallel_eper = Extract2DParallelEPER(
            region_list=region_list,
            serial_prescan=self.serial_prescan,
            serial_overscan=self.serial_overscan,
        )
        self.extract_serial_fpr = Extract2DSerialFPR(region_list=region_list)
        self.extract_serial_eper = Extract2DSerialEPER(region_list=region_list)

        self.extract_serial_calibration = Extract2DSerialCalibration(
            shape_2d=shape_2d,
            region_list=region_list,
            serial_prescan=self.serial_prescan,
            serial_overscan=self.serial_overscan,
        )

        self.extract_parallel_calibration = Extract2DParallelCalibration(
            shape_2d=shape_2d, region_list=region_list
        )

        self.extract_misc = Extract2DMisc(
            region_list=region_list,
            serial_prescan=self.serial_prescan,
            serial_overscan=self.serial_overscan,
        )

    def layout_extracted_from(
        self, extraction_region: aa.type.Region2DLike
    ) -> "Layout2D":
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

    def with_extracted_regions(
        self, extraction_region: aa.type.Region2DLike
    ) -> "Layout2D":

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
            return self.extract_parallel_fpr.binned_array_1d_from(
                array=array, pixels=(0, self.extract_parallel_fpr.total_rows_min)
            )
        elif line_region == "parallel_epers":
            return self.extract_parallel_eper.binned_array_1d_from(
                array=array, pixels=(0, self.smallest_parallel_rows_between_ci_regions)
            )
        elif line_region == "serial_front_edge":
            return self.extract_serial_fpr.binned_array_1d_from(
                array=array, pixels=(0, self.extract_serial_fpr.total_columns_min)
            )
        elif line_region == "serial_trails":
            return self.extract_serial_eper.binned_array_1d_from(
                array=array, pixels=(0, self.serial_eper_pixels)
            )
        else:
            raise exc.PlottingException(
                "The line region specified for the plotting of a line was invalid"
            )
