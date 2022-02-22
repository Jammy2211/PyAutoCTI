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


class Extractor2DSerialCalibration:
    def __init__(
        self,
        shape_2d,
        region_list,
        serial_prescan: Optional[aa.type.Region2DLike] = None,
        serial_overscan: Optional[aa.type.Region2DLike] = None,
    ):

        self.shape_2d = shape_2d
        self.region_list = list(map(aa.Region2D, region_list))
        self.serial_prescan = serial_prescan
        self.serial_overscan = serial_overscan

    def array_2d_list_from(self, array: aa.Array2D):
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

    def extracted_layout_from(self, layout, new_shape_2d, rows):

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

        new_layout = deepcopy(layout)
        new_layout.region_list = new_pattern_region_list_ci
        new_layout.serial_prescan = serial_prescan
        new_layout.serial_overscan = serial_overscan

        return new_layout

    def array_2d_from(self, array: aa.Array2D, rows: Tuple[int, int]) -> aa.Array2D:
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
        calibration_images = self.array_2d_list_from(array=array)
        calibration_images = list(
            map(lambda image: image[rows[0] : rows[1], :], calibration_images)
        )

        new_array = np.concatenate(calibration_images, axis=0)

        return aa.Array2D.manual(
            array=new_array, header=array.header, pixel_scales=array.pixel_scales
        )

    def mask_2d_from(self, mask: aa.Mask2D, rows: Tuple[int, int]) -> Mask2DCI:
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
