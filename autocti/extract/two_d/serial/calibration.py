from copy import deepcopy
import numpy as np
from typing import Optional, Tuple

import autoarray as aa

from autocti.mask.mask_2d import Mask2D


class Extract2DSerialCalibration:
    def __init__(
        self,
        shape_2d: Tuple[int, int],
        region_list: aa.type.Region2DList,
        serial_prescan: Optional[aa.type.Region2DLike] = None,
        serial_overscan: Optional[aa.type.Region2DLike] = None,
    ):
        """
        Class containing methods for extracting a serial calibration dataset from a 2D CTI calibration dataset.

        The serial calibration region is the region of a dataset that is necessary for fitting a serial-only CTI
        model. For example, for charge injection imaging, serial EPERs form only in rows of the CCD where charge
        is injected and the regions in between have no signal. The serial calibration dataset therefore extracts only
        these rows.

        A subset of the serial calibration data may also be extracted (e.g. only the first row of every charge region)
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
        serial_prescan
            Integer pixel coordinates specifying the corners of the serial prescan (top-row, bottom-row,
            left-column, right-column).
        serial_overscan
            Integer pixel coordinates specifying the corners of the serial overscan (top-row, bottom-row,
            left-column, right-column).
        """
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

    def mask_2d_from(self, mask: aa.Mask2D, rows: Tuple[int, int]) -> Mask2D:
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
        return Mask2D(
            mask=np.concatenate(calibration_masks, axis=0),
            pixel_scales=mask.pixel_scales,
        )

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
        [f#ff#f#f#f] = signal region (FPR) (0 / 1 indicate the region index)
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

        mask_2d = self.mask_2d_from(mask=array.mask, rows=rows)

        return aa.Array2D.manual_mask(
            array=new_array,
            mask=mask_2d,
            header=array.header,
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

    def imaging_ci_from(self, imaging_ci: "ImagingCI", rows) -> "ImagingCI":
        """
        Returnss a function to extract a serial section for given rows
        """

        from autocti.charge_injection.imaging.imaging import ImagingCI

        cosmic_ray_map = (
            imaging_ci.layout.extract.serial_calibration.array_2d_from(
                array=imaging_ci.cosmic_ray_map, rows=rows
            )
            if imaging_ci.cosmic_ray_map is not None
            else None
        )

        if imaging_ci.noise_scaling_map_dict is not None:

            noise_scaling_map_dict = {
                key: imaging_ci.layout.extract.serial_calibration.array_2d_from(
                    array=noise_scaling_map, rows=rows
                )
                for key, noise_scaling_map in imaging_ci.noise_scaling_map_dict.items()
            }

        else:

            noise_scaling_map_dict = None

        image = imaging_ci.layout.extract.serial_calibration.array_2d_from(
            array=imaging_ci.image, rows=rows
        )

        mask = self.mask_2d_from(mask=imaging_ci.mask, rows=rows)

        imaging_ci = ImagingCI(
            image=image,
            noise_map=imaging_ci.layout.extract.serial_calibration.array_2d_from(
                array=imaging_ci.noise_map, rows=rows
            ),
            pre_cti_data=imaging_ci.layout.extract.serial_calibration.array_2d_from(
                array=imaging_ci.pre_cti_data, rows=rows
            ),
            layout=imaging_ci.layout.extract.serial_calibration.extracted_layout_from(
                layout=imaging_ci.layout, new_shape_2d=image.shape, rows=rows
            ),
            cosmic_ray_map=cosmic_ray_map,
            noise_scaling_map_dict=noise_scaling_map_dict,
        )

        return imaging_ci.apply_mask(mask=mask)
