from copy import deepcopy

import numpy as np
from autoarray.structures import abstract_structure
from autoarray.structures.arrays import abstract_array
from autocti.charge_injection import ci_mask
from autocti.mask.mask import Mask2D
from autoarray.structures.frames import abstract_frame
from autoarray.instruments import euclid
from autoarray.util import array_util, frame_util


class AbstractCIFrame(abstract_frame.AbstractFrame):
    def __new__(
        cls,
        array,
        mask,
        ci_pattern,
        original_roe_corner=(1, 0),
        exposure_info=None,
        scans=None,
    ):
        """
        Class which represents the CCD quadrant of a charge injection image (e.g. the location of the parallel and
        serial front edge, trails).

        frame_geometry : CIFrame.CIQuadGeometry
            The quadrant geometry of the image, defining where the parallel / serial overscans are and
            therefore the direction of clocking and rotations before input into the cti algorithm.
        ci_pattern : CIPattern.CIPattern
            The charge injection ci_pattern (scans, normalization, etc.) of the charge injection image.
        """

        if type(array) is list:
            array = np.asarray(array)

        array[mask == True] = 0.0

        obj = array.view(cls)
        obj.mask = mask
        obj.store_in_1d = False
        obj.zoom_for_plot = False
        obj.original_roe_corner = original_roe_corner
        obj.exposure_info = exposure_info
        obj.scans = scans or abstract_frame.Scans()
        obj.ci_pattern = ci_pattern

        return obj

    def __array_finalize__(self, obj):

        super(AbstractCIFrame, self).__array_finalize__(obj)

        if isinstance(obj, AbstractCIFrame):
            if hasattr(obj, "mask"):
                self.mask = obj.mask
            if hasattr(obj, "original_roe_corner"):
                self.original_roe_corner = obj.original_roe_corner
            if hasattr(obj, "ci_pattern"):
                self.ci_pattern = obj.ci_pattern
            if hasattr(obj, "scans"):
                self.scans = obj.scans

    def _new_structure(self, array, mask, store_in_1d):
        return self.__class__(
            array=array,
            mask=mask,
            ci_pattern=self.ci_pattern,
            original_roe_corner=self.original_roe_corner,
            scans=self.scans,
            exposure_info=self.exposure_info,
        )

    @property
    def ci_regions_frame(self):
        """Extract an arrays of all of the charge-injection scans from a charge injection ci_frame.

        The diagram below illustrates the arrays that is extracted from a ci_frame:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
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

        The extracted ci_frame keeps just the charge injection region and replaces all other values with 0s:

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

        new_array = self.copy() * 0.0

        for region in self.ci_pattern.regions:
            new_array[region.slice] += self[region.slice]

        return new_array

    @property
    def non_ci_regions_frame(self):
        """Extract an arrays of all of the parallel trails following the charge-injection scans from a charge
        injection ci_frame.

        The diagram below illustrates the arrays that is extracted from a ci_frame:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
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

        The extracted ci_frame keeps just the trails following all charge injection scans and replaces all other
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

        non_ci_regions_array = self.copy()

        for region in self.ci_pattern.regions:
            non_ci_regions_array[region.slice] = 0.0

        return non_ci_regions_array

    @property
    def parallel_non_ci_regions_frame(self):
        """Extract an arrays of all of the parallel trails following the charge-injection scans from a charge
        injection ci_frame.

        The diagram below illustrates the arrays that is extracted from a ci_frame:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
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

        The extracted ci_frame keeps just the trails following all charge injection scans and replaces all other
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

        parallel_frame = self.non_ci_regions_frame

        parallel_frame[self.scans.serial_prescan.slice] = 0.0
        parallel_frame[self.scans.serial_overscan.slice] = 0.0

        return parallel_frame

    def parallel_edges_and_trails_frame(self, front_edge_rows=None, trails_rows=None):
        """Extract an arrays of all of the parallel front edges and trails of each the charge-injection scans from
        a charge injection ci_frame.

        One can specify the range of rows that are extracted, for example:

        front_edge_rows = (0, 1) will extract just the leading front edge row.
        front_edge_rows = (0, 2) will extract the leading two front edge rows.
        trails_rows = (0, 1) will extract the first row of trails closest to the charge injection region.

        The diagram below illustrates the arrays that is extracted from a ci_frame for front_edge_rows=(0,1) and
        trails_rows=(0,1):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
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

        The extracted ci_frame keeps just the leading edges and trails following all charge injection scans and
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
        array
        front_edge_rows : (int, int)
            The row indexes to extract the front edge between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows).
        trails_rows : (int, int)
            The row indexes to extract the trails between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows)
        """
        new_frame = self.copy() * 0.0

        if front_edge_rows is not None:

            front_regions = list(
                map(
                    lambda ci_region: self.parallel_front_edge_of_region(
                        ci_region, front_edge_rows
                    ),
                    self.ci_pattern.regions,
                )
            )

            front_edges = self.parallel_front_edge_arrays(rows=front_edge_rows)

            for i, region in enumerate(front_regions):
                new_frame[region.y0 : region.y1, region.x0 : region.x1] += front_edges[
                    i
                ]

        if trails_rows is not None:

            trails_regions = list(
                map(
                    lambda ci_region: self.parallel_trails_of_region(
                        ci_region, trails_rows
                    ),
                    self.ci_pattern.regions,
                )
            )

            trails = self.parallel_trails_arrays(rows=trails_rows)

            for i, region in enumerate(trails_regions):
                new_frame[region.y0 : region.y1, region.x0 : region.x1] += trails[i]

        return new_frame

    def parallel_calibration_frame_from_columns(self, columns):
        """Extract an parallel calibration array from a charge injection ci_frame, where this arrays is a sub-set of
        the ci_frame which be used for just parallel calibration. Specifically, this ci_frame is a specified number
        of columns closest to the read-out electronics.

        The diagram below illustrates the arrays that is extracted from a ci_frame with columns=(0, 3):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
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

        The extracted ci_frame keeps just the trails following all charge injection scans and replaces all other
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
        extraction_region = self.parallel_side_nearest_read_out_region(
            region=self.ci_pattern.regions[0], columns=columns
        )
        return CIFrame.extracted_ci_frame_from_ci_frame_and_extraction_region(
            ci_frame=self, extraction_region=extraction_region
        )

    def parallel_calibration_mask_from_mask_and_columns(self, mask, columns):
        extraction_region = self.parallel_side_nearest_read_out_region(
            region=self.ci_pattern.regions[0], columns=columns
        )
        return ci_mask.CIMask(
            mask=mask[extraction_region.slice], pixel_scales=mask.pixel_scales
        )

    @property
    def serial_trails_frame(self):
        """Extract an arrays of all of the serial trails in the serial overscan region, that are to the side of a
        charge-injection scans from a charge injection ci_frame.

        The diagram below illustrates the arrays that is extracted from a ci_frame:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
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

        The extracted ci_frame keeps just the trails following all charge injection scans and replaces all other
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
        array = self.serial_edges_and_trails_frame(
            trails_columns=(0, self.scans.serial_overscan.total_columns)
        )
        return array

    @property
    def serial_overscan_no_trails_frame(self):
        """Extract an arrays of all of the scans of the serial overscan that don't contain trails from a
        charge injection region (i.e. are not to the side of one).

        The diagram below illustrates the arrays that is extracted from a ci_frame:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
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

        The extracted ci_frame keeps just the trails following all charge injection scans and replaces all other
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
        new_array = self.copy() * 0.0
        overscan_slice = self.scans.serial_overscan.slice

        new_array[overscan_slice] = self[overscan_slice]

        trails_regions = list(
            map(
                lambda ci_region: self.serial_trails_of_region(
                    ci_region, (0, self.scans.serial_overscan.total_columns)
                ),
                self.ci_pattern.regions,
            )
        )

        for region in trails_regions:
            new_array[region.slice] = 0

        return new_array

    def serial_edges_and_trails_frame(
        self, front_edge_columns=None, trails_columns=None
    ):
        """Extract an arrays of all of the serial front edges and trails of each the charge-injection scans from
        a charge injection ci_frame.

        One can specify the range of columns that are extracted, for example:

        front_edge_columns = (0, 1) will extract just the leading front edge column.
        front_edge_columns = (0, 2) will extract the leading two front edge columns.
        trails_columns = (0, 1) will extract the first column of trails closest to the charge injection region.

        The diagram below illustrates the arrays that is extracted from a ci_frame for front_edge_columns=(0,2) and
        trails_columns=(0,2):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
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

        The extracted ci_frame keeps just the leading edge and trails following all charge injection scans and
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
        front_edge_columns : (int, int)
            The column indexes to extract the front edge between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd rows)
        trails_columns : (int, int)
            The column indexes to extract the trails between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd rows)
        """
        new_array = self.copy() * 0.0

        if front_edge_columns is not None:

            front_regions = list(
                map(
                    lambda ci_region: self.serial_front_edge_of_region(
                        ci_region, front_edge_columns
                    ),
                    self.ci_pattern.regions,
                )
            )

            front_edges = self.serial_front_edge_arrays(columns=front_edge_columns)

            for i, region in enumerate(front_regions):
                new_array[region.y0 : region.y1, region.x0 : region.x1] += front_edges[
                    i
                ]

        if trails_columns is not None:

            trails_regions = list(
                map(
                    lambda ci_region: self.serial_trails_of_region(
                        ci_region, trails_columns
                    ),
                    self.ci_pattern.regions,
                )
            )

            trails = self.serial_trails_arrays(columns=trails_columns)

            for i, region in enumerate(trails_regions):
                new_array[region.y0 : region.y1, region.x0 : region.x1] += trails[i]

        return new_array

    def serial_calibration_frame_from_rows(self, rows):
        """Extract a serial calibration array from a charge injection ci_frame, where this arrays is a sub-set of the
        ci_frame which can be used for serial-only calibration. Specifically, this ci_frame is all charge injection
        scans and their serial over-scan trails.

        The diagram below illustrates the arrays that is extracted from a ci_frame with column=5:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
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

        The extracted ci_frame keeps just the trails following all charge injection scans and replaces all other
        values with 0s:

        |                                      |
        |      [cccccccccccccccc][tst]         | Direction
        P      [cccccccccccccccc][tst]         | of
        |      [cccccccccccccccc][tst]         | clocking
               [cccccccccccccccc][tst]         |

        []     [=====================]
               <---------S----------
        """
        calibration_images = self.serial_calibration_sub_arrays
        calibration_images = list(
            map(lambda image: image[rows[0] : rows[1], :], calibration_images)
        )

        array = np.concatenate(calibration_images, axis=0)

        # TODO : can we generalize this method for multiple extracts? Feels too complicated so just doing it for this
        # TODO : specific case for now.

        serial_prescan = (
            (
                0,
                array.shape[0],
                self.scans.serial_prescan[2],
                self.scans.serial_prescan[3],
            )
            if self.scans.serial_prescan is not None
            else None
        )
        serial_overscan = (
            (
                0,
                array.shape[0],
                self.scans.serial_overscan[2],
                self.scans.serial_overscan[3],
            )
            if self.scans.serial_overscan is not None
            else None
        )

        x0 = self.ci_pattern.regions[0][2]
        x1 = self.ci_pattern.regions[0][3]
        offset = 0
        new_ci_pattern_regions = []
        for region in self.ci_pattern.regions:
            ysize = rows[1] - rows[0]
            new_ci_pattern_regions.append((offset, offset + ysize, x0, x1))
            offset += ysize

        new_ci_pattern = deepcopy(self.ci_pattern)
        new_ci_pattern.regions = new_ci_pattern_regions

        return CIFrame.manual(
            array=array,
            ci_pattern=new_ci_pattern,
            roe_corner=self.original_roe_corner,
            scans=abstract_frame.Scans(
                parallel_overscan=None,
                serial_prescan=serial_prescan,
                serial_overscan=serial_overscan,
            ),
            pixel_scales=self.pixel_scales,
        )

    def serial_calibration_mask_from_mask_and_rows(self, mask, rows):
        """Extract a serial calibration array from a charge injection ci_frame, where this arrays is a sub-set of the
        ci_frame which can be used for serial-only calibration. Specifically, this ci_frame is all charge injection
        scans and their serial over-scan trails.

        The diagram below illustrates the arrays that is extracted from a ci_frame with column=5:

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan    [cccccccccc] = charge injection region
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

        The extracted ci_frame keeps just the trails following all charge injection scans and replaces all other
        values with 0s:

        |                                      |
        |      [cccccccccccccccc][tst]         | Direction
        P      [cccccccccccccccc][tst]         | of
        |      [cccccccccccccccc][tst]         | clocking
               [cccccccccccccccc][tst]         |

        []     [=====================]
               <---------S----------
        """

        calibration_regions = list(
            map(
                lambda ci_region: self.serial_entire_rows_of_region(region=ci_region),
                self.ci_pattern.regions,
            )
        )
        calibration_masks = list(
            map(lambda region: mask[region.slice], calibration_regions)
        )

        calibration_masks = list(
            map(lambda mask: mask[rows[0] : rows[1], :], calibration_masks)
        )
        return ci_mask.CIMask(
            mask=np.concatenate(calibration_masks, axis=0),
            pixel_scales=mask.pixel_scales,
        )

    @property
    def serial_calibration_sub_arrays(self):
        """Extract each charge injection region image for the serial calibration arrays above."""

        calibration_regions = list(
            map(
                lambda ci_region: self.serial_entire_rows_of_region(region=ci_region),
                self.ci_pattern.regions,
            )
        )
        return list(map(lambda region: self[region.slice], calibration_regions))

    def parallel_front_edge_line_binned_over_columns(self, rows=None):
        front_stacked_array = self.parallel_front_edge_stacked_array(rows=rows)
        return np.ma.mean(np.ma.asarray(front_stacked_array), axis=1)

    def parallel_front_edge_stacked_array(self, rows=None):
        front_arrays = self.parallel_front_edge_arrays(rows=rows)
        return np.ma.mean(np.ma.asarray(front_arrays), axis=0)

    def parallel_front_edge_arrays(self, rows=None):
        """Extract a list of structures of the parallel front edge scans of a charge injection ci_frame.

        The diagram below illustrates the arrays that is extracted from a ci_frame for rows=(0, 1):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicates ci_region index)
        [xxxxxxxxxx]                [tttttttttt] = parallel / serial charge injection region trail

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][ttttttttttttttttttttt][sss]
          [...][c1c1cc1c1cc1cc1ccc1cc][sss]
        | [...][1c1c1cc1c1cc1ccc1cc1][sss]    |
        | [...][ttttttttttttttttttttt][sss]    | Direction
        P [...][ttttttttttttttttttttt][sss]    | of
        | [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
          [...][cc0ccc0cccc0cccc0cccc][sss]    |

        []     [=====================]
               <---------S----------

        The extracted ci_frame keeps just the front edges of all charge injection scans.

        list index 0:

        [c0c0c0cc0c0c0c0c0c0c0]

        list index 1:

        [1c1c1c1c1c1c1c1c1c1c1]

        Parameters
        ------------
        array
        rows : (int, int)
            The row indexes to extract the front edge between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows)
        """
        front_regions = self.parallel_front_edge_regions(rows=rows)
        front_arrays = list(map(lambda region: self[region.slice], front_regions))
        front_masks = list(map(lambda region: self.mask[region.slice], front_regions))
        front_arrays = list(
            map(
                lambda front_array, front_mask: np.ma.array(
                    front_array, mask=front_mask
                ),
                front_arrays,
                front_masks,
            )
        )
        return front_arrays

    def parallel_front_edge_regions(self, rows=None):
        """Calculate a list of the parallel front edge scans of a charge injection ci_frame.

        The diagram below illustrates the region that calculaed from a ci_frame for rows=(0, 1):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicates ci_region index)
        [xxxxxxxxxx]                [tttttttttt] = parallel / serial charge injection region trail

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][ttttttttttttttttttttt][sss]
          [...][c1c1cc1c1cc1cc1ccc1cc][sss]
        | [...][1c1c1cc1c1cc1ccc1cc1][sss]    |
        | [...][ttttttttttttttttttttt][sss]    | Direction
        P [...][ttttttttttttttttttttt][sss]    | of
        | [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
          [...][cc0ccc0cccc0cccc0cccc][sss]    |

        []     [=====================]
               <---------S----------

        The extracted ci_frame keeps just the front edges of all charge injection scans.

        list index 0:

        [0, 1, 3, 21] (serial prescan is 3 pixels)

        list index 1:

        [3, 4, 3, 21] (serial prescan is 3 pixels)

        Parameters
        ------------
        arrays
        rows : (int, int)
            The row indexes to extract the front edge between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows)
        """
        if rows is None:
            rows = (0, self.ci_pattern.total_rows_min)
        return list(
            map(
                lambda ci_region: self.parallel_front_edge_of_region(
                    region=ci_region, rows=rows
                ),
                self.ci_pattern.regions,
            )
        )

    def parallel_trails_line_binned_over_columns(self, rows=None):
        trails_stacked_array = self.parallel_trails_stacked_array(rows=rows)
        return np.ma.mean(np.ma.asarray(trails_stacked_array), axis=1)

    def parallel_trails_stacked_array(self, rows=None):
        trails_arrays = self.parallel_trails_arrays(rows=rows)
        return np.ma.mean(np.ma.asarray(trails_arrays), axis=0)

    def parallel_trails_arrays(self, rows=None):
        """Extract the parallel trails of a charge injection ci_frame.


        The diagram below illustrates the arrays that is extracted from a ci_frame for rows=(0, 1):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicates ci region index)
        [xxxxxxxxxx]
        [t#t#t#t#t#] = parallel / serial charge injection region trail (0 / 1 indicates ci region index)

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][t1t1t1t1t1t1t1t1t1t1t][sss]
          [...][c1c1cc1c1cc1cc1ccc1cc][sss]
        | [...][1c1c1cc1c1cc1ccc1cc1][sss]    |
        | [...][t0t0t0t0t0t0t0t0t0t0t][sss]    | Direction
        P [...][0t0t0t0t0t0t0t0t0t0t0][sss]    | of
        | [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
          [...][cc0ccc0cccc0cccc0cccc][sss]    |

        []     [=====================]
               <---------S----------

        The extracted ci_frame keeps just the trails following all charge injection scans:

        list index 0:

        [t0t0t0tt0t0t0t0t0t0t0]

        list index 1:

        [1t1t1t1t1t1t1t1t1t1t1]

        Parameters
        ------------
        array
        rows : (int, int)
            The row indexes to extract the trails between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows)
        """
        trails_regions = self.parallel_trails_regions(rows=rows)
        trails_arrays = list(map(lambda region: self[region.slice], trails_regions))
        trails_masks = list(map(lambda region: self.mask[region.slice], trails_regions))
        trails_arrays = list(
            map(
                lambda trails_array, front_mask: np.ma.array(
                    trails_array, mask=front_mask
                ),
                trails_arrays,
                trails_masks,
            )
        )
        return trails_arrays

    def parallel_trails_regions(self, rows=None):
        """
    Returns the parallel scans of a charge injection ci_frame.

        The diagram below illustrates the region that is calculated from a ci_frame for rows=(0, 1):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicates ci region index)
        [xxxxxxxxxx]
        [t#t#t#t#t#] = parallel / serial charge injection region trail (0 / 1 indicates ci region index)

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][t1t1t1t1t1t1t1t1t1t1t][sss]
          [...][c1c1cc1c1cc1cc1ccc1cc][sss]
        | [...][1c1c1cc1c1cc1ccc1cc1][sss]    |
        | [...][t0t0t0t0t0t0t0t0t0t0t][sss]    | Direction
        P [...][0t0t0t0t0t0t0t0t0t0t0][sss]    | of
        | [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
          [...][cc0ccc0cccc0cccc0cccc][sss]    |

        []     [=====================]
               <---------S----------

        The extracted ci_frame keeps just the trails following all charge injection scans:

        list index 0:

        [2, 4, 3, 21] (serial prescan is 3 pixels)

        list index 1:

        [6, 7, 3, 21] (serial prescan is 3 pixels)

        Parameters
        ------------
        arrays
        rows : (int, int)
            The row indexes to extract the trails between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows)
        """

        if rows is None:
            rows = (0, self.smallest_parallel_trails_rows_to_frame_edge)

        return list(
            map(
                lambda ci_region: self.parallel_trails_of_region(ci_region, rows),
                self.ci_pattern.regions,
            )
        )

    def serial_front_edge_line_binned_over_rows(self, columns=None):
        front_stacked_array = self.serial_front_edge_stacked_array(columns=columns)
        return np.ma.mean(np.ma.asarray(front_stacked_array), axis=0)

    def serial_front_edge_stacked_array(self, columns=None):
        front_arrays = self.serial_front_edge_arrays(columns=columns)
        return np.ma.mean(np.ma.asarray(front_arrays), axis=0)

    def serial_front_edge_arrays(self, columns=None):
        """Extract a list of the serial front edge structures of a charge injection ci_frame.

        The diagram below illustrates the arrays that is extracted from a ci_frame for columnss=(0, 3):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicates ci_region index)
        [xxxxxxxxxx]
        [tttttttttt] = parallel / serial charge injection region trail ((0 / 1 indicates ci_region index)

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][ttttttttttttttttttttt][sss]
          [...][c1c1cc1c1cc1cc1ccc1cc][sss]
        | [...][1c1c1cc1c1cc1ccc1cc1c][sss]    |
        | [...][ttttttttttttttttttttt][sss]    | Direction
        P [...][ttttttttttttttttttttt][sss]    | of
        | [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
          [...][cc0ccc0cccc0cccc0cccc][sss]    |

        []     [=====================]
               <---------S----------

        The extracted ci_frame keeps just the serial front edges of all charge injection scans.

        list index 0:

        [c0c0]

        list index 1:

        [1c1c]

        Parameters
        ------------
        array
        columns : (int, int)
            The column indexes to extract the front edge between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd
            columns)
        """
        front_regions = self.serial_front_edge_regions(columns=columns)
        front_arrays = list(map(lambda region: self[region.slice], front_regions))
        front_masks = list(map(lambda region: self.mask[region.slice], front_regions))
        front_arrays = list(
            map(
                lambda front_array, front_mask: np.ma.array(
                    front_array, mask=front_mask
                ),
                front_arrays,
                front_masks,
            )
        )
        return front_arrays

    def serial_front_edge_regions(self, columns=None):
        """
    Returns a list of the serial front edges scans of a charge injection ci_frame.

        The diagram below illustrates the region that is calculated from a ci_frame for columns=(0, 4):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicates ci_region index)
        [xxxxxxxxxx]
        [tttttttttt] = parallel / serial charge injection region trail ((0 / 1 indicates ci_region index)

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][ttttttttttttttttttttt][sss]
          [...][c1c1cc1c1cc1cc1ccc1cc][sss]
        | [...][1c1c1cc1c1cc1ccc1cc1c][sss]    |
        | [...][ttttttttttttttttttttt][sss]    | Direction
        P [...][ttttttttttttttttttttt][sss]    | of
        | [...][0ccc0cccc0cccc0cccc0c][sss]    | clocking
          [...][cc0ccc0cccc0cccc0cccc][sss]    |

        []     [=====================]
               <---------S----------

        The extracted ci_frame keeps just the serial front edges of all charge injection scans.

        list index 0:

        [0, 2, 3, 21] (serial prescan is 3 pixels)

        list index 1:

        [4, 6, 3, 21] (serial prescan is 3 pixels)

        Parameters
        ------------
        arrays
        columns : (int, int)
            The column indexes to extract the front edge between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd
            columns)
        """
        if columns is None:
            columns = (0, self.ci_pattern.total_columns_min)
        return list(
            map(
                lambda ci_region: self.serial_front_edge_of_region(ci_region, columns),
                self.ci_pattern.regions,
            )
        )

    def serial_trails_line_binned_over_rows(self, columns=None):
        trails_stacked_array = self.serial_trails_stacked_array(columns=columns)
        return np.ma.mean(np.ma.asarray(trails_stacked_array), axis=0)

    def serial_trails_stacked_array(self, columns=None):
        front_arrays = self.serial_trails_arrays(columns=columns)
        return np.ma.mean(np.ma.asarray(front_arrays), axis=0)

    def serial_trails_arrays(self, columns=None):
        """Extract a list of the serial trails of a charge injection ci_frame.

        The diagram below illustrates the arrays that is extracted from a ci_frame for columnss=(0, 3):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicates ci_region index)
        [xxxxxxxxxx]
        [tttttttttt] = parallel / serial charge injection region trail ((0 / 1 indicates ci_region index)

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][ttttttttttttttttttttt][sss]
          [...][c1c1cc1c1cc1cc1ccc1cc][st1]
        | [...][1c1c1cc1c1cc1ccc1cc1c][ts0]    |
        | [...][ttttttttttttttttttttt][sss]    | Direction
        P [...][ttttttttttttttttttttt][sss]    | of
        | [...][0ccc0cccc0cccc0cccc0c][st1]    | clocking
          [...][cc0ccc0cccc0cccc0cccc][ts0]    |

        []     [=====================]
               <---------S----------

        The extracted ci_frame keeps just the serial front edges of all charge injection scans.

        list index 0:

        [st0]

        list index 1:

        [st1]

        Parameters
        ------------
        array
        columns : (int, int)
            The column indexes to extract the trails between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd columns)
        """
        trails_regions = self.serial_trails_regions(columns=columns)
        trails_arrays = list(map(lambda region: self[region.slice], trails_regions))
        trails_masks = list(map(lambda region: self.mask[region.slice], trails_regions))
        trails_arrays = list(
            map(
                lambda trails_array, front_mask: np.ma.array(
                    trails_array, mask=front_mask
                ),
                trails_arrays,
                trails_masks,
            )
        )
        return trails_arrays

    def serial_trails_regions(self, columns=None):
        """
    Returns a list of the serial trails scans of a charge injection ci_frame.

        The diagram below illustrates the region is calculated from a ci_frame for columnss=(0, 4):

        ---KEY---
        ---------

        [] = read-out electronics   [==========] = read-out register

        [xxxxxxxxxx]                [..........] = serial prescan       [ssssssssss] = serial overscan
        [xxxxxxxxxx] = CCD panel    [pppppppppp] = parallel overscan
        [c#cc#c#c#c] = charge injection region (0 / 1 indicates ci_region index)
        [xxxxxxxxxx]
        [tttttttttt] = parallel / serial charge injection region trail ((0 / 1 indicates ci_region index)

        P = Parallel Direction      S = Serial Direction

               [ppppppppppppppppppppp]
               [ppppppppppppppppppppp]
          [...][ttttttttttttttttttttt][sss]
          [...][c1c1cc1c1cc1cc1ccc1cc][st1]
        | [...][1c1c1cc1c1cc1ccc1cc1c][ts0]    |
        | [...][ttttttttttttttttttttt][sss]    | Direction
        P [...][ttttttttttttttttttttt][sss]    | of
        | [...][0ccc0cccc0cccc0cccc0c][st1]    | clocking
          [...][cc0ccc0cccc0cccc0cccc][ts0]    |

        []     [=====================]
               <---------S----------

        The extracted ci_frame keeps just the serial front edges of all charge injection scans.

        list index 0:

        [0, 2, 22, 225 (serial prescan is 3 pixels)

        list index 1:

        [4, 6, 22, 25] (serial prescan is 3 pixels)

        Parameters
        ------------
        arrays
        columns : (int, int)
            The column indexes to extract the trails between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd columns)
        """
        if columns is None:
            columns = (0, self.scans.serial_trails_columns)
        return list(
            map(
                lambda ci_region: self.serial_trails_of_region(ci_region, columns),
                self.ci_pattern.regions,
            )
        )

    @property
    def parallel_serial_calibration_frame(self):
        return self

    @property
    def smallest_parallel_trails_rows_to_frame_edge(self):

        rows_between_regions = self.ci_pattern.rows_between_regions
        rows_between_regions.append(self.parallel_trail_size_to_frame_edge)
        return np.min(rows_between_regions)

    @property
    def parallel_trail_size_to_frame_edge(self):

        return self.shape_2d[0] - np.max(
            [region.y1 for region in self.ci_pattern.regions]
        )


class CIFrame(AbstractCIFrame):
    @classmethod
    def manual(
        cls,
        array,
        pixel_scales,
        ci_pattern,
        roe_corner=(1, 0),
        exposure_info=None,
        scans=None,
    ):
        """Abstract class for the geometry of a CTI Image.

        A FrameArray is stored as a 2D ndarrays. When this immage is passed to arctic, clocking goes towards
        the 'top' of the ndarrays (e.g. towards row 0). Trails therefore appear towards the 'bottom' of the arrays
        (e.g. the final row).

        Arctic has no in-built functionality for changing the direction of clocking depending on the input
        configuration file. Therefore, image rotations are handled before arctic is called, using the functions
        defined in this class (and its children). These routines define how an image is rotated before parallel
        and serial clocking and how to reorient the image back to its original orientation after clocking is performed.

        Currently, only four geometries are available, which are specific to Euclid (and documented in the
        *QuadGeometryEuclid* class).

        Parameters
        -----------
        parallel_overscan : frame.Region
            The parallel overscan region of the ci_frame.
        serial_prescan : frame.Region
            The serial prescan region of the ci_frame.
        serial_overscan : frame.Region
            The serial overscan region of the ci_frame.
        """

        if array is None:
            return None

        array = abstract_array.convert_array(array=array)

        pixel_scales = abstract_structure.convert_pixel_scales(
            pixel_scales=pixel_scales
        )

        mask = Mask2D.unmasked(shape_2d=array.shape, pixel_scales=pixel_scales)

        scans = abstract_frame.Scans.rotated_from_roe_corner(
            roe_corner=roe_corner, shape_2d=array.shape, scans=scans
        )

        return CIFrame(
            array=frame_util.rotate_array_from_roe_corner(
                array=array, roe_corner=roe_corner
            ),
            mask=mask,
            ci_pattern=frame_util.rotate_ci_pattern_from_roe_corner(
                ci_pattern=ci_pattern, shape_2d=array.shape, roe_corner=roe_corner
            ),
            original_roe_corner=roe_corner,
            exposure_info=exposure_info,
            scans=scans,
        )

    @classmethod
    def manual_mask(
        cls, array, mask, ci_pattern, roe_corner=(1, 0), exposure_info=None, scans=None
    ):
        """Abstract class for the geometry of a CTI Image.

        A FrameArray is stored as a 2D ndarrays. When this immage is passed to arctic, clocking goes towards
        the 'top' of the ndarrays (e.g. towards row 0). Trails therefore appear towards the 'bottom' of the arrays
        (e.g. the final row).

        Arctic has no in-built functionality for changing the direction of clocking depending on the input
        configuration file. Therefore, image rotations are handled before arctic is called, using the functions
        defined in this class (and its children). These routines define how an image is rotated before parallel
        and serial clocking and how to reorient the image back to its original orientation after clocking is performed.

        Currently, only four geometries are available, which are specific to Euclid (and documented in the
        *QuadGeometryEuclid* class).

        Parameters
        -----------
        parallel_overscan : Region
            The parallel overscan region of the ci_frame.
        serial_prescan : Region
            The serial prescan region of the ci_frame.
        serial_overscan : Region
            The serial overscan region of the ci_frame.
        """

        array = abstract_array.convert_array(array=array)

        array = frame_util.rotate_array_from_roe_corner(
            array=array, roe_corner=roe_corner
        )
        mask = frame_util.rotate_array_from_roe_corner(
            array=mask, roe_corner=roe_corner
        )

        array[mask == True] = 0.0

        scans = abstract_frame.Scans.rotated_from_roe_corner(
            roe_corner=roe_corner, shape_2d=array.shape, scans=scans
        )

        return CIFrame(
            array=array,
            mask=mask,
            ci_pattern=frame_util.rotate_ci_pattern_from_roe_corner(
                ci_pattern=ci_pattern, shape_2d=array.shape, roe_corner=roe_corner
            ),
            original_roe_corner=roe_corner,
            exposure_info=exposure_info,
            scans=scans,
        )

    @classmethod
    def full(
        cls,
        fill_value,
        shape_2d,
        pixel_scales,
        ci_pattern,
        roe_corner=(1, 0),
        exposure_info=None,
        scans=None,
    ):

        return cls.manual(
            array=np.full(fill_value=fill_value, shape=shape_2d),
            pixel_scales=pixel_scales,
            ci_pattern=ci_pattern,
            roe_corner=roe_corner,
            exposure_info=exposure_info,
            scans=scans,
        )

    @classmethod
    def ones(
        cls,
        shape_2d,
        pixel_scales,
        ci_pattern,
        roe_corner=(1, 0),
        exposure_info=None,
        scans=None,
    ):
        return cls.full(
            fill_value=1.0,
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            ci_pattern=ci_pattern,
            roe_corner=roe_corner,
            exposure_info=exposure_info,
            scans=scans,
        )

    @classmethod
    def zeros(
        cls,
        shape_2d,
        pixel_scales,
        ci_pattern,
        roe_corner=(1, 0),
        exposure_info=None,
        scans=None,
    ):
        return cls.full(
            fill_value=0.0,
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            ci_pattern=ci_pattern,
            roe_corner=roe_corner,
            exposure_info=exposure_info,
            scans=scans,
        )

    @classmethod
    def extracted_ci_frame_from_ci_frame_and_extraction_region(
        cls, ci_frame, extraction_region
    ):

        scans = abstract_frame.Scans.after_extraction(
            frame=ci_frame, extraction_region=extraction_region
        )

        return cls.manual(
            array=ci_frame[extraction_region.slice],
            ci_pattern=ci_frame.ci_pattern.with_extracted_regions(
                extraction_region=extraction_region
            ),
            exposure_info=ci_frame.exposure_info,
            scans=scans,
            roe_corner=ci_frame.original_roe_corner,
            pixel_scales=ci_frame.pixel_scales,
        )

    @classmethod
    def from_fits(
        cls,
        file_path,
        hdu,
        pixel_scales,
        ci_pattern,
        roe_corner=(1, 0),
        exposure_info=None,
        scans=None,
    ):
        """Load the image ci_data from a fits file.

        Params
        ----------
        path : str
            The path to the ci_data
        filename : str
            The file name of the fits image ci_data.
        hdu : int
            The HDU number in the fits file containing the image ci_data.
        frame_geometry : FrameArray.FrameGeometry
            The geometry of the ci_frame, defining the direction of parallel and serial clocking and the \
            locations of different scans of the CCD (overscans, prescan, etc.)
        """

        pixel_scales = abstract_structure.convert_pixel_scales(
            pixel_scales=pixel_scales
        )

        array = array_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu)

        return CIFrame.manual(
            array=array,
            ci_pattern=ci_pattern,
            roe_corner=roe_corner,
            exposure_info=exposure_info,
            scans=scans,
            pixel_scales=pixel_scales,
        )

    @classmethod
    def from_ci_frame(cls, ci_frame, mask):
        return CIFrame(
            array=ci_frame,
            mask=mask,
            ci_pattern=ci_frame.ci_pattern,
            original_roe_corner=ci_frame.original_roe_corner,
            exposure_info=ci_frame.exposure_info,
            scans=abstract_frame.Scans.from_frame(frame=ci_frame),
        )


class CIFrameEuclid(CIFrame):
    @classmethod
    def from_ccd_and_quadrant_id(
        cls,
        array,
        ccd_id,
        quadrant_id,
        ci_pattern,
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=51,
        serial_overscan_size=29,
        parallel_overscan_size=20,
    ):
        """Before reading this docstring, read the docstring for the __init__function above.

        In the Euclid FPA, the quadrant id ('E', 'F', 'G', 'H') depends on whether the CCD is located
        on the left side (rows 1-3) or right side (rows 4-6) of the FPA:

        LEFT SIDE ROWS 1-2-3
        --------------------

         <--------S-----------   ---------S----------->
        [] [========= 2 =========] [========= 3 =========] []          |
        /    [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  /          |
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | Direction arctic
        P   [xxxxxxxxx H xxxxxxxxx] [xxxxxxxxx G xxxxxxxxx]  P         | clocks an image
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | without any rotation
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | (e.g. towards row 0
                                                                       | of the ndarrays)
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
        P   [xxxxxxxxx E xxxxxxxxx] [xxxxxxxxx F xxxxxxxxx] P          |
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
            [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]            |

        [] [========= 0 =========] [========= 1 =========] []
            <---------S----------   ----------S----------->


        RIGHT SIDE ROWS 4-5-6
        ---------------------

         <--------S-----------   ---------S----------->
        [] [========= 2 =========] [========= 3 =========] []          |
        /    [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  /          |
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | Direction arctic
        P   [xxxxxxxxx F xxxxxxxxx] [xxxxxxxxx E xxxxxxxxx]  P         | clocks an image
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | without any rotation
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | (e.g. towards row 0
                                                                       | of the ndarrays)
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
        P   [xxxxxxxxx G xxxxxxxxx] [xxxxxxxxx H xxxxxxxxx] P          |
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
            [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]            |

        [] [========= 0 =========] [========= 1 =========] []
            <---------S----------   ----------S----------->

        Therefore, to setup a quadrant image with the correct frame_geometry using its CCD id (from which
        we can extract its row number) and quadrant id, we need to first determine if the CCD is on the left / right
        side and then use its quadrant id ('E', 'F', 'G' or 'H') to pick the correct quadrant.
        """

        row_index = ccd_id[-1]

        if (row_index in "123") and (quadrant_id == "E"):
            return CIFrameEuclid.bottom_left(
                array=array,
                ci_pattern=ci_pattern,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "123") and (quadrant_id == "F"):
            return CIFrameEuclid.bottom_right(
                array=array,
                ci_pattern=ci_pattern,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "123") and (quadrant_id == "G"):
            return CIFrameEuclid.top_right(
                array=array,
                ci_pattern=ci_pattern,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "123") and (quadrant_id == "H"):
            return CIFrameEuclid.top_left(
                array=array,
                ci_pattern=ci_pattern,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "456") and (quadrant_id == "E"):
            return CIFrameEuclid.top_right(
                array=array,
                ci_pattern=ci_pattern,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "456") and (quadrant_id == "F"):
            return CIFrameEuclid.top_left(
                array=array,
                ci_pattern=ci_pattern,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "456") and (quadrant_id == "G"):
            return CIFrameEuclid.bottom_left(
                array=array,
                ci_pattern=ci_pattern,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "456") and (quadrant_id == "H"):
            return CIFrameEuclid.bottom_right(
                array=array,
                ci_pattern=ci_pattern,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )

    @classmethod
    def top_left(
        cls,
        array,
        ci_pattern,
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=51,
        serial_overscan_size=29,
        parallel_overscan_size=20,
    ):

        scans = euclid.ScansEuclid.top_left(
            parallel_size=parallel_size,
            serial_size=serial_size,
            serial_prescan_size=serial_prescan_size,
            serial_overscan_size=serial_overscan_size,
            parallel_overscan_size=parallel_overscan_size,
        )

        return CIFrame.manual(
            array=array,
            pixel_scales=0.1,
            ci_pattern=ci_pattern,
            roe_corner=(0, 0),
            scans=scans,
        )

    @classmethod
    def top_right(
        cls,
        array,
        ci_pattern,
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=51,
        serial_overscan_size=29,
        parallel_overscan_size=20,
    ):

        scans = euclid.ScansEuclid.top_right(
            parallel_size=parallel_size,
            serial_size=serial_size,
            serial_prescan_size=serial_prescan_size,
            serial_overscan_size=serial_overscan_size,
            parallel_overscan_size=parallel_overscan_size,
        )

        return CIFrame.manual(
            array=array,
            pixel_scales=0.1,
            ci_pattern=ci_pattern,
            roe_corner=(0, 1),
            scans=scans,
        )

    @classmethod
    def bottom_left(
        cls,
        array,
        ci_pattern,
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=51,
        serial_overscan_size=29,
        parallel_overscan_size=20,
    ):

        scans = euclid.ScansEuclid.bottom_left(
            parallel_size=parallel_size,
            serial_size=serial_size,
            serial_prescan_size=serial_prescan_size,
            serial_overscan_size=serial_overscan_size,
            parallel_overscan_size=parallel_overscan_size,
        )

        return CIFrame.manual(
            array=array,
            pixel_scales=0.1,
            ci_pattern=ci_pattern,
            roe_corner=(1, 0),
            scans=scans,
        )

    @classmethod
    def bottom_right(
        cls,
        array,
        ci_pattern,
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=51,
        serial_overscan_size=29,
        parallel_overscan_size=20,
    ):

        scans = euclid.ScansEuclid.bottom_right(
            parallel_size=parallel_size,
            serial_size=serial_size,
            serial_prescan_size=serial_prescan_size,
            serial_overscan_size=serial_overscan_size,
            parallel_overscan_size=parallel_overscan_size,
        )

        return CIFrame.manual(
            array=array,
            pixel_scales=0.1,
            ci_pattern=ci_pattern,
            roe_corner=(1, 1),
            scans=scans,
        )
