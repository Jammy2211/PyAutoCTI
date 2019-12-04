import numpy as np

from autocti.structures import frame


class CIFrame(frame.Frame):
    
    def __new__(cls, array, ci_pattern, corner=(0,0), parallel_overscan=None, serial_prescan=None, 
                 serial_overscan=None, pixel_scales=None):
        """
        Class which represents the CCD quadrant of a charge injection image (e.g. the location of the parallel and
        serial front edge, trails).

        frame_geometry : CIFrame.CIQuadGeometry
            The quadrant geometry of the image, defining where the parallel / serial overscans are and
            therefore the direction of clocking and rotations before input into the cti algorithm.
        ci_pattern : CIPattern.CIPattern
            The charge injection ci_pattern (regions, normalization, etc.) of the charge injection image.
        """
        
        obj = super(CIFrame, cls).__new__(cls=cls, array=array, corner=corner, 
                                          parallel_overscan=parallel_overscan, serial_prescan=serial_prescan, 
                                          serial_overscan=serial_overscan, pixel_scales=pixel_scales)
        
        obj.ci_pattern = ci_pattern
        
        return obj

    def ci_regions_from_array(self, array):
        """Extract an arrays of all of the charge-injection regions from a charge injection ci_frame.

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

        new_array = np.zeros(shape=array.shape)

        for region in self.ci_pattern.regions:
            new_array[region.slice] += array[region.slice]

        return new_array

    def parallel_non_ci_regions_frame_from_frame(self, array):
        """Extract an arrays of all of the parallel trails following the charge-injection regions from a charge
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

        The extracted ci_frame keeps just the trails following all charge injection regions and replaces all other
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

        parallel_array = np.zeros(array.shape)

        x0 = self.parallel_overscan.x0
        x1 = self.parallel_overscan.x1

        parallel_array[:, x0:x1] = array[:, x0:x1]

        for region in self.ci_pattern.regions:
            parallel_array[region.slice] = 0

        return parallel_array.copy()

    def parallel_edges_and_trails_frame_from_frame(
        self, array, front_edge_rows=None, trails_rows=None
    ):
        """Extract an arrays of all of the parallel front edges and trails of each the charge-injection regions from
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

        The extracted ci_frame keeps just the leading edges and trails following all charge injection regions and   
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
        new_array = np.zeros(array.shape)

        if front_edge_rows is not None:

            front_regions = list(
                map(
                    lambda ci_region: self.parallel_front_edge_region(
                        ci_region, front_edge_rows
                    ),
                    self.ci_pattern.regions,
                )
            )

            front_edges = self.parallel_front_edge_arrays_from_frame(
                array, front_edge_rows
            )

            for i, region in enumerate(front_regions):
                new_array[region.y0 : region.y1, region.x0 : region.x1] += front_edges[
                    i
                ]

        if trails_rows is not None:

            trails_regions = list(
                map(
                    lambda ci_region: self.parallel_trails_region(
                        ci_region, trails_rows
                    ),
                    self.ci_pattern.regions,
                )
            )

            trails = self.parallel_trails_arrays_from_frame(array, trails_rows)

            for i, region in enumerate(trails_regions):
                new_array[region.y0 : region.y1, region.x0 : region.x1] += trails[i]

        return new_array

    def parallel_calibration_section_for_columns(self, array, columns):
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

        The extracted ci_frame keeps just the trails following all charge injection regions and replaces all other
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
        calibration_region = self.parallel_side_nearest_read_out_region(
            self.ci_pattern.regions[0], array.shape, columns
        )
        array = array[calibration_region.slice]
        return array

    def serial_all_trails_frame_from_frame(self, array):
        """Extract an arrays of all of the serial trails in the serial overscan region, that are to the side of a
        charge-injection regions from a charge injection ci_frame.

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

        The extracted ci_frame keeps just the trails following all charge injection regions and replaces all other
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
        array = self.serial_edges_and_trails_frame_from_frame(
            array=array,
            trails_columns=(0, self.serial_overscan.total_columns),
        )
        return array

    def serial_overscan_above_trails_frame_from_frame(self, array):
        """Extract an arrays of all of the regions of the serial overscan that don't contain trails from a
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

        The extracted ci_frame keeps just the trails following all charge injection regions and replaces all other
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
        new_array = np.zeros(array.shape)
        overscan_slice = self.serial_overscan.slice

        new_array[overscan_slice] = array[overscan_slice]

        trails_regions = list(
            map(
                lambda ci_region: self.serial_trails_region(
                    ci_region, (0, self.serial_overscan.total_columns)
                ),
                self.ci_pattern.regions,
            )
        )

        for region in trails_regions:
            new_array[region.slice] = 0

        return new_array

    def serial_edges_and_trails_frame_from_frame(
        self, array, front_edge_columns=None, trails_columns=None
    ):
        """Extract an arrays of all of the serial front edges and trails of each the charge-injection regions from
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

        The extracted ci_frame keeps just the leading edge and trails following all charge injection regions and
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
        new_array = np.zeros(array.shape)

        if front_edge_columns is not None:

            front_regions = list(
                map(
                    lambda ci_region: self.serial_front_edge_region(
                        ci_region, front_edge_columns
                    ),
                    self.ci_pattern.regions,
                )
            )

            front_edges = self.serial_front_edge_arrays_from_frame(
                array, front_edge_columns
            )

            for i, region in enumerate(front_regions):
                new_array[region.y0 : region.y1, region.x0 : region.x1] += front_edges[
                    i
                ]

        if trails_columns is not None:

            trails_regions = list(
                map(
                    lambda ci_region: self.serial_trails_region(
                        ci_region, trails_columns
                    ),
                    self.ci_pattern.regions,
                )
            )

            trails = self.serial_trails_arrays_from_frame(array, trails_columns)

            for i, region in enumerate(trails_regions):
                new_array[region.y0 : region.y1, region.x0 : region.x1] += trails[i]

        return new_array

    def serial_calibration_section_for_rows(self, array, rows):
        """Extract a serial calibration array from a charge injection ci_frame, where this arrays is a sub-set of the
        ci_frame which can be used for serial-only calibration. Specifically, this ci_frame is all charge injection
        regions and their serial over-scan trails.

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

        The extracted ci_frame keeps just the trails following all charge injection regions and replaces all other
        values with 0s:

        |                                      |
        |      [cccccccccccccccc][tst]         | Direction
        P      [cccccccccccccccc][tst]         | of
        |      [cccccccccccccccc][tst]         | clocking
               [cccccccccccccccc][tst]         |
                                                 
        []     [=====================]
               <---------S----------
        """
        calibration_images = self.serial_calibration_sub_arrays_from_frame(array=array)
        calibration_images = list(
            map(lambda image: image[rows[0] : rows[1], :], calibration_images)
        )
        array = np.concatenate(calibration_images, axis=0)
        return array

    def serial_calibration_sub_arrays_from_frame(self, array):
        """Extract each charge injection region image for the serial calibration arrays above."""

        calibration_regions = list(
            map(
                lambda ci_region: self.serial_prescan_ci_region_and_trails(
                    ci_region=ci_region, image_shape=array.shape
                ),
                self.ci_pattern.regions,
            )
        )
        return list(map(lambda region: array[region.slice], calibration_regions))

    def parallel_front_edge_line_binned_over_columns_from_frame(
        self, array, rows=None, mask=None
    ):
        front_stacked_array = self.parallel_front_edge_stacked_array_from_frame(
            array=array, rows=rows, mask=mask
        )
        return np.ma.mean(np.ma.asarray(front_stacked_array), axis=1)

    def parallel_front_edge_stacked_array_from_frame(self, array, rows=None, mask=None):
        front_arrays = self.parallel_front_edge_arrays_from_frame(
            array=array, rows=rows, mask=mask
        )
        return np.ma.mean(np.ma.asarray(front_arrays), axis=0)

    def parallel_front_edge_arrays_from_frame(self, array, rows=None, mask=None):
        """Extract a list of structures of the parallel front edge regions of a charge injection ci_frame.

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

        The extracted ci_frame keeps just the front edges of all charge injection regions.

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
        front_regions = self.parallel_front_edge_regions_from_frame(rows=rows)
        front_arrays = list(map(lambda region: array[region.slice], front_regions))
        if mask is not None:
            front_masks = list(map(lambda region: mask[region.slice], front_regions))
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

    def parallel_front_edge_regions_from_frame(self, rows=None):
        """Calculate a list of the parallel front edge regions of a charge injection ci_frame.

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

        The extracted ci_frame keeps just the front edges of all charge injection regions.

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
                lambda ci_region: self.parallel_front_edge_region(
                    ci_region, rows
                ),
                self.ci_pattern.regions,
            )
        )

    def parallel_trails_line_binned_over_columns_from_frame(
        self, array, rows=None, mask=None
    ):
        trails_stacked_array = self.parallel_trails_stacked_array_from_frame(
            array=array, rows=rows, mask=mask
        )
        return np.ma.mean(np.ma.asarray(trails_stacked_array), axis=1)

    def parallel_trails_stacked_array_from_frame(self, array, rows=None, mask=None):
        trails_arrays = self.parallel_trails_arrays_from_frame(
            array=array, rows=rows, mask=mask
        )
        return np.ma.mean(np.ma.asarray(trails_arrays), axis=0)

    def parallel_trails_arrays_from_frame(self, array, rows=None, mask=None):
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

        The extracted ci_frame keeps just the trails following all charge injection regions:

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
        trails_regions = self.parallel_trails_regions_from_frame(
            shape=array.shape, rows=rows
        )
        trails_arrays = list(map(lambda region: array[region.slice], trails_regions))
        if mask is not None:
            trails_masks = list(map(lambda region: mask[region.slice], trails_regions))
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

    def parallel_trails_regions_from_frame(self, shape, rows=None):
        if rows is None:
            rows = (0, self.smallest_parallel_trails_rows_from_shape(shape=shape))
        """Compute the parallel regions of a charge injection ci_frame.

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

        The extracted ci_frame keeps just the trails following all charge injection regions:

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
        return list(
            map(
                lambda ci_region: self.parallel_trails_region(
                    ci_region, rows
                ),
                self.ci_pattern.regions,
            )
        )

    def serial_front_edge_line_binned_over_rows_from_frame(
        self, array, columns=None, mask=None
    ):
        front_stacked_array = self.serial_front_edge_stacked_array_from_frame(
            array=array, columns=columns, mask=mask
        )
        return np.ma.mean(np.ma.asarray(front_stacked_array), axis=0)

    def serial_front_edge_stacked_array_from_frame(
        self, array, columns=None, mask=None
    ):
        front_arrays = self.serial_front_edge_arrays_from_frame(
            array=array, columns=columns, mask=mask
        )
        return np.ma.mean(np.ma.asarray(front_arrays), axis=0)

    def serial_front_edge_arrays_from_frame(self, array, columns=None, mask=None):
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

        The extracted ci_frame keeps just the serial front edges of all charge injection regions.

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
        front_regions = self.serial_front_edge_regions_from_frame(columns=columns)
        front_arrays = list(map(lambda region: array[region.slice], front_regions))
        if mask is not None:
            front_masks = list(map(lambda region: mask[region.slice], front_regions))
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

    def serial_front_edge_regions_from_frame(self, columns=None):
        """Compute a list of the serial front edges regions of a charge injection ci_frame.

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

        The extracted ci_frame keeps just the serial front edges of all charge injection regions.

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
                lambda ci_region: self.serial_front_edge_region(
                    ci_region, columns
                ),
                self.ci_pattern.regions,
            )
        )

    def serial_trails_line_binned_over_rows_from_frame(
        self, array, columns=None, mask=None
    ):
        trails_stacked_array = self.serial_trails_stacked_array_from_frame(
            array=array, columns=columns, mask=mask
        )
        return np.ma.mean(np.ma.asarray(trails_stacked_array), axis=0)

    def serial_trails_stacked_array_from_frame(self, array, columns=None, mask=None):
        front_arrays = self.serial_trails_arrays_from_frame(
            array=array, columns=columns, mask=mask
        )
        return np.ma.mean(np.ma.asarray(front_arrays), axis=0)

    def serial_trails_arrays_from_frame(self, array, columns=None, mask=None):
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

        The extracted ci_frame keeps just the serial front edges of all charge injection regions.

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
        trails_regions = self.serial_trails_regions_from_frame(columns=columns)
        trails_arrays = list(map(lambda region: array[region.slice], trails_regions))
        if mask is not None:
            trails_masks = list(map(lambda region: mask[region.slice], trails_regions))
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

    def serial_trails_regions_from_frame(self, columns=None):
        """Compute a list of the serial trails regions of a charge injection ci_frame.

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

        The extracted ci_frame keeps just the serial front edges of all charge injection regions.

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
            columns = (0, self.serial_trails_columns)
        return list(
            map(
                lambda ci_region: self.serial_trails_region(
                    ci_region, columns
                ),
                self.ci_pattern.regions,
            )
        )

    def parallel_serial_calibration_section(self, array):
        return array[
            0 : array.shape[0], self.serial_prescan.x0 : array.shape[1]
        ]

    def smallest_parallel_trails_rows_from_shape(self, shape):

        rows_between_regions = self.ci_pattern.rows_between_regions
        rows_to_image_edge = self.parallel_trail_size_to_image_edge(
            shape=shape, ci_pattern=self.ci_pattern
        )
        rows_between_regions.append(rows_to_image_edge)
        return np.min(rows_between_regions)
