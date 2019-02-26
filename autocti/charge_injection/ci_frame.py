import numpy as np

from autocti import exc
from autocti.model import pyarctic


def bin_array_across_serial(array, mask=None):
    return np.mean(np.ma.masked_array(array, mask), axis=1)


def bin_array_across_parallel(array, mask=None):
    return np.mean(np.ma.masked_array(array, mask), axis=0)


class ChInj(object):

    def __init__(self, frame_geometry, ci_pattern):
        """
        Class which represents the CCD quadrant of a charge injection image (e.g. the location of the parallel and
        serial front edge, trails).

        frame_geometry : CIFrame.CIQuadGeometry
            The quadrant geometry of the image, defining where the parallel / serial overscans are and
            therefore the direction of clocking and rotations before input into the cti algorithm.
        ci_pattern : CIPattern.CIPattern
            The charge injection ci_pattern (regions, normalization, etc.) of the charge injection image.
        """
        self.frame_geometry = frame_geometry
        self.ci_pattern = ci_pattern

    def ci_regions_from_array(self, array):
        """Extract an array of all of the charge-injection regions from a charge injection ci_frame.

        The diagram below illustrates the array that is extracted from a ci_frame:

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
        """Extract an array of all of the parallel trails following the charge-injection regions from a charge   
        injection ci_frame.

        The diagram below illustrates the array that is extracted from a ci_frame:

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

        array = array[:, :]

        for region in self.ci_pattern.regions:
            array[region.slice] = 0

        array[self.frame_geometry.serial_overscan.slice] = 0
        array[self.frame_geometry.serial_prescan.slice] = 0

        return array

    def parallel_edges_and_trails_frame_from_frame(self, array, front_edge_rows=None, trails_rows=None):
        """Extract an array of all of the parallel front edges and trails of each the charge-injection regions from   
        a charge injection ci_frame.

        One can specify the range of rows that are extracted, for example:

        front_edge_rows = (0, 1) will extract just the leading front edge row.
        front_edge_rows = (0, 2) will extract the leading two front edge rows.
        trails_rows = (0, 1) will extract the first row of trails closest to the charge injection region.

        The diagram below illustrates the array that is extracted from a ci_frame for front_edge_rows=(0,1) and   
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

            front_regions = list(map(lambda ci_region:
                                     self.frame_geometry.parallel_front_edge_region(ci_region, front_edge_rows),
                                     self.ci_pattern.regions))

            front_edges = self.parallel_front_edge_arrays_from_frame(array, front_edge_rows)

            for i, region in enumerate(front_regions):
                new_array[region.y0:region.y1, region.x0:region.x1] += front_edges[i]

        if trails_rows is not None:

            trails_regions = list(
                map(lambda ci_region: self.frame_geometry.parallel_trails_region(ci_region, trails_rows),
                    self.ci_pattern.regions))

            trails = self.parallel_trails_arrays_from_frame(array, trails_rows)

            for i, region in enumerate(trails_regions):
                new_array[region.y0:region.y1, region.x0:region.x1] += trails[i]

        return new_array

    def parallel_calibration_section_for_columns(self, array, columns):
        """Extract an parallel calibration array from a charge injection ci_frame, where this array is a sub-set of
        the ci_frame which be used for just parallel calibration. Specifically, this ci_frame is a specified number
        of columns closest to the read-out electronics.

        The diagram below illustrates the array that is extracted from a ci_frame with columns=(0, 3):

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
        calibration_region = self.frame_geometry.parallel_side_nearest_read_out_region(self.ci_pattern.regions[0],
                                                                                       array.shape, columns)
        array = array[calibration_region.slice]
        return array

    def serial_all_trails_frame_from_frame(self, array):
        """Extract an array of all of the serial trails in the serial overscan region, that are to the side of a
        charge-injection regions from a charge injection ci_frame.

        The diagram below illustrates the array that is extracted from a ci_frame:

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
        array = self.serial_edges_and_trails_frame_from_frame(array,
                                                              trails_columns=(
                                                                  0, self.frame_geometry.serial_overscan.total_columns))
        return array

    def serial_overscan_above_trails_frame_from_frame(self, array):
        """Extract an array of all of the regions of the serial overscan that don't contain trails from a   
        charge injection region (i.e. are not to the side of one).

        The diagram below illustrates the array that is extracted from a ci_frame:

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
        overscan_slice = self.frame_geometry.serial_overscan.slice

        new_array[overscan_slice] = array[overscan_slice]

        trails_regions = list(map(lambda ci_region:
                                  self.frame_geometry.serial_trails_region(ci_region, (
                                      0, self.frame_geometry.serial_overscan.total_columns)),
                                  self.ci_pattern.regions))

        for region in trails_regions:
            new_array[region.slice] = 0

        return new_array

    def serial_edges_and_trails_frame_from_frame(self, array, front_edge_columns=None, trails_columns=None):
        """Extract an array of all of the serial front edges and trails of each the charge-injection regions from   
        a charge injection ci_frame.

        One can specify the range of columns that are extracted, for example:

        front_edge_columns = (0, 1) will extract just the leading front edge column.
        front_edge_columns = (0, 2) will extract the leading two front edge columns.
        trails_columns = (0, 1) will extract the first column of trails closest to the charge injection region.

        The diagram below illustrates the array that is extracted from a ci_frame for front_edge_columns=(0,2) and   
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
                map(lambda ci_region: self.frame_geometry.serial_front_edge_region(ci_region, front_edge_columns),
                    self.ci_pattern.regions))

            front_edges = self.serial_front_edge_arrays_from_frame(array, front_edge_columns)

            for i, region in enumerate(front_regions):
                new_array[region.y0:region.y1, region.x0:region.x1] += front_edges[i]

        if trails_columns is not None:

            trails_regions = list(
                map(lambda ci_region: self.frame_geometry.serial_trails_region(ci_region, trails_columns),
                    self.ci_pattern.regions))

            trails = self.serial_trails_arrays_from_frame(array, trails_columns)

            for i, region in enumerate(trails_regions):
                new_array[region.y0:region.y1, region.x0:region.x1] += trails[i]

        return new_array

    def serial_calibration_section_for_rows(self, array, rows):
        """Extract a serial calibration array from a charge injection ci_frame, where this array is a sub-set of the
        ci_frame which can be used for serial-only calibration. Specifically, this ci_frame is all charge injection
        regions and their serial over-scan trails.

        The diagram below illustrates the array that is extracted from a ci_frame with column=5:

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
        calibration_images = list(map(lambda image: image[rows[0]:rows[1], :], calibration_images))
        array = np.concatenate(calibration_images, axis=0)
        return array

    def serial_calibration_sub_arrays_from_frame(self, array):
        """Extract each charge injection region image for the serial calibration array above."""

        calibration_regions = list(map(lambda ci_region:
                                       self.frame_geometry.serial_prescan_ci_region_and_trails(ci_region=ci_region,
                                                                                               image_shape=array.shape),
                                       self.ci_pattern.regions))
        return list(map(lambda region: array[region.slice], calibration_regions))

    def parallel_front_edge_arrays_from_frame(self, array, rows):
        """Extract a list of the parallel front edge regions of a charge injection ci_frame.

        The diagram below illustrates the array that is extracted from a ci_frame for rows=(0, 1):

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
        front_regions = list(map(lambda ci_region: self.frame_geometry.parallel_front_edge_region(ci_region, rows),
                                 self.ci_pattern.regions))
        front_arrays = np.array(list(map(lambda region: array[region.slice], front_regions)))
        return front_arrays

    def parallel_trails_arrays_from_frame(self, array, rows):
        """Extract the parallel trails of a charge injection ci_frame.


        The diagram below illustrates the array that is extracted from a ci_frame for rows=(0, 1):

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
        trails_regions = list(map(lambda ci_region: self.frame_geometry.parallel_trails_region(ci_region, rows),
                                  self.ci_pattern.regions))
        trails_arrays = np.array(list(map(lambda region: array[region.slice], trails_regions)))
        return trails_arrays

    def serial_front_edge_arrays_from_frame(self, array, columns):
        """Extract a list of the serial front edges regions of a charge injection ci_frame.

        The diagram below illustrates the array that is extracted from a ci_frame for columnss=(0, 4):

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
        front_regions = list(map(lambda ci_region: self.frame_geometry.serial_front_edge_region(ci_region, columns),
                                 self.ci_pattern.regions))
        front_arrays = np.array(list(map(lambda region: array[region.slice], front_regions)))
        return front_arrays

    def serial_trails_arrays_from_frame(self, array, columns):
        """Extract a list of the serial trails of a charge injection ci_frame.

        The diagram below illustrates the array that is extracted from a ci_frame for columnss=(0, 3):

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
        trails_regions = list(map(lambda ci_region: self.frame_geometry.serial_trails_region(ci_region, columns),
                                  self.ci_pattern.regions))
        trails_arrays = np.array(list(map(lambda region: array[region.slice], trails_regions)))
        return trails_arrays

    def parallel_serial_calibration_section(self, array):
        return array[0:array.shape[0], self.frame_geometry.serial_prescan.x1:array.shape[1]]

    def mask_containing_only_serial_trails(self, array):

        from autocti.data import mask as msk

        mask = np.full(array.shape, True)
        trails_regions = list(map(lambda ci_region: self.frame_geometry.serial_trails_region(ci_region,
                                                                                             columns=(
                                                                                                 0, array.shape[1])),
                                  self.ci_pattern.regions))
        for region in trails_regions:
            mask[region.y0:region.y1, region.x0:region.x1] = False

        return msk.Mask(array=mask, frame_geometry=self.frame_geometry)


class Region(object):

    def __init__(self, region):
        """Setup a region of an image, which could be where the parallel overscan, serial overscan, etc. are.

        This is defined as a tuple (y0, y1, x0, x1).

        Parameters
        -----------
        region : (int,)
            The coordinates on the image of the region (y0, y1, x0, y1).
        """

        if region[0] < 0 or region[1] < 0 or region[2] < 0 or region[3] < 0:
            raise exc.RegionException('A coordinate of the Region was specified as negative.')

        if region[0] >= region[1]:
            raise exc.RegionException('The first row in the Region was equal to or greater than the second row.')

        if region[2] >= region[3]:
            raise exc.RegionException('The first column in the Region was equal to greater than the second column.')
        self.region = region

    @property
    def total_rows(self):
        return self.y1 - self.y0

    @property
    def total_columns(self):
        return self.x1 - self.x0

    @property
    def y0(self):
        return self[0]

    @property
    def y1(self):
        return self[1]

    @property
    def x0(self):
        return self[2]

    @property
    def x1(self):
        return self[3]

    def __getitem__(self, item):
        return self.region[item]

    def __eq__(self, other):
        if self.region == other:
            return True
        return super().__eq__(other)

    def __repr__(self):
        return "<Region {} {} {} {}>".format(*self)

    @property
    def slice(self):
        return np.s_[self.y0:self.y1, self.x0:self.x1]

    @property
    def y_slice(self):
        return np.s_[self.y0:self.y1]

    @property
    def x_slice(self):
        return np.s_[self.x0:self.x1]

    @property
    def shape(self):
        return self.y1 - self.y0, self.x1 - self.x0


class FrameGeometry(object):

    def __init__(self, corner, parallel_overscan, serial_prescan, serial_overscan):
        """Abstract class for the geometry of a CTI Image.

        A ImageFrame is stored as a 2D NumPy array. When this immage is passed to arctic, clocking goes towards
        the 'top' of the NumPy array (e.g. towards row 0). Trails therefore appear towards the 'bottom' of the array   
        (e.g. the final row).

        Arctic has no in-built functionality for changing the direction of clocking depending on the input   
        configuration file. Therefore, image rotations are handled before arctic is called, using the functions   
        defined in this class (and its children). These routines define how an image is rotated before parallel   
        and serial clocking and how to reorient the image back to its original orientation after clocking is performed.

        Currently, only four geometries are available, which are specific to Euclid (and documented in the   
        *QuadGeometryEuclid* class).

        Parameters
        -----------
        parallel_overscan : ci_frame.Region
            The parallel overscan region of the ci_frame.
        serial_prescan : ci_frame.Region
            The serial prescan region of the ci_frame.
        serial_overscan : ci_frame.Region
            The serial overscan region of the ci_frame.
        """

        self.parallel_overscan = parallel_overscan
        self.serial_prescan = serial_prescan
        self.serial_overscan = serial_overscan
        self.corner = corner

    def add_cti(self, image, cti_params, cti_settings, use_parallel_poisson_densities=False):
        """add cti to an image.

        Parameters
        ----------
        image : ndarray
            The image cti is added too.
        cti_params : ArcticParams.ArcticParams
            The CTI model parameters (trap density, trap lifetimes etc.).
        cti_settings : ArcticSettings.ArcticSettings
            The settings that control the cti clocking algorithm (e.g. the ccd well_depth express option).
        """

        if cti_params.parallel_ccd is not None:
            image_pre_parallel_clocking = self.rotate_for_parallel_cti(image=image)
            image_post_parallel_clocking = pyarctic.call_arctic(image=image_pre_parallel_clocking,
                                                                species=cti_params.parallel_species,
                                                                ccd=cti_params.parallel_ccd,
                                                                settings=cti_settings.parallel,
                                                                correct_cti=False,
                                                                use_poisson_densities=use_parallel_poisson_densities)
            image = self.rotate_for_parallel_cti(image_post_parallel_clocking)

        if cti_params.serial_ccd is not None:
            image_pre_serial_clocking = self.rotate_before_serial_cti(image_pre_clocking=image)
            image_post_serial_clocking = pyarctic.call_arctic(image=image_pre_serial_clocking,
                                                              species=cti_params.serial_species,
                                                              ccd=cti_params.serial_ccd,
                                                              settings=cti_settings.serial,
                                                              correct_cti=False,
                                                              use_poisson_densities=False)
            image = self.rotate_after_serial_cti(image_post_serial_clocking)

        return image

    def correct_cti(self, image, cti_params, cti_settings):
        """Correct cti from an image.

        Parameters
        ----------
        image : ndarray
            The image cti is corrected from.
        cti_params : ArcticParams.ArcticParams
            The CTI model parameters (trap density, trap lifetimes etc.).
        cti_settings : ArcticSettings.ArcticSettings
            The settings that control the cti clocking algorithm (e.g. ccd well_depth express option).
        """

        if cti_settings.serial is not None:
            image_pre_serial_clocking = self.rotate_before_serial_cti(image_pre_clocking=image)
            image_post_serial_clocking = pyarctic.call_arctic(image=image_pre_serial_clocking,
                                                              species=cti_params.serial_species,
                                                              ccd=cti_params.serial_ccd,
                                                              settings=cti_settings.serial,
                                                              correct_cti=True,
                                                              use_poisson_densities=False)
            image = self.rotate_after_serial_cti(image_post_serial_clocking)

        if cti_settings.parallel is not None:
            image_pre_parallel_clocking = self.rotate_for_parallel_cti(image=image)
            image_post_parallel_clocking = pyarctic.call_arctic(image=image_pre_parallel_clocking,
                                                                species=cti_params.parallel_species,
                                                                ccd=cti_params.parallel_ccd,
                                                                settings=cti_settings.parallel,
                                                                correct_cti=True,
                                                                use_poisson_densities=False)
            image = self.rotate_for_parallel_cti(image_post_parallel_clocking)

        return image

    def rotate_for_parallel_cti(self, image):
        return flip(image) if self.corner[0] == 1 else image

    def rotate_before_serial_cti(self, image_pre_clocking):
        transposed = image_pre_clocking.T.copy()
        return flip(transposed) if self.corner[1] == 1 else transposed

    def rotate_after_serial_cti(self, image_post_clocking):
        flipped = flip(image_post_clocking) if self.corner[1] == 1 else image_post_clocking
        return flipped.T.copy()

    def parallel_trail_from_y(self, y, dy):
        """Coordinates of a parallel trail of size dy from coordinate y"""
        return y - dy * self.corner[0], y + 1 + dy * (1 - self.corner[0])

    def serial_trail_from_x(self, x, dx):
        """Coordinates of a serial trail of size dx from coordinate x"""
        return x - dx * self.corner[1], x + 1 + dx * (1 - self.corner[1])

    def parallel_front_edge_region(self, ci_region, rows=(0, 1)):
        check_parallel_front_edge_size(region=ci_region, rows=rows)
        if self.corner[0] == 0:
            y_coord = ci_region.y0
            y_min = y_coord + rows[0]
            y_max = y_coord + rows[1]
        else:
            y_coord = ci_region.y1
            y_min = y_coord - rows[1]
            y_max = y_coord - rows[0]
        return Region((y_min, y_max, ci_region.x0, ci_region.x1))

    def parallel_trails_region(self, ci_region, rows=(0, 1)):
        if self.corner[0] == 0:
            y_coord = ci_region.y1
            y_min = y_coord + rows[0]
            y_max = y_coord + rows[1]
        else:
            y_coord = ci_region.y0
            y_min = y_coord - rows[1]
            y_max = y_coord - rows[0]
        return Region((y_min, y_max, ci_region.x0, ci_region.x1))

    def x_limits(self, region, columns):
        if self.corner[1] == 0:
            x_coord = region.x0
            x_min = x_coord + columns[0]
            x_max = x_coord + columns[1]
        else:
            x_coord = region.x1
            x_min = x_coord - columns[1]
            x_max = x_coord - columns[0]
        return x_min, x_max

    def serial_front_edge_region(self, ci_region, columns=(0, 1)):
        check_serial_front_edge_size(ci_region, columns)
        x_min, x_max = self.x_limits(ci_region, columns)
        return Region((ci_region.y0, ci_region.y1, x_min, x_max))

    def parallel_side_nearest_read_out_region(self, ci_region, image_shape, columns=(0, 1)):
        x_min, x_max = self.x_limits(ci_region, columns)
        return Region((0, image_shape[0], x_min, x_max))

    def serial_trails_region(self, ci_region, columns=(0, 1)):
        if self.corner[1] == 0:
            x_coord = ci_region.x1
            x_min = x_coord + columns[0]
            x_max = x_coord + columns[1]
        else:
            x_coord = ci_region.x0
            x_min = x_coord - columns[1]
            x_max = x_coord - columns[0]
        return Region((ci_region.y0, ci_region.y1, x_min, x_max))

    def serial_prescan_ci_region_and_trails(self, ci_region, image_shape):
        if self.corner[1] == 0:
            x_min = 0
            x_max = image_shape[1]
        else:
            x_min = 0
            x_max = image_shape[1]
        return Region((ci_region.y0, ci_region.y1, x_min, x_max))


class QuadGeometryEuclid(FrameGeometry):

    def __init__(self, corner, parallel_overscan, serial_prescan, serial_overscan):
        """Abstract class for the ci_frame geometry of Euclid quadrants. CTI uses a bias corrected raw VIS ci_frame, which   
         is  described at http://euclid.esac.esa.int/dm/dpdd/latest/le1dpd/dpcards/le1_visrawframe.html

        A ImageFrame is stored as a 2D NumPy array. When an image is passed to arctic, clocking goes towards the 'top'
        of the NumPy array (e.g. towards row 0). Trails therefore appear towards the 'bottom' of the array (e.g. the   
        final row).

        Arctic has no in-built functionality for changing the direction of clocking depending on the input   
        configuration file. Therefore, image rotations are handled before arctic is called, using the functions   
        defined in this class (and its children). These routines define how an image is rotated before parallel   
        and serial clocking with arctic. They also define how to reorient the image to its original orientation after   
        clocking with arctic is performed.

        A Euclid CCD is defined as below:

        ---KEY---
        ---------

        [] = read-out electronics

        [==========] = read-out register

        [xxxxxxxxxx]
        [xxxxxxxxxx] = CCD panel
        [xxxxxxxxxx]

        P = Parallel Direction
        S = Serial Direction

             <--------S-----------   ---------S----------->
        [] [========= 2 =========] [========= 3 =========] []          |
        /    [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  /          |
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | Direction arctic
        P   [xxxxxxxxx 2 xxxxxxxxx] [xxxxxxxxx 3 xxxxxxxxx]  P         | clocks an image
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | without any rotation
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | (e.g. towards row 0
                                                                       | of the NumPy array)
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
        P   [xxxxxxxxx 0 xxxxxxxxx] [xxxxxxxxx 1 xxxxxxxxx] P          |
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
            [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]            |
                                                                        
        [] [========= 0 =========] [========= 1 =========] []
            <---------S----------   ----------S----------->

        Note that the arrow on the right defines the direction of clocking by arctic without any rotation. Therefore,   
        there are 8 circumstances of how arctic requires an image to be rotated before clocking:

        - Quadrant 0 - QuadGeometryEuclid.bottom_left()  - Parallel Clocking - No rotation.
        - Quadrant 0 - QuadGeometryEuclid.bottom_left()  - Serial Clocking   - Rotation 90 degrees clockwise.
        - Quadrant 1 - QuadGeometryEuclid.bottom_right() - Parallel Clocking - No rotation.
        - Quadrant 1 - QuadGeometryEuclid.bottom_right() - Serial Clocking   - Rotation 270 degrees clockwise.
        - Quadrant 2 - QuadGeometryEuclid.top_left()     - Parallel Clocking - Rotation 180 degrees.
        - Quadrant 2 - QuadGeometryEuclid.top_left()     - Serial Clocking   - Rotation 90 degrees clockwise.
        - Quadrant 3 - QuadGeometryEuclid.top_right()    - Parallel Clocking - Rotation 180 degrees.
        - Quadrant 3 - QuadGeometryEuclid.top_right()    - Serial Clocking   - Rotation 270 degrees clockwise

        After clocking has been performed with arctic (and CTI is added / corrected), it must be re-rotated back to   
        its original orientation. This rotation is the reverse of what is specified above.

        Rotations are performed using flipup / fliplr routines, but will ultimately use the Euclid Image Tools library.

        """
        super(QuadGeometryEuclid, self).__init__(corner=corner, parallel_overscan=parallel_overscan,
                                                 serial_prescan=serial_prescan, serial_overscan=serial_overscan)

    @classmethod
    def from_ccd_and_quadrant_id(cls, ccd_id, quad_id):
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
                                                                       | of the NumPy array)
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
                                                                       | of the NumPy array)
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

        if (row_index in '123') and (quad_id == 'E'):
            return QuadGeometryEuclid.bottom_left()
        elif (row_index in '123') and (quad_id == 'F'):
            return QuadGeometryEuclid.bottom_right()
        elif (row_index in '123') and (quad_id == 'G'):
            return QuadGeometryEuclid.top_right()
        elif (row_index in '123') and (quad_id == 'H'):
            return QuadGeometryEuclid.top_left()
        elif (row_index in '456') and (quad_id == 'E'):
            return QuadGeometryEuclid.top_right()
        elif (row_index in '456') and (quad_id == 'F'):
            return QuadGeometryEuclid.top_left()
        elif (row_index in '456') and (quad_id == 'G'):
            return QuadGeometryEuclid.bottom_left()
        elif (row_index in '456') and (quad_id == 'H'):
            return QuadGeometryEuclid.bottom_right()

    @classmethod
    def bottom_left(cls):
        return QuadGeometryEuclid(corner=(0, 0),
                                  parallel_overscan=Region((2066, 2086, 51, 2099)),
                                  serial_prescan=Region((0, 2086, 0, 51)),
                                  serial_overscan=Region((0, 2086, 2099, 2119)))

    @classmethod
    def bottom_right(cls):
        return QuadGeometryEuclid(corner=(0, 1),
                                  parallel_overscan=Region((2066, 2086, 20, 2068)),
                                  serial_prescan=Region((0, 2086, 2068, 2119)),
                                  serial_overscan=Region((0, 2086, 0, 20)))

    @classmethod
    def top_left(cls):
        return QuadGeometryEuclid(corner=(1, 0),
                                  parallel_overscan=Region((0, 20, 51, 2099)),
                                  serial_prescan=Region((0, 2086, 0, 51)),
                                  serial_overscan=Region((0, 2086, 2099, 2119)))

    @classmethod
    def top_right(cls):
        return QuadGeometryEuclid(corner=(1, 1),
                                  parallel_overscan=Region((0, 20, 20, 2068)),
                                  serial_prescan=Region((0, 2086, 2068, 2119)),
                                  serial_overscan=Region((0, 2086, 0, 20)))


def flip(image):
    return image[::-1, :]


def check_parallel_front_edge_size(region, rows):
    # TODO: are these checks important?
    if rows[0] < 0 or rows[1] < 1 or rows[1] > region.y1 - region.y0 or rows[0] >= rows[1]:
        raise exc.CIPatternException('The number of rows to extract from the leading edge is bigger than the entire'
                                     'ci ci_region')


def check_serial_front_edge_size(region, columns):
    if columns[0] < 0 or columns[1] < 1 or columns[1] > region.x1 - region.x0 or columns[0] >= columns[1]:
        raise exc.CIPatternException('The number of columns to extract from the leading edge is bigger than the entire'
                                     'ci ci_region')
