import numpy as np

from autocti import exc
from autocti.data import cti_image
from autocti.tools import imageio


def bin_array_across_serial(array, mask=None):
    return np.mean(np.ma.masked_array(array, mask), axis=1)


def bin_array_across_parallel(array, mask=None):
    return np.mean(np.ma.masked_array(array, mask), axis=0)


class ChInj(np.ndarray):
    """Class which represents the CCD quadrant of a charge injection image (e.g. the location of the parallel and \
    serial front edge, trails).

    frame_geometry : CIFrame.CIQuadGeometry
        The quadrant geometry of the image, defining where the parallel / serial overscans are and \
        therefore the direction of clocking and rotations before input into the cti algorithm.
    ci_pattern : CIPattern.CIPattern
        The charge injection ci_pattern (regions, normalization, etc.) of the charge injection image.
    """

    frame_geometry = None
    ci_pattern = None

    def ci_regions_frame_from_frame(self):
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
        \/[...][ccccccccccccccccccccc][sss]    |
                                               \/
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
        \/[000][ccccccccccccccccccccc][000]    |
                                               \/
        []     [=====================]
               <---------S----------



        """

        array = np.zeros(self.shape)

        for region in self.ci_pattern.regions:
            array = region.add_region_from_image_to_array(image=self, array=array)

        return self.__class__(self.frame_geometry, self.ci_pattern, array)

    def parallel_non_ci_regions_frame_from_frame(self):
        """Extract an array of all of the parallel trails following the charge-injection regions from a charge \
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
        \/[...][ccccccccccccccccccccc][sss]    |
                                               \/
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
        \/[000][000000000000000000000][000]    |
                                               \/
        []     [=====================]
               <---------S----------
        """

        array = self[:, :]

        for region in self.ci_pattern.regions:
            region.set_region_on_array_to_zeros(array=array)

        self.frame_geometry.serial_overscan.set_region_on_array_to_zeros(array=array)
        self.frame_geometry.serial_prescan.set_region_on_array_to_zeros(array=array)

        return self.__class__(self.frame_geometry, self.ci_pattern, array)

    def parallel_edges_and_trails_frame_from_frame(self, front_edge_rows=None, trails_rows=None):
        """Extract an array of all of the parallel front edges and trails of each the charge-injection regions from \
        a charge injection ci_frame.

        One can specify the range of rows that are extracted, for example:

        front_edge_rows = (0, 1) will extract just the leading front edge row.
        front_edge_rows = (0, 2) will extract the leading two front edge rows.
        trails_rows = (0, 1) will extract the first row of trails closest to the charge injection region.

        The diagram below illustrates the array that is extracted from a ci_frame for front_edge_rows=(0,1) and \
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
        \/[...][ccccccccccccccccccccc][sss]    |
                                               \/
        []     [=====================]
               <---------S----------

        The extracted ci_frame keeps just the leading edges and trails following all charge injection regions and \
        replaces all other values with 0s:

               [000000000000000000000]
               [000000000000000000000]
          [000][ttttttttttttttttttttt][000]
          [000][000000000000000000000][000]
        | [000][ccccccccccccccccccccc][000]    |
        | [000][000000000000000000000][000]    | Direction
        P [000][ttttttttttttttttttttt][000]    | of
        | [000][000000000000000000000][000]    | clocking
        \/[000][ccccccccccccccccccccc][000]    |
                                               \/
        []     [=====================]
               <---------S----------

        Parameters
        ------------
        front_edge_rows : (int, int)
            The row indexes to extract the front edge between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows).
        trails_rows : (int, int)
            The row indexes to extract the trails between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows)
        """
        array = np.zeros(self.shape)

        if front_edge_rows is not None:

            front_regions = list(map(lambda ci_region:
                                     self.frame_geometry.parallel_front_edge_region(ci_region, front_edge_rows),
                                     self.ci_pattern.regions))

            front_edges = self.parallel_front_edge_arrays_from_frame(front_edge_rows)

            for i, region in enumerate(front_regions):
                array[region.y0:region.y1, region.x0:region.x1] += front_edges[i]

        if trails_rows is not None:

            trails_regions = list(
                map(lambda ci_region: self.frame_geometry.parallel_trails_region(ci_region, trails_rows),
                    self.ci_pattern.regions))

            trails = self.parallel_trails_arrays_from_frame(trails_rows)

            for i, region in enumerate(trails_regions):
                array[region.y0:region.y1, region.x0:region.x1] += trails[i]

        return self.__class__(self.frame_geometry, self.ci_pattern, array)

    def parallel_calibration_section_for_columns(self, columns):
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
        \/[...][ccccccccccccccccccccc][sss]    |
                                               \/
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
        \/     [ccc]                           |
                                               \/
        []     [=====================]
               <---------S----------
        """
        calibration_region = self.frame_geometry.parallel_side_nearest_read_out_region(self.ci_pattern.regions[0],
                                                                                       self.shape, columns)
        array = calibration_region.extract_region_from_array(self)
        return self.__class__(frame_geometry=self.frame_geometry, ci_pattern=self.ci_pattern, array=array)

    def serial_all_trails_frame_from_frame(self):
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
        \/[...][ccccccccccccccccccccc][sts]    |
                                               \/
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
        \/[000][000000000000000000000][sts]    |
                                               \/
        []     [=====================]
               <---------S----------
        """
        array = self.serial_edges_and_trails_frame_from_frame(
            trails_columns=(0, self.frame_geometry.serial_overscan.total_columns))
        return self.__class__(self.frame_geometry, self.ci_pattern, array)

    def serial_overscan_non_trails_frame_from_frame(self):
        """Extract an array of all of the regions of the serial overscan that don't contain trails from a \
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
        \/[...][ccccccccccccccccccccc][sts]    |
                                               \/
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
        \/[000][000000000000000000000][000]    |
                                               \/
        []     [=====================]
               <---------S----------
        """
        array = self.frame_geometry.serial_overscan.add_region_from_image_to_array(image=self,
                                                                                   array=np.zeros(self.shape))

        trails_regions = list(map(lambda ci_region:
                                  self.frame_geometry.serial_trails_region(ci_region, (
                                      0, self.frame_geometry.serial_overscan.total_columns)),
                                  self.ci_pattern.regions))

        for i, region in enumerate(trails_regions):
            array = region.set_region_on_array_to_zeros(array)

        return self.__class__(self.frame_geometry, self.ci_pattern, array)

    def serial_edges_and_trails_frame_from_frame(self, front_edge_columns=None, trails_columns=None):
        """Extract an array of all of the serial front edges and trails of each the charge-injection regions from \
        a charge injection ci_frame.

        One can specify the range of columns that are extracted, for example:

        front_edge_columns = (0, 1) will extract just the leading front edge column.
        front_edge_columns = (0, 2) will extract the leading two front edge columns.
        trails_columns = (0, 1) will extract the first column of trails closest to the charge injection region.

        The diagram below illustrates the array that is extracted from a ci_frame for front_edge_columns=(0,2) and \
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
        \/[...][ccccccccccccccccccccc][tst]    |
                                               \/
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
        \/[000][cc0000000000000000000][st0]    |
                                               \/
        []     [=====================]
               <---------S----------

        Parameters
        ------------
        front_edge_columns : (int, int)
            The column indexes to extract the front edge between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd rows)
        trails_columns : (int, int)
            The column indexes to extract the trails between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd rows)
        """
        array = np.zeros(self.shape)

        if front_edge_columns is not None:

            front_regions = list(
                map(lambda ci_region: self.frame_geometry.serial_front_edge_region(ci_region, front_edge_columns),
                    self.ci_pattern.regions))

            front_edges = self.serial_front_edge_arrays_from_frame(front_edge_columns)

            for i, region in enumerate(front_regions):
                array[region.y0:region.y1, region.x0:region.x1] += front_edges[i]

        if trails_columns is not None:

            trails_regions = list(
                map(lambda ci_region: self.frame_geometry.serial_trails_region(ci_region, trails_columns),
                    self.ci_pattern.regions))

            trails = self.serial_trails_arrays_from_frame(trails_columns)

            for i, region in enumerate(trails_regions):
                array[region.y0:region.y1, region.x0:region.x1] += trails[i]

        return self.__class__(self.frame_geometry, self.ci_pattern, array)

    def serial_calibration_section_for_column_and_rows(self, from_column, rows):
        """Extract a serial calibration array from a charge injection ci_frame, where this array is a sub-set of the
        ci_frame which can be used for serial-only calibration. Specifically, this ci_frame is all charge injection
        regions and their serial over-scan trails, specified from a certain column from the read-out electronics.

        The diagram below illustrates the array that is extracted from a ci_frame with from_column=5:

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
        \/[...][ccccccccccccccccccccc][sts]    |
                                               \/
        []     [=====================]
               <---------S----------

        The extracted ci_frame keeps just the trails following all charge injection regions and replaces all other
        values with 0s:

        |                                      |
        |      [cccccccccccccccc][tst]         | Direction
        P      [cccccccccccccccc][tst]         | of
        |      [cccccccccccccccc][tst]         | clocking
        \/     [cccccccccccccccc][tst]         |
                                               \/
        []     [=====================]
               <---------S----------
        """
        calibration_images = self.serial_calibration_sub_arrays_from_frame(from_column)
        calibration_images = list(map(lambda image: image[rows[0]:rows[1], :], calibration_images))
        array = np.concatenate(calibration_images, axis=0)
        return self.__class__(frame_geometry=self.frame_geometry, ci_pattern=self.ci_pattern, array=array)

    def serial_calibration_sub_arrays_from_frame(self, from_column):
        """Extract each charge injection region image for the serial calibration array above."""

        calibration_regions = list(map(lambda ci_region:
                                       self.frame_geometry.serial_ci_region_and_trails(ci_region, self.shape,
                                                                                       from_column),
                                       self.ci_pattern.regions))
        return list(map(lambda region: region.extract_region_from_array(self), calibration_regions))

    def parallel_front_edge_arrays_from_frame(self, rows):
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
        \/[...][cc0ccc0cccc0cccc0cccc][sss]    |
                                               \/
        []     [=====================]
               <---------S----------

        The extracted ci_frame keeps just the front edges of all charge injection regions.

        list index 0:

        [c0c0c0cc0c0c0c0c0c0c0]

        list index 1:

        [1c1c1c1c1c1c1c1c1c1c1]

        Parameters
        ------------
        rows : (int, int)
            The row indexes to extract the front edge between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows)
        """
        front_regions = list(map(lambda ci_region: self.frame_geometry.parallel_front_edge_region(ci_region, rows),
                                 self.ci_pattern.regions))
        front_arrays = np.array(list(map(lambda region: region.extract_region_from_array(self), front_regions)))
        return list(map(lambda front_array:
                        self.__class__(frame_geometry=self.frame_geometry, ci_pattern=self.ci_pattern,
                                       array=front_array),
                        front_arrays))

    def parallel_trails_arrays_from_frame(self, rows):
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
        \/[...][cc0ccc0cccc0cccc0cccc][sss]    |
                                               \/
        []     [=====================]
               <---------S----------

        The extracted ci_frame keeps just the trails following all charge injection regions:

        list index 0:

        [t0t0t0tt0t0t0t0t0t0t0]

        list index 1:

        [1t1t1t1t1t1t1t1t1t1t1]

        Parameters
        ------------
        rows : (int, int)
            The row indexes to extract the trails between (e.g. rows(0, 3) extracts the 1st, 2nd and 3rd rows)
        """
        trails_regions = list(map(lambda ci_region: self.frame_geometry.parallel_trails_region(ci_region, rows),
                                  self.ci_pattern.regions))
        trails_arrays = np.array(list(map(lambda region: region.extract_region_from_array(self), trails_regions)))
        return list(map(lambda front_array:
                        self.__class__(frame_geometry=self.frame_geometry, ci_pattern=self.ci_pattern,
                                       array=front_array),
                        trails_arrays))

    def serial_front_edge_arrays_from_frame(self, columns):
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
        \/[...][cc0ccc0cccc0cccc0cccc][sss]    |
                                               \/
        []     [=====================]
               <---------S----------

        The extracted ci_frame keeps just the serial front edges of all charge injection regions.

        list index 0:

        [c0c0]

        list index 1:

        [1c1c]

        Parameters
        ------------
        columns : (int, int)
            The column indexes to extract the front edge between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd
            columns)
        """
        front_regions = list(map(lambda ci_region: self.frame_geometry.serial_front_edge_region(ci_region, columns),
                                 self.ci_pattern.regions))
        front_arrays = np.array(list(map(lambda region: region.extract_region_from_array(self), front_regions)))
        return list(map(lambda front_array:
                        self.__class__(frame_geometry=self.frame_geometry, ci_pattern=self.ci_pattern,
                                       array=front_array),
                        front_arrays))

    def serial_trails_arrays_from_frame(self, columns):
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
        \/[...][cc0ccc0cccc0cccc0cccc][ts0]    |
                                               \/
        []     [=====================]
               <---------S----------

        The extracted ci_frame keeps just the serial front edges of all charge injection regions.

        list index 0:

        [st0]

        list index 1:

        [st1]

        Parameters
        ------------
        columns : (int, int)
            The column indexes to extract the trails between (e.g. columns(0, 3) extracts the 1st, 2nd and 3rd columns)
        """
        trails_regions = list(map(lambda ci_region: self.frame_geometry.serial_trails_region(ci_region, columns),
                                  self.ci_pattern.regions))
        trails_arrays = np.array(list(map(lambda region: region.extract_region_from_array(self), trails_regions)))
        return list(map(lambda front_array:
                        self.__class__(frame_geometry=self.frame_geometry, ci_pattern=self.ci_pattern,
                                       array=front_array),
                        trails_arrays))

    def parallel_serial_calibration_section(self):
        return self.__class__(frame_geometry=self.frame_geometry, ci_pattern=self.ci_pattern,
                              array=self[0:self.shape[0], self.frame_geometry.serial_prescan.x1:self.shape[1]])

    def mask_containing_only_serial_trails(self):

        from autocti.data import mask as msk

        mask = np.full(self.shape, True)
        trails_regions = list(map(lambda ci_region: self.frame_geometry.serial_trails_region(ci_region,
                                                                                             columns=(
                                                                                                 0, self.shape[1])),
                                  self.ci_pattern.regions))
        for region in trails_regions:
            mask[region.y0:region.y1, region.x0:region.x1] = False

        return msk.Mask(array=mask, frame_geometry=self.frame_geometry)

    def check_frame_geometry(self):
        if not isinstance(self.frame_geometry, CIQuadGeometry):
            raise exc.CIFrameException('You must supply the CI Frame with a CIQuadGeometry (Not a ci_image.Geometry)')


class CIFrame(cti_image.ImageFrame, ChInj):

    def __new__(cls, frame_geometry, ci_pattern, array, **kwargs):
        """Class which represents the CCD quadrant of a charge injection image (e.g. the location of the parallel and
        serial front edge, trails).

        Parameters
        ----------
        frame_geometry : CIFrame.CIQuadGeometry
            The quadrant geometry of the image, defining where the parallel / serial overscans are and
            therefore the direction of clocking and rotations before input into the cti algorithm.
        ci_pattern : CIPattern.CIPattern
            The charge injection ci_pattern (regions, normalization, etc.) of the charge injection image.
        array : ndarray
            2D Array of array charge injection image ci_data.
        """
        ci_inj = super(CIFrame, cls).__new__(cls, frame_geometry, array)
        ci_inj.check_frame_geometry()
        ci_inj.ci_pattern = ci_pattern
        return ci_inj

    def __init__(self, frame_geometry, ci_pattern, array):
        super(CIFrame, self).__init__(frame_geometry, array)
        self.check_frame_geometry()
        self.ci_pattern = ci_pattern

    @classmethod
    def from_fits_and_ci_pattern(cls, path, filename, hdu, frame_geometry, ci_pattern):
        """Load the image ci_data from a fits file.

        Params
        ----------
        path : str
            The path to the ci_data
        filename : str
            The file phase_name of the fits image ci_data.
        hdu : int
            The HDU number in the fits file containing the image ci_data.
        frame_geometry : CIFrame.CIQuadGeometry
            The quadrant geometry of the image, defining where the parallel / serial overscans are and \
            therefore the direction of clocking and rotations before input into the cti algorithm.
        ci_pattern : CIPattern.CIPattern
            The charge injection ci_pattern (regions, normalization, etc.) of the charge injection image.
        """
        return cls(frame_geometry=frame_geometry, ci_pattern=ci_pattern,
                   array=imageio.numpy_array_from_fits(path, filename, hdu))

    @classmethod
    def from_single_value(cls, value, shape, frame_geometry, ci_pattern):
        """
        Creates an instance of Array and fills it with a single value

        Params
        ----------
        value: float
            The value with which the array should be filled
        shape: (int, int)
            The image_shape of the array
        frame_geometry : CIFrame.CIQuadGeometry
            The quadrant geometry of the image, defining where the parallel / serial overscans are and \
            therefore the direction of clocking and rotations before input into the cti algorithm.
        ci_pattern : CIPattern.CIPattern
            The charge injection ci_pattern (regions, normalization, etc.) of the charge injection image.

        Returns
        -------
        array: ScaledArray
            An array filled with a single value
        """
        array = np.ones(shape) * value
        return cls(frame_geometry=frame_geometry, ci_pattern=ci_pattern, array=array)


class CIFrameCTI(cti_image.CTIImage, ChInj):

    def __new__(cls, frame_geometry, ci_pattern, array, **kwargs):
        """Class which represents the CCD quadrant of a charge injection image (e.g. the location of the parallel and \
        serial front edge, trails), including routes to add cti to or correct cti from the image.

        Parameters
        ----------
        frame_geometry : CIFrame.CIQuadGeometry
            The quadrant geometry of the image, defining where the parallel / serial overscans are and \
            therefore the direction of clocking and rotations before input into the cti algorithm.
        ci_pattern : CIPattern.CIPattern
            The charge injection ci_pattern (regions, normalization, etc.) of the charge injection image.
        array : ndarray
            2D Array of array charge injection image ci_data.
        """
        ci_inj = super(CIFrameCTI, cls).__new__(cls, frame_geometry, array)
        ci_inj.check_frame_geometry()
        ci_inj.ci_pattern = ci_pattern
        return ci_inj

    def __init__(self, frame_geometry, ci_pattern, array):
        super(CIFrameCTI, self).__init__(frame_geometry, array)
        self.check_frame_geometry()
        self.ci_pattern = ci_pattern

    @classmethod
    def from_fits_and_ci_pattern(cls, path, filename, hdu, frame_geometry, ci_pattern):
        """Load the image ci_data from a fits file.

        Params
        ----------
        path : str
            The path to the ci_data
        filename : str
            The file phase_name of the fits image ci_data.
        hdu : int
            The HDU number in the fits file containing the image ci_data.
        frame_geometry : CIFrame.CIQuadGeometry
            The quadrant geometry of the image, defining where the parallel / serial overscans are and \
            therefore the direction of clocking and rotations before input into the cti algorithm.
        ci_pattern : CIPattern.CIPattern
            The charge injection ci_pattern (regions, normalization, etc.) of the charge injection image.
        """
        return cls(frame_geometry=frame_geometry, ci_pattern=ci_pattern,
                   array=imageio.numpy_array_from_fits(path, filename, hdu))


class CIQuadGeometry(object):
    pass


class CIQuadGeometryEuclidBL(cti_image.QuadGeometryEuclidBL, CIQuadGeometry):

    def __init__(self):
        """This class represents the quadrant geometry of a Euclid charge injection image in the bottom-left of a \
        CCD (see **QuadGeometryEuclid** for a description of the Euclid CCD / FPA)"""
        super(CIQuadGeometryEuclidBL, self).__init__()

    @staticmethod
    def parallel_front_edge_region(region, rows=(0, 1)):
        """Extract the leading edge of a charge injection region which is located in the bottom-left quadrant of a \
        Euclid CCD (see *CIPatternGeometry*)."""
        check_parallel_front_edge_size(region, rows)
        return cti_image.Region((region.y0 + rows[0], region.y0 + rows[1], region.x0, region.x1))

    @staticmethod
    def parallel_trails_region(region, rows=(0, 1)):
        """Extract the trails after a charge injection region which is located in the bottom-left quadrant of a \
        Euclid CCD (see *CIPatternGeometry* for a description of where this is extracted)."""
        return cti_image.Region((region.y1 + rows[0], region.y1 + rows[1], region.x0, region.x1))

    @staticmethod
    def parallel_side_nearest_read_out_region(region, image_shape, columns=(0, 1)):
        return cti_image.Region((0, image_shape[0], region.x0 + columns[0], region.x0 + columns[1]))

    @staticmethod
    def serial_front_edge_region(region, columns=(0, 1)):
        """Extract the leading edge of a charge injection region which is located in the bottom-left quadrant of a \
        Euclid CCD (see *CIPatternGeometry*)."""
        check_serial_front_edge_size(region, columns)
        return cti_image.Region((region.y0, region.y1, region.x0 + columns[0], region.x0 + columns[1]))

    @staticmethod
    def serial_trails_region(region, columns=(0, 1)):
        """Extract the trails after a charge injection region which is located in the bottom-left quadrant of a \
        Euclid CCD (see *CIPatternGeometry* for a description of where this is extracted)."""
        return cti_image.Region((region.y0, region.y1, region.x1 + columns[0], region.x1 + columns[1]))

    @staticmethod
    def serial_ci_region_and_trails(region, image_shape, from_column):
        return cti_image.Region((region.y0, region.y1, from_column + region.x0, image_shape[1]))


class CIQuadGeometryEuclidBR(cti_image.QuadGeometryEuclidBR, CIQuadGeometry):

    def __init__(self):
        """This class represents the quadrant geometry of a Euclid charge injection image in the bottom-right of a \
        CCD (see **QuadGeometryEuclid** for a description of the Euclid CCD / FPA)"""
        super(CIQuadGeometryEuclidBR, self).__init__()

    @staticmethod
    def parallel_front_edge_region(region, rows=(0, 1)):
        """Extract the leading edge of a charge injection region which is located in the bottom-left quadrant of a \
        Euclid CCD (see *CIPatternGeometry*)."""
        check_parallel_front_edge_size(region, rows)
        return cti_image.Region((region.y0 + rows[0], region.y0 + rows[1], region.x0, region.x1))

    @staticmethod
    def parallel_trails_region(region, rows=(0, 1)):
        """Extract the trails after a charge injection region which is located in the bottom-left quadrant of a \
        Euclid CCD (see *CIPatternGeometry* for a description of where this is extracted)."""
        return cti_image.Region((region.y1 + rows[0], region.y1 + rows[1], region.x0, region.x1))

    @staticmethod
    def parallel_side_nearest_read_out_region(region, image_shape, columns=(0, 1)):
        return cti_image.Region((0, image_shape[0], region.x1 - columns[1], region.x1 - columns[0]))

    @staticmethod
    def serial_front_edge_region(region, columns=(0, 1)):
        """Extract the leading edge of a charge injection region which is located in the bottom-left quadrant of a \
        Euclid CCD (see *CIPatternGeometry*)."""
        check_serial_front_edge_size(region, columns)
        return cti_image.Region((region.y0, region.y1, region.x1 - columns[1], region.x1 - columns[0]))

    @staticmethod
    def serial_trails_region(region, columns=(0, 1)):
        """Extract the trails after a charge injection region which is located in the bottom-left quadrant of a \
        Euclid CCD (see *CIPatternGeometry* for a description of where this is extracted)."""
        return cti_image.Region((region.y0, region.y1, region.x0 - columns[1], region.x0 - columns[0]))

    @staticmethod
    def serial_ci_region_and_trails(region, image_shape, from_column):
        return cti_image.Region((region.y0, region.y1, 0, region.x1 - from_column))


class CIQuadGeometryEuclidTL(cti_image.QuadGeometryEuclidTL, CIQuadGeometry):

    def __init__(self):
        """This class represents the quadrant geometry of a Euclid charge injection image in the top-left of a \
        CCD (see **QuadGeometryEuclid** for a description of the Euclid CCD / FPA)"""
        super(CIQuadGeometryEuclidTL, self).__init__()

    @staticmethod
    def parallel_front_edge_region(region, rows=(0, 1)):
        """Extract the leading edge of a charge injection region which is located in the top-left quadrant of a \
        Euclid CCD (see *CIPatternGeometry* for a description of where this is extracted)."""
        check_parallel_front_edge_size(region, rows)
        return cti_image.Region((region.y1 - rows[1], region.y1 - rows[0], region.x0, region.x1))

    @staticmethod
    def parallel_trails_region(region, rows=(0, 1)):
        """Extract the trails after a charge injection region which is located in the top_left quadrant of a \
        Euclid CCD (see *CIPatternGeometry* for a description of where this is extracted)."""
        return cti_image.Region((region.y0 - rows[1], region.y0 - rows[0], region.x0, region.x1))

    @staticmethod
    def parallel_side_nearest_read_out_region(region, image_shape, columns=(0, 1)):
        return cti_image.Region((0, image_shape[0], region.x0 + columns[0], region.x0 + columns[1]))

    @staticmethod
    def serial_front_edge_region(region, columns=(0, 1)):
        """Extract the leading edge of a charge injection region which is located in the bottom-left quadrant of a \
        Euclid CCD (see *CIPatternGeometry*)."""
        check_serial_front_edge_size(region, columns)
        return cti_image.Region((region.y0, region.y1, region.x0 + columns[0], region.x0 + columns[1]))

    @staticmethod
    def serial_trails_region(region, columns=(0, 1)):
        """Extract the trails after a charge injection region which is located in the bottom-left quadrant of a \
        Euclid CCD (see *CIPatternGeometry* for a description of where this is extracted)."""
        return cti_image.Region((region.y0, region.y1, region.x1 + columns[0], region.x1 + columns[1]))

    @staticmethod
    def serial_ci_region_and_trails(region, image_shape, from_column):
        return cti_image.Region((region.y0, region.y1, from_column + region.x0, image_shape[1]))


class CIQuadGeometryEuclidTR(cti_image.QuadGeometryEuclidTR, CIQuadGeometry):

    def __init__(self):
        """This class represents the quadrant geometry of a Euclid charge injection image in the top-right of a \
        CCD (see **QuadGeometryEuclid** for a description of the Euclid CCD / FPA)"""
        super(CIQuadGeometryEuclidTR, self).__init__()

    @staticmethod
    def parallel_front_edge_region(region, rows=(0, 1)):
        """Extract the leading edge of a charge injection region which is located in the top-left quadrant of a \
        Euclid CCD (see *CIPatternGeometry* for a description of where this is extracted)."""
        check_parallel_front_edge_size(region, rows)
        return cti_image.Region((region.y1 - rows[1], region.y1 - rows[0], region.x0, region.x1))

    @staticmethod
    def parallel_trails_region(region, rows=(0, 1)):
        """Extract the trails after a charge injection region which is located in the top_left quadrant of a \
        Euclid CCD (see *CIPatternGeometry* for a description of where this is extracted)."""
        return cti_image.Region((region.y0 - rows[1], region.y0 - rows[0], region.x0, region.x1))

    @staticmethod
    def parallel_side_nearest_read_out_region(region, image_shape, columns=(0, 1)):
        return cti_image.Region((0, image_shape[0], region.x1 - columns[1], region.x1 - columns[0]))

    @staticmethod
    def serial_front_edge_region(region, columns=(0, 1)):
        """Extract the leading edge of a charge injection region which is located in the bottom-left quadrant of a \
        Euclid CCD (see *CIPatternGeometry*)."""
        check_serial_front_edge_size(region, columns)
        return cti_image.Region((region.y0, region.y1, region.x1 - columns[1], region.x1 - columns[0]))

    @staticmethod
    def serial_trails_region(region, columns=(0, 1)):
        """Extract the trails after a charge injection region which is located in the bottom-left quadrant of a \
        Euclid CCD (see *CIPatternGeometry* for a description of where this is extracted)."""
        return cti_image.Region((region.y0, region.y1, region.x0 - columns[1], region.x0 - columns[0]))

    @staticmethod
    def serial_ci_region_and_trails(region, image_shape, from_column):
        return cti_image.Region((region.y0, region.y1, 0, region.x1 - from_column))


def check_parallel_front_edge_size(region, rows):
    if rows[0] < 0 or rows[1] < 1 or rows[1] > region.y1 - region.y0 or rows[0] >= rows[1]:
        raise exc.CIPatternException('The number of rows to extract from the leading edge is bigger than the entire'
                                     'ci ci_region')


def check_serial_front_edge_size(region, columns):
    if columns[0] < 0 or columns[1] < 1 or columns[1] > region.x1 - region.x0 or columns[0] >= columns[1]:
        raise exc.CIPatternException('The number of columns to extract from the leading edge is bigger than the entire'
                                     'ci ci_region')
