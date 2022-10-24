from autoarray.layout.layout import Layout2D
from autoarray.layout.region import Region2D


class Layout2DEuclid(Layout2D):
    @classmethod
    def from_fits_header(cls, ext_header):
        """
        Use an input array of a Euclid quadrant and its corresponding .fits file header to rotate the quadrant to
        the correct orientation for arCTIc clocking.

        See the docstring of the `Array2DEuclid` class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """

        ccd_id = ext_header["CCDID"]
        quadrant_id = ext_header["QUADID"]

        parallel_overscan_size = ext_header.get("OVRSCANY", default=None)
        if parallel_overscan_size is None:
            parallel_overscan_size = 0
        serial_overscan_size = ext_header.get("OVRSCANX", default=None)
        serial_prescan_size = ext_header.get("PRESCANX", default=None)
        serial_size = ext_header.get("NAXIS1", default=None)
        parallel_size = ext_header.get("NAXIS2", default=None)

        return cls.from_ccd_and_quadrant_id(
            ccd_id=ccd_id,
            quadrant_id=quadrant_id,
            parallel_size=parallel_size,
            serial_size=serial_size,
            parallel_overscan_size=parallel_overscan_size,
            serial_prescan_size=serial_prescan_size,
            serial_overscan_size=serial_overscan_size,
        )

    @classmethod
    def from_ccd_and_quadrant_id(
        cls,
        ccd_id: str,
        quadrant_id: str,
        parallel_size: int = 2086,
        serial_size: int = 2128,
        serial_prescan_size: int = 51,
        serial_overscan_size: int = 29,
        parallel_overscan_size: int = 20,
    ):
        """
        Use an input array of a Euclid quadrant, its ccd_id and quadrant_id  to rotate the quadrant to
        the correct orientation for arCTIc clocking.

        See the docstring of the `Array2DEuclid` class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """

        row_index = ccd_id[-1]

        if (row_index in "123") and (quadrant_id == "E"):
            return Layout2DEuclid.bottom_left(
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "123") and (quadrant_id == "F"):
            return Layout2DEuclid.bottom_right(
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "123") and (quadrant_id == "G"):
            return Layout2DEuclid.top_right(
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "123") and (quadrant_id == "H"):
            return Layout2DEuclid.top_left(
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "456") and (quadrant_id == "E"):
            return Layout2DEuclid.top_right(
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "456") and (quadrant_id == "F"):
            return Layout2DEuclid.top_left(
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "456") and (quadrant_id == "G"):
            return Layout2DEuclid.bottom_left(
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "456") and (quadrant_id == "H"):
            return Layout2DEuclid.bottom_right(
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )

    @classmethod
    def top_left(
        cls,
        parallel_size: int = 2086,
        serial_size: int = 2128,
        serial_prescan_size: int = 51,
        serial_overscan_size: int = 29,
        parallel_overscan_size: int = 20,
    ):
        """
        Load the layout of a Euclid quadrant where the read out electronics (ROE) are at the top left of the CCD.

        When the corresponding array of data is loaded, it is rotated to ensure that parallel and serial CTI clocking
        go in the correct direction for arctic clocking.

        This function first species the layout regions on the unrotated image array (e.g. the parallel overscan indexes
        are defined at the bottom of the data). The layout is updated to account for this rotation (e.g. for
        a top left CCD the parallel overscan indexes are flipped to the top of the array).

        ROE's in the `top_left` correspond to 1/4 of the 144 quadrants on a Euclid FPA, specifically:

        - Those in the top half of the FPA (CCDID = 1, 2 or 3) and quadrant id H.
        - Those in the bottom half of the FPA (CCDID = 4, 5 or 6) and quadrant id F.

        The data is rotated by a flip-up-down, such that the ROE electrons are in the bottom left corner after rotation.

        See the docstring of the `Array2DEuclid` class for a complete description of the Euclid FPA, quadrants and
        rotations.

        Parameters
        ----------
        array_electrons
            The array of data (in units of electrons) which has the ROE in the top left and is rotated for arctic
            clocking.
        ccd_id
            The ID of the euclid CCD (1, 2, 3, 4, 5, or 6) indicating which half of the FP the CCD is in.
        quadrant_id
            The ID of the quadrant (E, G, F or H).
        """
        if parallel_overscan_size > 0:

            parallel_overscan = Region2D(
                (
                    0,
                    parallel_overscan_size,
                    serial_prescan_size,
                    serial_size - serial_overscan_size,
                )
            )

        else:

            parallel_overscan = None

        serial_prescan = Region2D((0, parallel_size, 0, serial_prescan_size))
        serial_overscan = Region2D(
            (
                0,
                parallel_size - parallel_overscan_size,
                serial_size - serial_overscan_size,
                serial_size,
            )
        )

        layout_2d = Layout2DEuclid(
            shape_2d=(parallel_size, serial_size),
            original_roe_corner=(0, 0),
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

        return layout_2d.new_rotated_from(roe_corner=(0, 0))

    @classmethod
    def top_right(
        cls,
        parallel_size: int = 2086,
        serial_size: int = 2128,
        serial_prescan_size: int = 51,
        serial_overscan_size: int = 29,
        parallel_overscan_size: int = 20,
    ):
        """
        Load the layout of a Euclid quadrant where the read out electronics (ROE) are at the top right of the CCD.

        When the corresponding array of data is loaded, it is rotated to ensure that parallel and serial CTI clocking
        go in the correct direction for arctic clocking.

        This function first species the layout regions on the unrotated image array (e.g. the serial prescan indexes
        are defined on the right-hand size of the data). The layout is updated to account for this rotation (e.g. the
        serial prescan indexes now become defined on the left-hand side).

        ROE's in the `top_right` correspond to 1/4 of the 144 quadrants on a Euclid FPA, specifically:

        - Those in the top half of the FPA (CCDID = 1, 2 or 3) and quadrant id G.
        - Those in the bottom half of the FPA (CCDID = 4, 5 or 6) and quadrant id E.

        The data is rotated by a flip-up-down and flip left-right, such that the ROE electrons are in the bottom left
        corner after rotation.

        See the docstring of the `Array2DEuclid` class for a complete description of the Euclid FPA, quadrants and
        rotations.

        Parameters
        ----------
        parallel_size
            The size (number of pixels) of the array across the parallel direction.
        serial_size
            The size (number of pixels) of the array across the serial direction.
        serial_prescan_size
            The size (number of pixels) of the serial prescan.
        serial_overscan_size
            The size (number of pixels) of the serial overscan.
        parallel_overscan_size
            The size (number of pixels) of the parallel overscan.
        """
        if parallel_overscan_size > 0:

            parallel_overscan = Region2D(
                (
                    0,
                    parallel_overscan_size,
                    serial_overscan_size,
                    serial_size - serial_prescan_size,
                )
            )

        else:

            parallel_overscan = None

        serial_prescan = Region2D(
            (0, parallel_size, serial_size - serial_prescan_size, serial_size)
        )
        serial_overscan = Region2D(
            (0, parallel_size - parallel_overscan_size, 0, serial_overscan_size)
        )

        layout_2d = Layout2DEuclid(
            shape_2d=(parallel_size, serial_size),
            original_roe_corner=(0, 1),
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

        return layout_2d.new_rotated_from(roe_corner=(0, 1))

    @classmethod
    def bottom_left(
        cls,
        parallel_size: int = 2086,
        serial_size: int = 2128,
        serial_prescan_size: int = 51,
        serial_overscan_size: int = 29,
        parallel_overscan_size: int = 20,
    ):
        """
        Load the layout of a Euclid quadrant where the read out electronics (ROE) are at the bottom left of the CCD.

        When the corresponding array of data is loaded, it is rotated to ensure that parallel and serial CTI clocking
        go in the correct direction for arctic clocking.

        This function first species the layout regions on the unrotated image array (e.g. the serial prescan indexes
        are defined on the left-hand size of the data). The layout is updated to account for this rotation (e.g. for
        a bottom left CCD no rotation is performed, thus the serial prescan indexes stay on the left-hand side).

        ROE's in the `bottom left` correspond to 1/4 of the 144 quadrants on a Euclid FPA, specifically:

        - Those in the top half of the FPA (CCDID = 1, 2 or 3) and quadrant id E.
        - Those in the bottom half of the FPA (CCDID = 4, 5 or 6) and quadrant id G.

        No rotation is performed for arCTIc clocking because the code assume the ROE are at the bottom left.

        See the docstring of the `Array2DEuclid` class for a complete description of the Euclid FPA, quadrants and
        rotations.

        Parameters
        ----------
        parallel_size
            The size (number of pixels) of the array across the parallel direction.
        serial_size
            The size (number of pixels) of the array across the serial direction.
        serial_prescan_size
            The size (number of pixels) of the serial prescan.
        serial_overscan_size
            The size (number of pixels) of the serial overscan.
        parallel_overscan_size
            The size (number of pixels) of the parallel overscan.
        """
        if parallel_overscan_size > 0:

            parallel_overscan = Region2D(
                (
                    parallel_size - parallel_overscan_size,
                    parallel_size,
                    serial_prescan_size,
                    serial_size - serial_overscan_size,
                )
            )

        else:

            parallel_overscan = None

        serial_prescan = Region2D((0, parallel_size, 0, serial_prescan_size))
        serial_overscan = Region2D(
            (
                0,
                parallel_size - parallel_overscan_size,
                serial_size - serial_overscan_size,
                serial_size,
            )
        )

        layout_2d = Layout2DEuclid(
            shape_2d=(parallel_size, serial_size),
            original_roe_corner=(1, 0),
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

        return layout_2d.new_rotated_from(roe_corner=(1, 0))

    @classmethod
    def bottom_right(
        cls,
        parallel_size: int = 2086,
        serial_size: int = 2128,
        serial_prescan_size: int = 51,
        serial_overscan_size: int = 29,
        parallel_overscan_size: int = 20,
    ):
        """
        Load the layout of a Euclid quadrant where the read out electronics (ROE) are at the bottom right of the CCD.

        When the corresponding array of data is loaded, it is rotated to ensure that parallel and serial CTI clocking
        go in the correct direction for arctic clocking.

        This function first species the layout regions on the unrotated image array (e.g. the serial prescan indexes
        are defined on the right-hand size of the data). The layout is updated to account for this rotation (e.g. the
        serial prescan indexes now become defined on the left-hand side).

        ROE's in the `bottom right` correspond to 1/4 of the 144 quadrants on a Euclid FPA, specifically:

        - Those in the top half of the FPA (CCDID = 1, 2 or 3) and quadrant id F.
        - Those in the bottom half of the FPA (CCDID = 4, 5 or 6) and quadrant id H.

        The data and layout coordinates are rotated by a flip left-right, such that the ROE electrons are in the
        bottom left corner after rotation.

        See the docstring of the `Array2DEuclid` class for a complete description of the Euclid FPA, quadrants and
        rotations.

        Parameters
        ----------
        parallel_size
            The size (number of pixels) of the array across the parallel direction.
        serial_size
            The size (number of pixels) of the array across the serial direction.
        serial_prescan_size
            The size (number of pixels) of the serial prescan.
        serial_overscan_size
            The size (number of pixels) of the serial overscan.
        parallel_overscan_size
            The size (number of pixels) of the parallel overscan.
        """
        if parallel_overscan_size > 0:

            parallel_overscan = Region2D(
                (
                    parallel_size - parallel_overscan_size,
                    parallel_size,
                    serial_overscan_size,
                    serial_size - serial_prescan_size,
                )
            )

        else:

            parallel_overscan = None

        serial_prescan = Region2D(
            (0, parallel_size, serial_size - serial_prescan_size, serial_size)
        )
        serial_overscan = Region2D(
            (0, parallel_size - parallel_overscan_size, 0, serial_overscan_size)
        )

        layout_2d = Layout2DEuclid(
            shape_2d=(parallel_size, serial_size),
            original_roe_corner=(1, 1),
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

        return layout_2d.new_rotated_from(roe_corner=(1, 1))
