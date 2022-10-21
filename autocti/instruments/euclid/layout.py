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
        ccd_id,
        quadrant_id,
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=51,
        serial_overscan_size=29,
        parallel_overscan_size=20,
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
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=51,
        serial_overscan_size=29,
        parallel_overscan_size=20,
    ):
        """


        Parameters
        ----------
        parallel_size
        serial_size
        serial_prescan_size
        serial_overscan_size
        parallel_overscan_size

        Returns
        -------

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
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=51,
        serial_overscan_size=29,
        parallel_overscan_size=20,
    ):

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
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=51,
        serial_overscan_size=29,
        parallel_overscan_size=20,
    ):

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
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=51,
        serial_overscan_size=29,
        parallel_overscan_size=20,
    ):

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
