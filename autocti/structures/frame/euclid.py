from autocti.structures.frame import abstract_frame
from autocti.structures import frame as f
from autocti.structures import region as reg


class EuclidFrame(f.Frame):
    @classmethod
    def from_fits_header(cls, array, ext_header):
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
                                                                       | of the NumPy arrays)
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
                                                                       | of the NumPy arrays)
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

        ccd_id = ext_header["CCDID"]
        quadrant_id = ext_header["QUADID"]

        parallel_overscan_size = ext_header.get("PAROVRX", default=None)
        if parallel_overscan_size is None:
            parallel_overscan_size = 0
        serial_overscan_size = ext_header.get("OVRSCANX", default=None)
        serial_prescan_size = ext_header.get("PRESCANX", default=None)
        serial_size = ext_header.get("NAXIS1", default=None)
        parallel_size = ext_header.get("NAXIS2", default=None)

        return cls.from_ccd_and_quadrant_id(
            array=array,
            ccd_id=ccd_id,
            quadrant_id=quadrant_id,
            parallel_size=parallel_size,
            serial_size=serial_size,
            serial_prescan_size=serial_prescan_size,
            serial_overscan_size=serial_overscan_size,
            parallel_overscan_size=parallel_overscan_size,
        )

    @classmethod
    def from_ccd_and_quadrant_id(
        cls,
        array,
        ccd_id,
        quadrant_id,
        parallel_size=2086,
        serial_size=2119,
        serial_prescan_size=51,
        serial_overscan_size=20,
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
                                                                       | of the NumPy arrays)
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
                                                                       | of the NumPy arrays)
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
            return EuclidFrame.bottom_left(
                array=array,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "123") and (quadrant_id == "F"):
            return EuclidFrame.bottom_right(
                array=array,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "123") and (quadrant_id == "G"):
            return EuclidFrame.top_right(
                array=array,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "123") and (quadrant_id == "H"):
            return EuclidFrame.top_left(
                array=array,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "456") and (quadrant_id == "E"):
            return EuclidFrame.top_right(
                array=array,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "456") and (quadrant_id == "F"):
            return EuclidFrame.top_left(
                array=array,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "456") and (quadrant_id == "G"):
            return EuclidFrame.bottom_left(
                array=array,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "456") and (quadrant_id == "H"):
            return EuclidFrame.bottom_right(
                array=array,
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
        parallel_size=2086,
        serial_size=2119,
        serial_prescan_size=51,
        serial_overscan_size=20,
        parallel_overscan_size=20,
    ):

        if parallel_overscan_size > 0:

            parallel_overscan = reg.Region(
                (
                    0,
                    parallel_overscan_size,
                    serial_prescan_size,
                    serial_size - serial_overscan_size,
                )
            )

        else:

            parallel_overscan = None

        serial_prescan = reg.Region((0, parallel_size, 0, serial_prescan_size))
        serial_overscan = reg.Region(
            (
                0,
                parallel_size - parallel_overscan_size,
                serial_size - serial_overscan_size,
                serial_size,
            )
        )

        return f.Frame.manual(
            array=array,
            roe_corner=(0, 0),
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

    @classmethod
    def top_right(
        cls,
        array,
        parallel_size=2086,
        serial_size=2119,
        parallel_overscan_size=20,
        serial_prescan_size=51,
        serial_overscan_size=20,
    ):

        if parallel_overscan_size > 0:

            parallel_overscan = reg.Region(
                (
                    0,
                    parallel_overscan_size,
                    serial_overscan_size,
                    serial_size - serial_prescan_size,
                )
            )

        else:

            parallel_overscan = None

        serial_prescan = reg.Region(
            (0, parallel_size, serial_size - serial_prescan_size, serial_size)
        )
        serial_overscan = reg.Region(
            (0, parallel_size - parallel_overscan_size, 0, serial_overscan_size)
        )

        return f.Frame.manual(
            array=array,
            roe_corner=(0, 1),
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

    @classmethod
    def bottom_left(
        cls,
        array,
        parallel_size=2086,
        serial_size=2119,
        serial_prescan_size=51,
        serial_overscan_size=20,
        parallel_overscan_size=20,
    ):

        if parallel_overscan_size > 0:

            parallel_overscan = reg.Region(
                (
                    parallel_size - parallel_overscan_size,
                    parallel_size,
                    serial_prescan_size,
                    serial_size - serial_overscan_size,
                )
            )

        else:

            parallel_overscan = None

        serial_prescan = reg.Region((0, parallel_size, 0, serial_prescan_size))
        serial_overscan = reg.Region(
            (
                0,
                parallel_size - parallel_overscan_size,
                serial_size - serial_overscan_size,
                serial_size,
            )
        )

        return f.Frame.manual(
            array=array,
            roe_corner=(1, 0),
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

    @classmethod
    def bottom_right(
        cls,
        array,
        parallel_size=2086,
        serial_size=2119,
        serial_prescan_size=51,
        serial_overscan_size=20,
        parallel_overscan_size=20,
    ):

        if parallel_overscan_size > 0:

            parallel_overscan = reg.Region(
                (
                    parallel_size - parallel_overscan_size,
                    parallel_size,
                    serial_overscan_size,
                    serial_size - serial_prescan_size,
                )
            )

        else:

            parallel_overscan = None

        serial_prescan = reg.Region(
            (0, parallel_size, serial_size - serial_prescan_size, serial_size)
        )
        serial_overscan = reg.Region(
            (0, parallel_size - parallel_overscan_size, 0, serial_overscan_size)
        )

        return f.Frame.manual(
            array=array,
            roe_corner=(1, 1),
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )


class MaskedEuclidFrame(abstract_frame.AbstractFrame):
    @classmethod
    def from_ccd_and_quadrant_id(
        cls,
        array,
        mask,
        ccd_id,
        quadrant_id,
        parallel_size=2086,
        serial_size=2119,
        serial_prescan_size=51,
        serial_overscan_size=20,
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
                                                                       | of the NumPy arrays)
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
                                                                       | of the NumPy arrays)
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
            return MaskedEuclidFrame.bottom_left(
                array=array,
                mask=mask,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "123") and (quadrant_id == "F"):
            return MaskedEuclidFrame.bottom_right(
                array=array,
                mask=mask,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "123") and (quadrant_id == "G"):
            return MaskedEuclidFrame.top_right(
                array=array,
                mask=mask,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "123") and (quadrant_id == "H"):
            return MaskedEuclidFrame.top_left(
                array=array,
                mask=mask,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "456") and (quadrant_id == "E"):
            return MaskedEuclidFrame.top_right(
                array=array,
                mask=mask,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "456") and (quadrant_id == "F"):
            return MaskedEuclidFrame.top_left(
                array=array,
                mask=mask,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "456") and (quadrant_id == "G"):
            return MaskedEuclidFrame.bottom_left(
                array=array,
                mask=mask,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "456") and (quadrant_id == "H"):
            return MaskedEuclidFrame.bottom_right(
                array=array,
                mask=mask,
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
        mask,
        parallel_size=2086,
        serial_size=2119,
        serial_prescan_size=51,
        serial_overscan_size=20,
        parallel_overscan_size=20,
    ):
        return f.MaskedFrame.manual(
            array=array,
            mask=mask,
            roe_corner=(0, 0),
            parallel_overscan=reg.Region((0, 20, 51, 2099)),
            serial_prescan=reg.Region((0, 2086, 0, 51)),
            serial_overscan=reg.Region((0, 2086, 2099, 2119)),
        )

    @classmethod
    def top_right(
        cls,
        array,
        mask,
        parallel_size=2086,
        serial_size=2119,
        serial_prescan_size=51,
        serial_overscan_size=20,
        parallel_overscan_size=20,
    ):
        return f.MaskedFrame.manual(
            array=array,
            mask=mask,
            roe_corner=(0, 1),
            parallel_overscan=reg.Region((0, 20, 20, 2068)),
            serial_prescan=reg.Region((0, 2086, 2068, 2119)),
            serial_overscan=reg.Region((0, 2086, 0, 20)),
        )

    @classmethod
    def bottom_left(
        cls,
        array,
        mask,
        parallel_size=2086,
        serial_size=2119,
        serial_prescan_size=51,
        serial_overscan_size=20,
        parallel_overscan_size=20,
    ):
        return f.MaskedFrame.manual(
            array=array,
            mask=mask,
            roe_corner=(1, 0),
            parallel_overscan=reg.Region((2066, 2086, 51, 2099)),
            serial_prescan=reg.Region((0, 2086, 0, 51)),
            serial_overscan=reg.Region((0, 2086, 2099, 2119)),
        )

    @classmethod
    def bottom_right(
        cls,
        array,
        mask,
        parallel_size=2086,
        serial_size=2119,
        serial_prescan_size=51,
        serial_overscan_size=20,
        parallel_overscan_size=20,
    ):
        return f.MaskedFrame.manual(
            array=array,
            mask=mask,
            roe_corner=(1, 1),
            parallel_overscan=reg.Region((2066, 2086, 20, 2068)),
            serial_prescan=reg.Region((0, 2086, 2068, 2119)),
            serial_overscan=reg.Region((0, 2086, 0, 20)),
        )
