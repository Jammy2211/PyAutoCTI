import numpy as np

from autoarray.util import array_util
from autocti.structures.frame import AbstractFrame, Frame, Region
from autocti.charge_injection.ci_frame import AbstractCIFrame, CIFrame
from autocti.util import rotate_util


class MaskedFrame(AbstractFrame):
    @classmethod
    def manual(
        cls,
        array,
        mask,
        roe_corner=(1, 0),
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
    ):
        """Abstract class for the geometry of a CTI Image.

        A FrameArray is stored as a 2D NumPy arrays. When this immage is passed to arctic, clocking goes towards
        the 'top' of the NumPy arrays (e.g. towards row 0). Trails therefore appear towards the 'bottom' of the arrays
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

        if type(array) is list:
            array = np.asarray(array)

        array = rotate_util.rotate_array_from_roe_corner(
            array=array, roe_corner=roe_corner
        )
        mask = rotate_util.rotate_array_from_roe_corner(
            array=mask, roe_corner=roe_corner
        )

        array[mask == True] = 0.0

        return Frame(
            array=array,
            mask=mask,
            original_roe_corner=roe_corner,
            parallel_overscan=rotate_util.rotate_region_from_roe_corner(
                region=parallel_overscan, shape_2d=array.shape, roe_corner=roe_corner
            ),
            serial_prescan=rotate_util.rotate_region_from_roe_corner(
                region=serial_prescan, shape_2d=array.shape, roe_corner=roe_corner
            ),
            serial_overscan=rotate_util.rotate_region_from_roe_corner(
                region=serial_overscan, shape_2d=array.shape, roe_corner=roe_corner
            ),
        )

    @classmethod
    def full(
        cls,
        fill_value,
        mask,
        roe_corner=(1, 0),
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
    ):

        return cls.manual(
            array=np.full(fill_value=fill_value, shape=mask.shape_2d),
            roe_corner=roe_corner,
            mask=mask,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

    @classmethod
    def ones(
        cls,
        mask,
        roe_corner=(1, 0),
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
    ):
        return cls.full(
            fill_value=1.0,
            mask=mask,
            roe_corner=roe_corner,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

    @classmethod
    def zeros(
        cls,
        mask,
        roe_corner=(1, 0),
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
    ):
        return cls.full(
            fill_value=0.0,
            mask=mask,
            roe_corner=roe_corner,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

    @classmethod
    def from_fits(
        cls,
        file_path,
        hdu,
        mask,
        roe_corner=(1, 0),
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
    ):
        """Load the image ci_data from a fits file.

        Params
        ----------
        path : str
            The path to the ci_data
        filename : str
            The file phase_name of the fits image ci_data.
        hdu : int
            The HDU number in the fits file containing the image ci_data.
        frame_geometry : FrameArray.FrameGeometry
            The geometry of the ci_frame, defining the direction of parallel and serial clocking and the \
            locations of different regions of the CCD (overscans, prescan, etc.)
        """
        return cls.manual(
            array=array_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu),
            mask=mask,
            roe_corner=roe_corner,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

    @classmethod
    def from_frame(cls, frame, mask):
        return Frame(
            array=frame,
            mask=mask,
            original_roe_corner=frame.original_roe_corner,
            parallel_overscan=frame.parallel_overscan,
            serial_prescan=frame.serial_prescan,
            serial_overscan=frame.serial_overscan,
        )


class MaskedEuclidFrame(AbstractFrame):
    @classmethod
    def ccd_and_quadrant_id(cls, array, mask, ccd_id, quad_id):
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

        if (row_index in "123") and (quad_id == "E"):
            return MaskedEuclidFrame.bottom_left(array=array, mask=mask)
        elif (row_index in "123") and (quad_id == "F"):
            return MaskedEuclidFrame.bottom_right(array=array, mask=mask)
        elif (row_index in "123") and (quad_id == "G"):
            return MaskedEuclidFrame.top_right(array=array, mask=mask)
        elif (row_index in "123") and (quad_id == "H"):
            return MaskedEuclidFrame.top_left(array=array, mask=mask)
        elif (row_index in "456") and (quad_id == "E"):
            return MaskedEuclidFrame.top_right(array=array, mask=mask)
        elif (row_index in "456") and (quad_id == "F"):
            return MaskedEuclidFrame.top_left(array=array, mask=mask)
        elif (row_index in "456") and (quad_id == "G"):
            return MaskedEuclidFrame.bottom_left(array=array, mask=mask)
        elif (row_index in "456") and (quad_id == "H"):
            return MaskedEuclidFrame.bottom_right(array=array, mask=mask)

    @classmethod
    def top_left(cls, array, mask):
        return MaskedFrame.manual(
            array=array,
            mask=mask,
            roe_corner=(0, 0),
            parallel_overscan=Region((0, 20, 51, 2099)),
            serial_prescan=Region((0, 2086, 0, 51)),
            serial_overscan=Region((0, 2086, 2099, 2119)),
        )

    @classmethod
    def top_right(cls, array, mask):
        return MaskedFrame.manual(
            array=array,
            mask=mask,
            roe_corner=(0, 1),
            parallel_overscan=Region((0, 20, 20, 2068)),
            serial_prescan=Region((0, 2086, 2068, 2119)),
            serial_overscan=Region((0, 2086, 0, 20)),
        )

    @classmethod
    def bottom_left(cls, array, mask):
        return MaskedFrame.manual(
            array=array,
            mask=mask,
            roe_corner=(1, 0),
            parallel_overscan=Region((2066, 2086, 51, 2099)),
            serial_prescan=Region((0, 2086, 0, 51)),
            serial_overscan=Region((0, 2086, 2099, 2119)),
        )

    @classmethod
    def bottom_right(cls, array, mask):
        return MaskedFrame.manual(
            array=array,
            mask=mask,
            roe_corner=(1, 1),
            parallel_overscan=Region((2066, 2086, 20, 2068)),
            serial_prescan=Region((0, 2086, 2068, 2119)),
            serial_overscan=Region((0, 2086, 0, 20)),
        )


class MaskedCIFrame(AbstractCIFrame):
    @classmethod
    def manual(
        cls,
        array,
        mask,
        ci_pattern,
        roe_corner=(1, 0),
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
    ):
        """Abstract class for the geometry of a CTI Image.

        A FrameArray is stored as a 2D NumPy arrays. When this immage is passed to arctic, clocking goes towards
        the 'top' of the NumPy arrays (e.g. towards row 0). Trails therefore appear towards the 'bottom' of the arrays
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

        if type(array) is list:
            array = np.asarray(array)

        array = rotate_util.rotate_array_from_roe_corner(
            array=array, roe_corner=roe_corner
        )
        mask = rotate_util.rotate_array_from_roe_corner(
            array=mask, roe_corner=roe_corner
        )

        array[mask == True] = 0.0

        return CIFrame(
            array=array,
            mask=mask,
            ci_pattern=rotate_util.rotate_ci_pattern_from_roe_corner(
                ci_pattern=ci_pattern, shape_2d=array.shape, roe_corner=roe_corner
            ),
            original_roe_corner=roe_corner,
            parallel_overscan=rotate_util.rotate_region_from_roe_corner(
                region=parallel_overscan, shape_2d=array.shape, roe_corner=roe_corner
            ),
            serial_prescan=rotate_util.rotate_region_from_roe_corner(
                region=serial_prescan, shape_2d=array.shape, roe_corner=roe_corner
            ),
            serial_overscan=rotate_util.rotate_region_from_roe_corner(
                region=serial_overscan, shape_2d=array.shape, roe_corner=roe_corner
            ),
        )

    @classmethod
    def full(
        cls,
        fill_value,
        mask,
        ci_pattern,
        roe_corner=(1, 0),
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
    ):

        return cls.manual(
            array=np.full(fill_value=fill_value, shape=mask.shape_2d),
            ci_pattern=ci_pattern,
            roe_corner=roe_corner,
            mask=mask,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

    @classmethod
    def ones(
        cls,
        mask,
        ci_pattern,
        roe_corner=(1, 0),
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
    ):
        return cls.full(
            fill_value=1.0,
            ci_pattern=ci_pattern,
            mask=mask,
            roe_corner=roe_corner,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

    @classmethod
    def zeros(
        cls,
        mask,
        ci_pattern,
        roe_corner=(1, 0),
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
    ):
        return cls.full(
            fill_value=0.0,
            ci_pattern=ci_pattern,
            mask=mask,
            roe_corner=roe_corner,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

    @classmethod
    def from_fits(
        cls,
        file_path,
        hdu,
        mask,
        ci_pattern,
        roe_corner=(1, 0),
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
    ):
        """Load the image ci_data from a fits file.

        Params
        ----------
        path : str
            The path to the ci_data
        filename : str
            The file phase_name of the fits image ci_data.
        hdu : int
            The HDU number in the fits file containing the image ci_data.
        frame_geometry : FrameArray.FrameGeometry
            The geometry of the ci_frame, defining the direction of parallel and serial clocking and the \
            locations of different regions of the CCD (overscans, prescan, etc.)
        """
        return cls.manual(
            array=array_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu),
            mask=mask,
            ci_pattern=ci_pattern,
            roe_corner=roe_corner,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

    @classmethod
    def from_ci_frame(cls, ci_frame, mask):
        return CIFrame(
            array=ci_frame,
            mask=mask,
            ci_pattern=ci_frame.ci_pattern,
            original_roe_corner=ci_frame.original_roe_corner,
            parallel_overscan=ci_frame.parallel_overscan,
            serial_prescan=ci_frame.serial_prescan,
            serial_overscan=ci_frame.serial_overscan,
        )
