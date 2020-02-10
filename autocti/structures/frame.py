import numpy as np

from autoarray.util import array_util
from autoarray.structures import arrays
from autoarray.mask import mask as msk
from autocti.util import rotate_util
from autocti.structures import region as reg


class AbstractFrame(arrays.AbstractArray):
    def __new__(
        cls,
        array,
        mask,
        original_roe_corner=(1, 0),
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
        parallel_overscan : frame.Region
            The parallel overscan region of the ci_frame.
        serial_prescan : frame.Region
            The serial prescan region of the ci_frame.
        serial_overscan : frame.Region
            The serial overscan region of the ci_frame.
        """

        if type(array) is list:
            array = np.asarray(array)

        if isinstance(parallel_overscan, tuple):
            parallel_overscan = reg.Region(region=parallel_overscan)

        if isinstance(serial_prescan, tuple):
            serial_prescan = reg.Region(region=serial_prescan)

        if isinstance(serial_overscan, tuple):
            serial_overscan = reg.Region(region=serial_overscan)

        array[mask == True] = 0.0

        obj = super(AbstractFrame, cls).__new__(
            cls=cls, array=array, mask=mask, store_in_1d=False
        )

        obj.original_roe_corner = original_roe_corner
        obj.parallel_overscan = parallel_overscan
        obj.serial_prescan = serial_prescan
        obj.serial_overscan = serial_overscan

        return obj

    def __array_finalize__(self, obj):

        super(AbstractFrame, self).__array_finalize__(obj)

        if isinstance(obj, AbstractFrame):
            if hasattr(obj, "roe_corner"):
                self.roe_corner = obj.roe_corner

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(AbstractFrame, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        class_dict = {}
        for key, value in self.__dict__.items():
            class_dict[key] = value
        new_state = pickled_state[2] + (class_dict,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return pickled_state[0], pickled_state[1], new_state

    # noinspection PyMethodOverriding
    def __setstate__(self, state):

        for key, value in state[-1].items():
            setattr(self, key, value)
        super(AbstractFrame, self).__setstate__(state[0:-1])

    @property
    def binned_across_parallel(self):
        return np.mean(np.ma.masked_array(self, self.mask), axis=0)

    @property
    def binned_across_serial(self):
        return np.mean(np.ma.masked_array(self, self.mask), axis=1)

    def parallel_trail_from_y(self, y, dy):
        """Coordinates of a parallel trail of size dy from coordinate y"""
        return (int(y - dy), int(y + 1))

    def serial_trail_from_x(self, x, dx):
        """Coordinates of a serial trail of size dx from coordinate x"""
        return (int(x), int(x + 1 + dx))

    def parallel_front_edge_of_region(self, region, rows):

        reg.check_parallel_front_edge_size(region=region, rows=rows)

        y_coord = region.y0
        y_min = y_coord + rows[0]
        y_max = y_coord + rows[1]

        return reg.Region((y_min, y_max, region.x0, region.x1))

    def parallel_trails_of_region(self, region, rows=(0, 1)):
        y_coord = region.y1
        y_min = y_coord + rows[0]
        y_max = y_coord + rows[1]
        return reg.Region((y_min, y_max, region.x0, region.x1))

    def parallel_side_nearest_read_out_region(self, region, columns=(0, 1)):
        x_min, x_max = self.x_limits(region, columns)
        return reg.Region(region=(0, self.shape_2d[0], x_min, x_max))

    def serial_front_edge_of_region(self, region, columns=(0, 1)):
        reg.check_serial_front_edge_size(region, columns)
        x_min, x_max = self.x_limits(region, columns)
        return reg.Region(region=(region.y0, region.y1, x_min, x_max))

    def serial_trails_of_region(self, region, columns=(0, 1)):
        x_coord = region.x1
        x_min = x_coord + columns[0]
        x_max = x_coord + columns[1]
        return reg.Region(region=(region.y0, region.y1, x_min, x_max))

    def serial_entire_rows_of_region(self, region):
        return reg.Region(region=(region.y0, region.y1, 0, self.shape_2d[1]))

    @property
    def serial_trails_columns(self):
        return self.serial_overscan[3] - self.serial_overscan[2]

    def x_limits(self, region, columns):
        x_coord = region.x0
        x_min = x_coord + columns[0]
        x_max = x_coord + columns[1]
        return x_min, x_max


class Frame(AbstractFrame):
    @classmethod
    def manual(
        cls,
        array,
        roe_corner=(1, 0),
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
        pixel_scales=None,
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
        parallel_overscan : frame.Region
            The parallel overscan region of the ci_frame.
        serial_prescan : frame.Region
            The serial prescan region of the ci_frame.
        serial_overscan : frame.Region
            The serial overscan region of the ci_frame.
        """

        if type(array) is list:
            array = np.asarray(array)

        if type(pixel_scales) is float:
            pixel_scales = (pixel_scales, pixel_scales)

        mask = msk.Mask.unmasked(shape_2d=array.shape, pixel_scales=pixel_scales)

        return Frame(
            array=rotate_util.rotate_array_from_roe_corner(
                array=array, roe_corner=roe_corner
            ),
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
        shape_2d,
        roe_corner=(1, 0),
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
        pixel_scales=None,
    ):

        return cls.manual(
            array=np.full(fill_value=fill_value, shape=shape_2d),
            roe_corner=roe_corner,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
            pixel_scales=pixel_scales,
        )

    @classmethod
    def ones(
        cls,
        shape_2d,
        roe_corner=(1, 0),
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
        pixel_scales=None,
    ):
        return cls.full(
            fill_value=1.0,
            shape_2d=shape_2d,
            roe_corner=roe_corner,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
            pixel_scales=pixel_scales,
        )

    @classmethod
    def zeros(
        cls,
        shape_2d,
        roe_corner=(1, 0),
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
        pixel_scales=None,
    ):
        return cls.full(
            fill_value=0.0,
            shape_2d=shape_2d,
            roe_corner=roe_corner,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
            pixel_scales=pixel_scales,
        )

    @classmethod
    def from_fits(
        cls,
        file_path,
        hdu,
        roe_corner=(1, 0),
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
        pixel_scales=None,
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

        if type(pixel_scales) is float:
            pixel_scales = (pixel_scales, pixel_scales)

        array = array_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu)

        mask = msk.Mask.unmasked(shape_2d=array.shape, pixel_scales=pixel_scales)

        return cls.manual(
            array=array,
            mask=mask,
            roe_corner=roe_corner,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
            pixel_scales=pixel_scales,
        )


class EuclidFrame(Frame):
    @classmethod
    def ccd_and_quadrant_id(cls, array, ccd_id, quad_id):
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
            return EuclidFrame.bottom_left(array=array)
        elif (row_index in "123") and (quad_id == "F"):
            return EuclidFrame.bottom_right(array=array)
        elif (row_index in "123") and (quad_id == "G"):
            return EuclidFrame.top_right(array=array)
        elif (row_index in "123") and (quad_id == "H"):
            return EuclidFrame.top_left(array=array)
        elif (row_index in "456") and (quad_id == "E"):
            return EuclidFrame.top_right(array=array)
        elif (row_index in "456") and (quad_id == "F"):
            return EuclidFrame.top_left(array=array)
        elif (row_index in "456") and (quad_id == "G"):
            return EuclidFrame.bottom_left(array=array)
        elif (row_index in "456") and (quad_id == "H"):
            return EuclidFrame.bottom_right(array=array)

    @classmethod
    def top_left(cls, array):
        return Frame.manual(
            array=array,
            roe_corner=(0, 0),
            parallel_overscan=reg.Region((0, 20, 51, 2099)),
            serial_prescan=reg.Region((0, 2086, 0, 51)),
            serial_overscan=reg.Region((0, 2086, 2099, 2119)),
        )

    @classmethod
    def top_right(cls, array):
        return Frame.manual(
            array=array,
            roe_corner=(0, 1),
            parallel_overscan=reg.Region((0, 20, 20, 2068)),
            serial_prescan=reg.Region((0, 2086, 2068, 2119)),
            serial_overscan=reg.Region((0, 2086, 0, 20)),
        )

    @classmethod
    def bottom_left(cls, array):
        return Frame.manual(
            array=array,
            roe_corner=(1, 0),
            parallel_overscan=reg.Region((2066, 2086, 51, 2099)),
            serial_prescan=reg.Region((0, 2086, 0, 51)),
            serial_overscan=reg.Region((0, 2086, 2099, 2119)),
        )

    @classmethod
    def bottom_right(cls, array):
        return Frame.manual(
            array=array,
            roe_corner=(1, 1),
            parallel_overscan=reg.Region((2066, 2086, 20, 2068)),
            serial_prescan=reg.Region((0, 2086, 2068, 2119)),
            serial_overscan=reg.Region((0, 2086, 0, 20)),
        )
