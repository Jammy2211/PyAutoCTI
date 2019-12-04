import numpy as np

from autoarray.util import array_util
from autoarray.structures import arrays
from autoarray.mask import mask as msk
from autocti import exc
from autocti.model import pyarctic


class FrameArray(arrays.AbstractArray):
    def __new__(
        cls,
        array,
        corner,
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
        parallel_overscan : ci_frame.Region
            The parallel overscan region of the ci_frame.
        serial_prescan : ci_frame.Region
            The serial prescan region of the ci_frame.
        serial_overscan : ci_frame.Region
            The serial overscan region of the ci_frame.
        """

        if type(array) is list:
            array = np.asarray(array)

        if type(pixel_scales) is float:
            pixel_scales = (pixel_scales, pixel_scales)

        mask = msk.Mask.unmasked(shape_2d=array.shape, pixel_scales=pixel_scales)

        obj = super(FrameArray, cls).__new__(cls=cls, array=array, mask=mask, store_in_1d=False)

        obj.corner = corner
        obj.parallel_overscan = parallel_overscan
        obj.serial_prescan = serial_prescan
        obj.serial_overscan = serial_overscan

        return obj

    @classmethod
    def from_fits(
        cls,
        file_path,
        hdu,
        corner,
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
        return cls(
            array=array_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu),
            corner=corner,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
            pixel_scales=pixel_scales,
        )

    def __array_finalize__(self, obj):

        super(FrameArray, self).__array_finalize__(obj)

        if isinstance(obj, FrameArray):
            self.corner = obj.corner

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(FrameArray, self).__reduce__()
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
        super(FrameArray, self).__setstate__(state[0:-1])

    def add_cti(
        self, image, cti_params, cti_settings, use_parallel_poisson_densities=False
    ):
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

        if cti_params.parallel_ccd_volume is not None:
            image_pre_parallel_clocking = self.rotated_for_parallel_cti(image=image)
            image_post_parallel_clocking = pyarctic.call_arctic(
                image=image_pre_parallel_clocking,
                species=cti_params.parallel_species,
                ccd=cti_params.parallel_ccd_volume,
                settings=cti_settings.parallel,
                correct_cti=False,
                use_poisson_densities=use_parallel_poisson_densities,
            )
            image = self.rotated_for_parallel_cti(image_post_parallel_clocking)

        if cti_params.serial_ccd_volume is not None:
            image_pre_serial_clocking = self.rotated_before_serial_clocking(
                image_pre_clocking=image
            )
            image_post_serial_clocking = pyarctic.call_arctic(
                image=image_pre_serial_clocking,
                species=cti_params.serial_species,
                ccd=cti_params.serial_ccd_volume,
                settings=cti_settings.serial,
                correct_cti=False,
                use_poisson_densities=False,
            )
            image = self.rotated_after_serial_clocking(image_post_serial_clocking)

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
            image_pre_serial_clocking = self.rotated_before_serial_clocking(
                image_pre_clocking=image
            )
            image_post_serial_clocking = pyarctic.call_arctic(
                image=image_pre_serial_clocking,
                species=cti_params.serial_species,
                ccd=cti_params.serial_ccd_volume,
                settings=cti_settings.serial,
                correct_cti=True,
                use_poisson_densities=False,
            )
            image = self.rotated_after_serial_clocking(image_post_serial_clocking)

        if cti_settings.parallel is not None:
            image_pre_parallel_clocking = self.rotated_for_parallel_cti(image=image)
            image_post_parallel_clocking = pyarctic.call_arctic(
                image=image_pre_parallel_clocking,
                species=cti_params.parallel_species,
                ccd=cti_params.parallel_ccd_volume,
                settings=cti_settings.parallel,
                correct_cti=True,
                use_poisson_densities=False,
            )
            image = self.rotated_for_parallel_cti(image_post_parallel_clocking)

        return image

    @property
    def rotated_for_parallel_cti(self):
        return flip(self.in_2d) if self.corner[0] == 1 else self.in_2d

    @property
    def rotated_before_serial_clocking(self):
        transposed = self.in_2d.T.copy()
        return flip(transposed) if self.corner[1] == 1 else transposed

    @property
    def rotated_after_serial_clocking(self):
        flipped = flip(self.in_2d) if self.corner[1] == 1 else self.in_2d
        return flipped.T.copy()

    def parallel_trail_from_y(self, y, dy):
        """Coordinates of a parallel trail of size dy from coordinate y"""
        return int(y - dy * self.corner[0]), int(y + 1 + dy * (1 - self.corner[0]))

    def serial_trail_from_x(self, x, dx):
        """Coordinates of a serial trail of size dx from coordinate x"""
        return int(x - dx * self.corner[1]), int(x + 1 + dx * (1 - self.corner[1]))

    def parallel_front_edge_region(self, ci_region, rows):

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

    def parallel_side_nearest_read_out_region(
        self, ci_region, image_shape, columns=(0, 1)
    ):
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

    def parallel_trail_size_to_image_edge(self, ci_pattern, shape):

        if self.corner[0] == 0:
            return shape[0] - np.max([region.y1 for region in ci_pattern.regions])
        else:
            return np.min([region.y0 for region in ci_pattern.regions])

    @property
    def serial_trails_columns(self):
        return self.serial_overscan[3] - self.serial_overscan[2]


class EuclidArray(FrameArray):
    @classmethod
    def euclid_parallel_line(cls):
        return EuclidArray(
            corner=(0, 0),
            parallel_overscan=Region((2066, 2086, 0, 1)),
            serial_prescan=None,
            serial_overscan=None,
        )

    @classmethod
    def euclid_serial_line(cls):
        return EuclidArray(
            corner=(0, 0),
            parallel_overscan=None,
            serial_prescan=Region((0, 1, 0, 51)),
            serial_overscan=Region((0, 1, 2099, 2119)),
        )

    @classmethod
    def euclid_from_ccd_and_quadrant_id(cls, array, ccd_id, quad_id):
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
            return EuclidArray.euclid_bottom_left(array=array)
        elif (row_index in "123") and (quad_id == "F"):
            return EuclidArray.euclid_bottom_right(array=array)
        elif (row_index in "123") and (quad_id == "G"):
            return EuclidArray.euclid_top_right(array=array)
        elif (row_index in "123") and (quad_id == "H"):
            return EuclidArray.euclid_top_left(array=array)
        elif (row_index in "456") and (quad_id == "E"):
            return EuclidArray.euclid_top_right(array=array)
        elif (row_index in "456") and (quad_id == "F"):
            return EuclidArray.euclid_top_left(array=array)
        elif (row_index in "456") and (quad_id == "G"):
            return EuclidArray.euclid_bottom_left(array=array)
        elif (row_index in "456") and (quad_id == "H"):
            return EuclidArray.euclid_bottom_right(array=array)

    @classmethod
    def euclid_bottom_left(cls, array):
        return EuclidArray(
            array=array,
            corner=(0, 0),
            parallel_overscan=Region((2066, 2086, 51, 2099)),
            serial_prescan=Region((0, 2086, 0, 51)),
            serial_overscan=Region((0, 2086, 2099, 2119)),
        )

    @classmethod
    def euclid_bottom_right(cls, array):
        return EuclidArray(
            array=array,
            corner=(0, 1),
            parallel_overscan=Region((2066, 2086, 20, 2068)),
            serial_prescan=Region((0, 2086, 2068, 2119)),
            serial_overscan=Region((0, 2086, 0, 20)),
        )

    @classmethod
    def euclid_top_left(cls, array):
        return EuclidArray(
            array=array,
            corner=(1, 0),
            parallel_overscan=Region((0, 20, 51, 2099)),
            serial_prescan=Region((0, 2086, 0, 51)),
            serial_overscan=Region((0, 2086, 2099, 2119)),
        )

    @classmethod
    def euclid_top_right(cls, array):
        return EuclidArray(
            array=array,
            corner=(1, 1),
            parallel_overscan=Region((0, 20, 20, 2068)),
            serial_prescan=Region((0, 2086, 2068, 2119)),
            serial_overscan=Region((0, 2086, 0, 20)),
        )


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
            raise exc.RegionException(
                "A coordinate of the Region was specified as negative."
            )

        if region[0] >= region[1]:
            raise exc.RegionException(
                "The first row in the Region was equal to or greater than the second row."
            )

        if region[2] >= region[3]:
            raise exc.RegionException(
                "The first column in the Region was equal to greater than the second column."
            )
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
        return np.s_[self.y0 : self.y1, self.x0 : self.x1]

    @property
    def y_slice(self):
        return np.s_[self.y0 : self.y1]

    @property
    def x_slice(self):
        return np.s_[self.x0 : self.x1]

    @property
    def shape(self):
        return self.y1 - self.y0, self.x1 - self.x0


def check_parallel_front_edge_size(region, rows):
    # TODO: are these checks important?
    if (
        rows[0] < 0
        or rows[1] < 1
        or rows[1] > region.y1 - region.y0
        or rows[0] >= rows[1]
    ):
        raise exc.CIPatternException(
            "The number of rows to extract from the leading edge is bigger than the entire"
            "ci ci_region"
        )


def check_serial_front_edge_size(region, columns):
    if (
        columns[0] < 0
        or columns[1] < 1
        or columns[1] > region.x1 - region.x0
        or columns[0] >= columns[1]
    ):
        raise exc.CIPatternException(
            "The number of columns to extract from the leading edge is bigger than the entire"
            "ci ci_region"
        )


def bin_array_across_serial(array, mask=None):
    return np.mean(np.ma.masked_array(array, mask), axis=1)


def bin_array_across_parallel(array, mask=None):
    return np.mean(np.ma.masked_array(array, mask), axis=0)


def flip(image):
    return image[::-1, :]
