import numpy as np

from autoarray.structures import abstract_structure
from autocti.structures.frame import abstract_frame
from autocti.mask import mask as msk
from autocti.util import array_util
from autocti.util import frame_util


class Frame(abstract_frame.AbstractFrame):
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

        pixel_scales = abstract_structure.convert_pixel_scales(
            pixel_scales=pixel_scales
        )

        mask = msk.Mask.unmasked(shape_2d=array.shape, pixel_scales=pixel_scales)

        return Frame(
            array=frame_util.rotate_array_from_roe_corner(
                array=array, roe_corner=roe_corner
            ),
            mask=mask,
            original_roe_corner=roe_corner,
            parallel_overscan=frame_util.rotate_region_from_roe_corner(
                region=parallel_overscan, shape_2d=array.shape, roe_corner=roe_corner
            ),
            serial_prescan=frame_util.rotate_region_from_roe_corner(
                region=serial_prescan, shape_2d=array.shape, roe_corner=roe_corner
            ),
            serial_overscan=frame_util.rotate_region_from_roe_corner(
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
    def extracted_frame_from_frame_and_extraction_region(cls, frame, extraction_region):

        return cls.manual(
            array=frame[extraction_region.slice],
            parallel_overscan=frame_util.region_after_extraction(
                original_region=frame.parallel_overscan,
                extraction_region=extraction_region,
            ),
            serial_prescan=frame_util.region_after_extraction(
                original_region=frame.serial_prescan,
                extraction_region=extraction_region,
            ),
            serial_overscan=frame_util.region_after_extraction(
                original_region=frame.serial_overscan,
                extraction_region=extraction_region,
            ),
            roe_corner=frame.original_roe_corner,
            pixel_scales=frame.pixel_scales,
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

        pixel_scales = abstract_structure.convert_pixel_scales(
            pixel_scales=pixel_scales
        )

        array = array_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu)

        return cls.manual(
            array=array,
            roe_corner=roe_corner,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
            pixel_scales=pixel_scales,
        )


class MaskedFrame(abstract_frame.AbstractFrame):
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

        array = frame_util.rotate_array_from_roe_corner(
            array=array, roe_corner=roe_corner
        )
        mask = frame_util.rotate_array_from_roe_corner(
            array=mask, roe_corner=roe_corner
        )

        array[mask == True] = 0.0

        return Frame(
            array=array,
            mask=mask,
            original_roe_corner=roe_corner,
            parallel_overscan=frame_util.rotate_region_from_roe_corner(
                region=parallel_overscan, shape_2d=array.shape, roe_corner=roe_corner
            ),
            serial_prescan=frame_util.rotate_region_from_roe_corner(
                region=serial_prescan, shape_2d=array.shape, roe_corner=roe_corner
            ),
            serial_overscan=frame_util.rotate_region_from_roe_corner(
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
