import numpy as np
from autocti import exc
from autocti.structures import region as reg
from autocti.util import array_util


class Mask(np.ndarray):
    def __new__(cls, mask_2d, pixel_scales=None, origin=(0.0, 0.0), *args, **kwargs):
        """ A mask, which is applied to data to extract a set of unmasked image pixels (i.e. mask entry \
        is *False* or 0) which are then fitted in an analysis.

        The mask retains the pixel scale of the array and has a centre and origin.

        Parameters
        ----------
        mask_2d: ndarray
            An array of bools representing the mask.
        pixel_scales: (float, float)
            The arc-second to pixel conversion factor of each pixel.
        origin : (float, float)
            The (y,x) arc-second origin of the mask's coordinate system.
        centre : (float, float)
            The (y,x) arc-second centre of the mask provided it is a standard geometric shape (e.g. a circle).
        """
        # noinspection PyArgumentList

        mask_2d = mask_2d.astype("bool")
        obj = mask_2d.view(cls)
        obj.pixel_scales = pixel_scales
        obj.origin = origin
        return obj

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(Mask, self).__reduce__()
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
        super(Mask, self).__setstate__(state[0:-1])

    def __array_finalize__(self, obj):

        if isinstance(obj, Mask):
            self.pixel_scales = obj.pixel_scales
            self.origin = obj.origin
        else:
            self.origin = (0.0, 0.0)
            self.pixel_scales = None

    @classmethod
    def manual(cls, mask_2d, pixel_scales=None, origin=(0.0, 0.0), invert=False):

        if type(mask_2d) is list:
            mask_2d = np.asarray(mask_2d).astype("bool")

        if invert:
            mask_2d = np.invert(mask_2d)

        if type(pixel_scales) is float:
            pixel_scales = (pixel_scales, pixel_scales)

        if len(mask_2d.shape) != 2:
            raise exc.MaskException("The input mask_2d is not a two dimensional array")

        return Mask(mask_2d=mask_2d, pixel_scales=pixel_scales, origin=origin)

    @classmethod
    def unmasked(cls, shape_2d, pixel_scales=None, origin=(0.0, 0.0), invert=False):
        """Setup a mask where all pixels are unmasked.

        Parameters
        ----------
        shape : (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scales : float or (float, float)
            The arc-second to pixel conversion factor of each pixel.
        """
        return cls.manual(
            mask_2d=np.full(shape=shape_2d, fill_value=False),
            pixel_scales=pixel_scales,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def from_masked_regions(cls, shape_2d, masked_regions):

        mask = cls.unmasked(shape_2d=shape_2d)
        masked_regions = list(
            map(lambda region: reg.Region(region=region), masked_regions)
        )
        for region in masked_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        return mask

    @classmethod
    def from_cosmic_ray_map(
        cls,
        cosmic_ray_map,
        cosmic_ray_parallel_buffer=0,
        cosmic_ray_serial_buffer=0,
        cosmic_ray_diagonal_buffer=0,
    ):
        """
        Create the mask used for CTI Calibration, which is all False unless specific regions are input for masking.

        Parameters
        ----------
        shape_2d : (int, int)
            The dimensions of the 2D mask.
        frame_geometry : ci_frame.CIQuadGeometry
            The quadrant geometry of the simulated image, defining where the parallel / serial overscans are and \
            therefore the direction of clocking and rotations before input into the cti algorithm.
        cosmic_ray_map : ndarray
            2D arrays flagging where cosmic rays on the image.
        cosmic_ray_parallel_buffer : int
            If a cosmic-ray mask is supplied, the number of pixels from each ray pixels are masked in the parallel \
            direction.
        cosmic_ray_serial_buffer : int
            If a cosmic-ray mask is supplied, the number of pixels from each ray pixels are masked in the serial \
            direction.
        """
        mask = cls.unmasked(shape_2d=cosmic_ray_map.shape_2d)

        cosmic_ray_mask = (cosmic_ray_map > 0.0).astype("bool")

        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if cosmic_ray_mask[y, x]:
                    y0, y1 = cosmic_ray_map.parallel_trail_from_y(
                        y=y, dy=cosmic_ray_parallel_buffer
                    )
                    mask[y0:y1, x] = True
                    x0, x1 = cosmic_ray_map.serial_trail_from_x(
                        x=x, dx=cosmic_ray_serial_buffer
                    )
                    mask[y, x0:x1] = True
                    y0, y1 = cosmic_ray_map.parallel_trail_from_y(
                        y=y, dy=cosmic_ray_diagonal_buffer
                    )
                    x0, x1 = cosmic_ray_map.serial_trail_from_x(
                        x=x, dx=cosmic_ray_diagonal_buffer
                    )
                    mask[y0:y1, x0:x1] = True

        return mask

    @classmethod
    def from_fits(cls, file_path, pixel_scales, hdu=0, origin=(0.0, 0.0)):
        """
        Loads the image from a .fits file.

        Parameters
        ----------
        file_path : str
            The full path of the fits file.
        hdu : int
            The HDU number in the fits file containing the image image.
        pixel_scales : float or (float, float)
            The arc-second to pixel conversion factor of each pixel.
        """

        if type(pixel_scales) is not tuple:
            if type(pixel_scales) is float or int:
                pixel_scales = (float(pixel_scales), float(pixel_scales))

        mask = cls(
            array_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu),
            pixel_scales=pixel_scales,
            origin=origin,
        )

        return mask

    def output_to_fits(self, file_path, overwrite=False):

        array_util.numpy_array_2d_to_fits(
            array_2d=self.astype("float"), file_path=file_path, overwrite=overwrite
        )

    @property
    def pixel_scale(self):
        if self.pixel_scales[0] == self.pixel_scales[1]:
            return self.pixel_scales[0]
        else:
            raise exc.MaskException(
                "Cannot return a pixel_scale for a a grid where each dimension has a "
                "different pixel scale (e.g. pixel_scales[0] != pixel_scales[1]"
            )

    @property
    def pixels_in_mask(self):
        return int(np.size(self) - np.sum(self))

    @property
    def is_all_false(self):
        return self.pixels_in_mask == self.shape_2d[0] * self.shape_2d[1]

    @property
    def shape_1d(self):
        return self.pixels_in_mask

    @property
    def shape_2d(self):
        return self.shape

    @property
    def shape_2d_scaled(self):
        return (
            float(self.pixel_scales[0] * self.shape[0]),
            float(self.pixel_scales[1] * self.shape[1]),
        )

    @property
    def scaled_maxima(self):
        return (
            (self.shape_2d_scaled[0] / 2.0) + self.origin[0],
            (self.shape_2d_scaled[1] / 2.0) + self.origin[1],
        )

    @property
    def scaled_minima(self):
        return (
            (-(self.shape_2d_scaled[0] / 2.0)) + self.origin[0],
            (-(self.shape_2d_scaled[1] / 2.0)) + self.origin[1],
        )

    @property
    def extent(self):
        return np.asarray(
            [
                self.scaled_minima[1],
                self.scaled_maxima[1],
                self.scaled_minima[0],
                self.scaled_maxima[0],
            ]
        )
