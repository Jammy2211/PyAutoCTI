import numpy as np

from autoarray.mask import mask as msk
from autocti.structures import frame


class Mask(msk.Mask):
    def __new__(cls, mask_2d, pixel_scales=None, *args, **kwargs):

        obj = super(Mask, cls).__new__(
            cls=cls, mask_2d=mask_2d, pixel_scales=pixel_scales,
        )

        return obj

    @classmethod
    def from_masked_regions(cls, shape_2d, masked_regions, **kwargs):

        mask = cls.unmasked(shape_2d=shape_2d)
        masked_regions = list(map(lambda region: frame.Region(region=region), masked_regions))
        for region in masked_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        return mask

    @classmethod
    def from_cosmic_ray_image(
        cls,
        cosmic_ray_image,
        cosmic_ray_parallel_buffer=0,
        cosmic_ray_serial_buffer=0,
        cosmic_ray_diagonal_buffer=0,
        **kwargs
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
        cosmic_ray_image : ndarray
            2D arrays flagging where cosmic rays on the image.
        cosmic_ray_parallel_buffer : int
            If a cosmic-ray mask is supplied, the number of pixels from each ray pixels are masked in the parallel \
            direction.
        cosmic_ray_serial_buffer : int
            If a cosmic-ray mask is supplied, the number of pixels from each ray pixels are masked in the serial \
            direction.
        """
        mask = cls.unmasked(shape_2d=cosmic_ray_image.shape_2d)

        cosmic_ray_mask = (cosmic_ray_image > 0.0).astype("bool")

        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if cosmic_ray_mask[y, x]:
                    y0, y1 = cosmic_ray_image.parallel_trail_from_y(
                        y=y, dy=cosmic_ray_parallel_buffer
                    )
                    mask[y0:y1, x] = True
                    x0, x1 = cosmic_ray_image.serial_trail_from_x(
                        x=x, dx=cosmic_ray_serial_buffer
                    )
                    mask[y, x0:x1] = True
                    y0, y1 = cosmic_ray_image.parallel_trail_from_y(
                        y=y, dy=cosmic_ray_diagonal_buffer
                    )
                    x0, x1 = cosmic_ray_image.serial_trail_from_x(
                        x=x, dx=cosmic_ray_diagonal_buffer
                    )
                    mask[y0:y1, x0:x1] = True

        return mask
