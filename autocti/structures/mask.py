from autoarray.mask import mask as msk
from autocti.structures import region as reg


class Mask(msk.Mask):
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
        cosmic_ray_map : arrays.Array
            2D arrays flagging where cosmic rays on the image.
        cosmic_ray_parallel_buffer : int
            The number of pixels from each ray pixels are masked in the parallel direction.
        cosmic_ray_serial_buffer : int
            The number of pixels from each ray pixels are masked in the serial direction.
        cosmic_ray_diagonal_buffer : int
            The number of pixels from each ray pixels are masked in the digonal up from the parallel + serial direction.
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
