from autocti.mask import mask as msk


class MetaDataset:
    def __init__(self, model, settings):

        self.model = model
        self.settings = settings

    @property
    def is_parallel_fit(self):
        if (
            self.model.parallel_ccd_volume is not None
            and self.model.serial_ccd_volume is None
        ):
            return True
        else:
            return False

    @property
    def is_serial_fit(self):
        if (
            self.model.parallel_ccd_volume is None
            and self.model.serial_ccd_volume is not None
        ):
            return True
        else:
            return False

    @property
    def is_parallel_and_serial_fit(self):
        if (
            self.model.parallel_ccd_volume is not None
            and self.model.serial_ccd_volume is not None
        ):
            return True
        else:
            return False

    def mask_for_analysis_from_cosmic_ray_map(self, cosmic_ray_map, mask):

        cosmic_ray_mask = (
            msk.Mask.from_cosmic_ray_map(
                cosmic_ray_map=cosmic_ray_map,
                cosmic_ray_parallel_buffer=self.settings.cosmic_ray_parallel_buffer,
                cosmic_ray_serial_buffer=self.settings.cosmic_ray_serial_buffer,
                cosmic_ray_diagonal_buffer=self.settings.cosmic_ray_diagonal_buffer,
            )
            if cosmic_ray_map is not None
            else None
        )

        if cosmic_ray_map is not None:
            return mask + cosmic_ray_mask

        return mask
