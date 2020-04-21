import autofit as af
import autoarray as aa
from autocti.util import exc
from autocti.fit import fit
from autocti.structures import mask as msk
from autoarray.operators.inversion import pixelizations as pix

import numpy as np


def isprior(obj):
    if isinstance(obj, af.PriorModel):
        return True
    return False


def isinstance_or_prior(obj, cls):
    if isinstance(obj, cls):
        return True
    if isinstance(obj, af.PriorModel) and obj.cls == cls:
        return True
    return False


class MetaDataset:
    def __init__(
        self,
        model,
        parallel_total_density_range=None,
        serial_total_density_range=None,
        cosmic_ray_parallel_buffer=10,
        cosmic_ray_serial_buffer=10,
        cosmic_ray_diagonal_buffer=3,
    ):

        self.model = model
        self.parallel_total_density_range = parallel_total_density_range
        self.serial_total_density_range = serial_total_density_range
        self.cosmic_ray_parallel_buffer = cosmic_ray_parallel_buffer
        self.cosmic_ray_serial_buffer = cosmic_ray_serial_buffer
        self.cosmic_ray_diagonal_buffer = cosmic_ray_diagonal_buffer

    @property
    def is_only_parallel_fit(self):
        if (
            self.model.parallel_ccd_volume is not None
            and self.model.serial_ccd_volume is None
        ):
            return True
        else:
            return False

    @property
    def is_only_serial_fit(self):
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
                cosmic_ray_parallel_buffer=self.cosmic_ray_parallel_buffer,
                cosmic_ray_serial_buffer=self.cosmic_ray_serial_buffer,
                cosmic_ray_diagonal_buffer=self.cosmic_ray_diagonal_buffer,
            )
            if cosmic_ray_map is not None
            else None
        )

        if cosmic_ray_map is not None:
            return mask + cosmic_ray_mask

        return mask
