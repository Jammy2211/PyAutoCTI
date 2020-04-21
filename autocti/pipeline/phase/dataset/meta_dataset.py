import autofit as af
import autoarray as aa
from autocti.util import exc
from autocti.fit import fit
from autocti.charge_injection import ci_mask
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
        columns=None,
        rows=None,
        parallel_front_edge_mask_rows=None,
        parallel_trails_mask_rows=None,
        parallel_total_density_range=None,
        serial_front_edge_mask_columns=None,
        serial_trails_mask_columns=None,
        serial_total_density_range=None,
        cosmic_ray_parallel_buffer=10,
        cosmic_ray_serial_buffer=10,
        cosmic_ray_diagonal_buffer=3,
    ):

        self.model = model
        self.columns = columns
        self.rows = rows
        self.parallel_front_edge_mask_rows = parallel_front_edge_mask_rows
        self.parallel_trails_mask_rows = parallel_trails_mask_rows
        self.parallel_total_density_range = parallel_total_density_range
        self.serial_front_edge_mask_columns = serial_front_edge_mask_columns
        self.serial_trails_mask_columns = serial_trails_mask_columns
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

    def masks_for_analysis_from_datasets(self, datasets, masks):

        cosmic_ray_masks = list(
            map(
                lambda data: ci_mask.CIMask.from_cosmic_ray_map(
                    shape_2d=data.shape,
                    frame_geometry=data.ci_frame.frame_geometry,
                    cosmic_ray_map=data.cosmic_ray_map,
                    cosmic_ray_parallel_buffer=self.cosmic_ray_parallel_buffer,
                    cosmic_ray_serial_buffer=self.cosmic_ray_serial_buffer,
                    cosmic_ray_diagonal_buffer=self.cosmic_ray_diagonal_buffer,
                )
                if data.cosmic_ray_map is not None
                else None,
                datasets,
            )
        )

        masks = list(
            map(
                lambda mask, cosmic_ray_mask: mask + cosmic_ray_mask
                if cosmic_ray_mask is not None
                else mask,
                masks,
                cosmic_ray_masks,
            )
        )

        if self.parallel_front_edge_mask_rows is not None:
            parallel_front_edge_masks = list(
                map(
                    lambda data: ci_mask.CIMask.masked_parallel_front_edge_from_ci_frame(
                        shape=data.shape,
                        ci_frame=data.ci_frame,
                        rows=self.parallel_front_edge_mask_rows,
                    ),
                    datasets,
                )
            )

            masks = list(
                map(
                    lambda mask, parallel_front_edge_mask: mask
                    + parallel_front_edge_mask,
                    masks,
                    parallel_front_edge_masks,
                )
            )

        if self.parallel_trails_mask_rows is not None:
            parallel_trails_masks = list(
                map(
                    lambda data: ci_mask.CIMask.masked_parallel_trails_from_ci_frame(
                        shape=data.shape,
                        ci_frame=data.ci_frame,
                        rows=self.parallel_trails_mask_rows,
                    ),
                    datasets,
                )
            )

            masks = list(
                map(
                    lambda mask, parallel_trails_mask: mask + parallel_trails_mask,
                    masks,
                    parallel_trails_masks,
                )
            )

        if self.serial_front_edge_mask_columns is not None:
            serial_front_edge_masks = list(
                map(
                    lambda data: ci_mask.CIMask.masked_serial_front_edge_from_ci_frame(
                        shape=data.shape,
                        ci_frame=data.ci_frame,
                        columns=self.serial_front_edge_mask_columns,
                    ),
                    datasets,
                )
            )

            masks = list(
                map(
                    lambda mask, serial_front_edge_mask: mask + serial_front_edge_mask,
                    masks,
                    serial_front_edge_masks,
                )
            )

        if self.serial_trails_mask_columns is not None:
            serial_trails_masks = list(
                map(
                    lambda data: ci_mask.CIMask.masked_serial_trails_from_ci_frame(
                        shape=data.shape,
                        ci_frame=data.ci_frame,
                        columns=self.serial_trails_mask_columns,
                    ),
                    datasets,
                )
            )

            masks = list(
                map(
                    lambda mask, serial_trails_mask: mask + serial_trails_mask,
                    masks,
                    serial_trails_masks,
                )
            )

        return masks
