import autofit as af
from autocti.data import mask as msk

import numpy as np

class CIMask(msk.Mask):

    def __new__(cls, array, *args, **kwargs):
        mask = np.array(array, dtype='bool').view(cls)
        return mask

    @classmethod
    def masked_parallel_front_edge_from_ci_frame(cls, shape, ci_frame, rows, invert=False):

        front_edge_regions = ci_frame.parallel_front_edge_regions_from_frame(rows=rows)
        mask = np.full(shape, False)

        for region in front_edge_regions:
            mask[region.y0:region.y1, region.x0:region.x1] = True

        if invert: mask = np.invert(mask)

        return CIMask(array=mask.astype('bool'), frame_geometry=ci_frame.frame_geometry)

    @classmethod
    def masked_parallel_trails_from_ci_frame(cls, shape, ci_frame, rows, invert=False):

        trails_regions = ci_frame.parallel_trails_regions_from_frame(shape=shape, rows=rows)
        mask = np.full(shape, False)

        for region in trails_regions:
            mask[region.y0:region.y1, region.x0:region.x1] = True

        if invert: mask = np.invert(mask)

        return CIMask(array=mask.astype('bool'), frame_geometry=ci_frame.frame_geometry)

    @classmethod
    def masked_serial_front_edge_from_ci_frame(cls, shape, ci_frame, columns, invert=False):

        front_edge_regions = ci_frame.serial_front_edge_regions_from_frame(columns=columns)
        mask = np.full(shape, False)

        for region in front_edge_regions:
            mask[region.y0:region.y1, region.x0:region.x1] = True

        if invert: mask = np.invert(mask)

        return CIMask(array=mask.astype('bool'), frame_geometry=ci_frame.frame_geometry)

    @classmethod
    def masked_serial_trails_from_ci_frame(cls, shape, ci_frame, columns, invert=False):

        trails_regions = ci_frame.serial_trails_regions_from_frame(columns=columns)
        mask = np.full(shape, False)

        for region in trails_regions:
            mask[region.y0:region.y1, region.x0:region.x1] = True

        if invert: mask = np.invert(mask)

        return CIMask(array=mask.astype('bool'), frame_geometry=ci_frame.frame_geometry)