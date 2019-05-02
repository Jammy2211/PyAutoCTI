from autocti.data import mask as msk

import numpy as np

class CIMask(msk.Mask):

    def __new__(cls, array, *args, **kwargs):
        mask = np.array(array, dtype='float64').view(cls)
        return mask

    @classmethod
    def masked_parallel_front_edge_from_ci_frame(self, shape, ci_frame, rows):

        front_edge_regions = ci_frame.parallel_front_edge_regions_from_frame(rows=rows)
        mask = np.full(shape, True)

        for region in front_edge_regions:
            mask[region.y0:region.y1, region.x0:region.x1] = False

        return CIMask(array=mask, frame_geometry=ci_frame.frame_geometry)

    @classmethod
    def masked_parallel_trails_from_ci_frame(self, shape, ci_frame, rows):

        trails_regions = ci_frame.parallel_trails_regions_from_frame(rows=rows)
        mask = np.full(shape, True)

        for region in trails_regions:
            mask[region.y0:region.y1, region.x0:region.x1] = False

        return CIMask(array=mask, frame_geometry=ci_frame.frame_geometry)

    @classmethod
    def masked_serial_front_edge_from_ci_frame(self, shape, ci_frame, columns):

        front_edge_regions = ci_frame.serial_front_edge_regions_from_frame(columns=columns)
        mask = np.full(shape, True)

        for region in front_edge_regions:
            mask[region.y0:region.y1, region.x0:region.x1] = False

        return CIMask(array=mask, frame_geometry=ci_frame.frame_geometry)

    @classmethod
    def masked_serial_trails_from_ci_frame(self, shape, ci_frame, columns):

        trails_regions = ci_frame.serial_trails_regions_from_frame(columns=columns)
        mask = np.full(shape, True)

        for region in trails_regions:
            mask[region.y0:region.y1, region.x0:region.x1] = False

        return CIMask(array=mask, frame_geometry=ci_frame.frame_geometry)