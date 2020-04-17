import numpy as np


class AbstractStructure(np.ndarray):
    def __new__(cls, structure, mask, *args, **kwargs):
        """"""
        obj = structure.view(cls)
        obj.mask = mask
        return obj

    def __array_finalize__(self, obj):

        if isinstance(obj, AbstractStructure):
            if hasattr(obj, "mask"):
                self.mask = obj.mask

    @property
    def shape_2d(self):
        return self.mask.shape

    @property
    def pixel_scales(self):
        return self.mask.pixel_scales

    @property
    def pixel_scale(self):
        return self.mask.pixel_scale

    @property
    def origin(self):
        return self.mask.origin

    @property
    def total_pixels(self):
        return self.shape[0] * self.shape[1]

    @property
    def shape_2d_scaled(self):
        return self.mask.shape_2d_scaled

    @property
    def scaled_maxima(self):
        return self.mask.scaled_maxima

    @property
    def scaled_minima(self):
        return self.mask.scaled_minima

    @property
    def extent(self):
        return self.mask.extent
