import numpy as np
from typing import Tuple

import autoarray as aa

from autocti import exc


class Layout1D(aa.Layout1D):
    def __init__(
        self,
        shape_1d: Tuple[int],
        region_list: aa.type.Region1DList,
        prescan: aa.type.Region1DLike = None,
        overscan: aa.type.Region1DLike = None,
    ):
        """
        Abstract base class for a charge injection layout_ci, which defines the regions charge injections appears \
         on a charge-injection array, the input normalization and other properties.

        Parameters
        -----------
        region_list
            A list of the integer coordinates specifying the corners of each charge injection region \
            (top-row, bottom-row, left-column, right-column).
        """

        self.region_list = list(map(aa.Region1D, region_list))

        for region in self.region_list:

            if region.x1 > shape_1d[0]:
                raise exc.LayoutException(
                    "The charge injection layout_ci regions are bigger than the image image_shape"
                )

        from autocti.extract.one_d.master import Extract1DMaster

        self.extract = Extract1DMaster.from_region_list(
            region_list=region_list, prescan=prescan, overscan=overscan
        )

        super().__init__(shape_1d=shape_1d, prescan=prescan, overscan=overscan)

    @property
    def pixels_between_regions(self):
        return [
            self.region_list[i + 1].x0 - self.region_list[i].x1
            for i in range(len(self.region_list) - 1)
        ]

    @property
    def trail_size_to_array_edge(self):
        return self.shape_1d[0] - np.max([region.x1 for region in self.region_list])

    @property
    def smallest_trails_pixels_to_array_edge(self):

        pixels_between_regions = self.pixels_between_regions
        pixels_between_regions.append(self.trail_size_to_array_edge)
        return np.min(pixels_between_regions)

    def extract_region_from(self, array: aa.Array1D, region: str):

        if region == "front_edge":
            return self.extract.fpr.stacked_array_1d_from(
                array=array, pixels=(0, self.extract.fpr.total_pixels_min)
            )
        elif region == "trails":
            return self.extract.eper.stacked_array_1d_from(
                array=array, pixels=(0, self.smallest_trails_pixels_to_array_edge)
            )
        else:
            raise exc.PlottingException(
                "The line region specified for the plotting of a line was invalid"
            )
