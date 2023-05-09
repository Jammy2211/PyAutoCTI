import numpy as np
from typing import Optional, Tuple

import autoarray as aa

from autocti.extract.settings import SettingsExtract
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
        ----------
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

        self.extract = Extract1DMaster(
            shape_1d=shape_1d,
            region_list=region_list,
            prescan=prescan,
            overscan=overscan,
        )

        super().__init__(shape_1d=shape_1d, prescan=prescan, overscan=overscan)

    @property
    def parallel_rows_between_regions(self):
        return self.extract.fpr.parallel_rows_between_regions

    @property
    def trail_size_to_array_edge(self):
        return self.extract.fpr.trail_size_to_array_edge

    @property
    def smallest_eper_pixels_to_array_edge(self):
        parallel_rows_between_regions = self.parallel_rows_between_regions
        parallel_rows_between_regions.append(self.trail_size_to_array_edge)
        return np.min(parallel_rows_between_regions)

    def extract_region_from(self, array: aa.Array1D, region: Optional):
        if region is None:
            return array

        if region == "fpr":
            return self.extract.fpr.stacked_array_1d_from(
                array=array,
                settings=SettingsExtract(pixels=(0, self.extract.fpr.total_pixels_min)),
            )
        elif region == "eper":
            return self.extract.eper.stacked_array_1d_from(
                array=array,
                settings=SettingsExtract(
                    pixels=(0, self.smallest_eper_pixels_to_array_edge)
                ),
            )
        else:
            raise exc.PlottingException(
                "The line region specified for the plotting of a line was invalid"
            )
