import numpy as np
import math
from typing import Dict, List, Optional, Tuple
import autoarray as aa

from autocti.charge_injection import ci_util
from autocti.layout.two_d import Layout2D


class Layout2DCI(Layout2D):
    def __init__(
        self,
        shape_2d: Tuple[int, int],
        region_list: aa.type.Region2DList,
        original_roe_corner: Tuple[int, int] = (1, 0),
        parallel_overscan: Optional[aa.type.Region2DLike] = None,
        serial_prescan: Optional[aa.type.Region2DLike] = None,
        serial_overscan: Optional[aa.type.Region2DLike] = None,
        electronics: Optional["ElectronicsCI"] = None,
    ):
        """
        A charge injection layout, which defines the regions charge injections appear on a charge injection image.

        It also contains over regions of the image, for example the serial prescan, overscan and paralle overscan.

        Parameters
        -----------
        shape_2d
            The two dimensional shape of the charge injection imaging, corresponding to the number of rows (pixels
            in parallel direction) and columns (pixels in serial direction).
        region_list
            Integer pixel coordinates specifying the corners of each charge injection region (top-row, bottom-row,
            left-column, right-column).
        original_roe_corner
            The original read-out electronics corner of the charge injeciton imaging, which is internally rotated to a
            common orientation in **PyAutoCTI**.
        parallel_overscan
            Integer pixel coordinates specifying the corners of the parallel overscan (top-row, bottom-row,
            left-column, right-column).
        serial_prescan
            Integer pixel coordinates specifying the corners of the serial prescan (top-row, bottom-row,
            left-column, right-column).
        serial_overscan
            Integer pixel coordinates specifying the corners of the serial overscan (top-row, bottom-row,
            left-column, right-column).
        electronics
            The charge injection electronics parameters of the image (e.g. the IG1 and IG2 voltages).
        """

        super().__init__(
            shape_2d=shape_2d,
            region_list=region_list,
            original_roe_corner=original_roe_corner,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

        self.electronics = electronics

    @property
    def extract(self) -> "Extract2DMaster":

        from autocti.extract.two_d.master import Extract2DMaster

        return Extract2DMaster(
            shape_2d=self.shape_2d,
            region_list=self.region_list,
            parallel_overscan=self.parallel_overscan,
            serial_prescan=self.serial_prescan,
            serial_overscan=self.serial_overscan,
        )

    @classmethod
    def from_euclid_fits_header(cls, ext_header, do_rotation):

        serial_overscan_size = ext_header.get("OVRSCANX", default=None)
        serial_prescan_size = ext_header.get("PRESCANX", default=None)
        serial_size = ext_header.get("NAXIS1", default=None)
        parallel_size = ext_header.get("NAXIS2", default=None)

        electronics = ElectronicsCI.from_ext_header(ext_header=ext_header)

        layout = aa.euclid.Layout2DEuclid.from_fits_header(ext_header=ext_header)

        if do_rotation:
            roe_corner = layout.original_roe_corner
        else:
            roe_corner = (1, 0)

        region_ci_list = ci_util.region_list_ci_via_electronics_from(
            injection_start=electronics.injection_start,
            injection_on=electronics.injection_on,
            injection_off=electronics.injection_off,
            injection_total=electronics.injection_total,
            serial_prescan_size=serial_prescan_size,
            serial_overscan_size=serial_overscan_size,
            serial_size=serial_size,
            parallel_size=parallel_size,
            roe_corner=roe_corner,
        )

        return cls(
            shape_2d=(parallel_size, serial_size),
            region_list=region_ci_list,
            original_roe_corner=layout.original_roe_corner,
            parallel_overscan=layout.parallel_overscan,
            serial_prescan=layout.serial_prescan,
            serial_overscan=layout.serial_overscan,
            electronics=electronics,
        )

    def pre_cti_data_uniform_from(
        self, norm: float, pixel_scales: aa.type.PixelScales
    ) -> aa.Array2D:
        """
        Use this charge injection layout to generate a pre-cti charge injection image.

        This is performed by going to its charge injection regions and adding an input charge injection
        normalization value.

        Parameters
        -----------
        norm
            The normalization of the charge injection region.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        """

        pre_cti_data = np.zeros(self.shape_2d)

        for region in self.region_list:
            pre_cti_data[region.slice] += norm

        return aa.Array2D.manual(array=pre_cti_data, pixel_scales=pixel_scales)

    def pre_cti_data_non_uniform_from(
        self,
        injection_norm_list: List[float],
        pixel_scales: aa.type.PixelScales,
        row_slope: Optional[float] = 0.0,
    ) -> aa.Array2D:
        """
        Use this charge injection layout to generate a pre-cti charge injection image. This is performed by going 
        to its charge injection regions and adding an input normalization value to each column, which are
        passed in as a list.

        For one column of a non-uniform charge injection pre-cti image, this function assumes that each non-uniform 
        charge injection region has the same overall normalization value. This assumes the spikes / troughs in the
        injection current that cause non-uniformity occur in an identical fashion for each charge injection region.

        Non-uniformity across the rows of a charge injection layout_ci is due to a drop-off in voltage in the current.
        Therefore, it appears smooth and be modeled as an analytic function, which this code assumes is a
        power-law with slope `row_slope`.

        Parameters
        -----------
        shape_native
            The image_shape of the pre_cti_datas to be created.
        row_slope
            The power-law slope of non-uniformity in the row charge injection profile.
        ci_seed
            Input ci_seed for the random number generator to give reproducible results. A new ci_seed is always used for each \
            pre_cti_datas, ensuring each non-uniform ci_region has the same column non-uniformity layout_ci.
        """

        pre_cti_data = np.zeros(self.shape_2d)

        for region in self.region_list:

            pre_cti_data[region.slice] += ci_util.region_ci_from(
                region_dimensions=region.shape,
                injection_norm_list=injection_norm_list,
                row_slope=row_slope,
            )

        return aa.Array2D.manual(array=pre_cti_data, pixel_scales=pixel_scales)

    def pre_cti_data_non_uniform_via_lists_from(
        self,
        injection_norm_lists: List[List[float]],
        pixel_scales: aa.type.PixelScales,
        row_slope: Optional[float] = 0.0,
    ) -> aa.Array2D:
        """
        Use this charge injection layout to generate a pre-cti charge injection image. This is performed by going
        to its charge injection regions and adding an input normalization value to each column, which are
        passed in as a list.

        For one column of a non-uniform charge injection pre-cti image, this function assumes that each non-uniform
        charge injection region can have a different normalization value. The normalizztion values `injection_norm_lists`
        are passed as a list of lists so that the normalization of each injection in each region can be specified.

        Non-uniformity across the rows of a charge injection layout_ci is due to a drop-off in voltage in the current.
        Therefore, it appears smooth and be modeled as an analytic function, which this code assumes is a
        power-law with slope `row_slope`.

        Parameters
        -----------
        shape_native
            The image_shape of the pre_cti_datas to be created.
        row_slope
            The power-law slope of non-uniformity in the row charge injection profile.
        ci_seed
            Input ci_seed for the random number generator to give reproducible results. A new ci_seed is always used for each \
            pre_cti_datas, ensuring each non-uniform ci_region has the same column non-uniformity layout_ci.
        """

        pre_cti_data = np.zeros(self.shape_2d)

        for region_index, region in enumerate(self.region_list):

            pre_cti_data[region.slice] += ci_util.region_ci_from(
                region_dimensions=region.shape,
                injection_norm_list=injection_norm_lists[region_index],
                row_slope=row_slope,
            )

        return aa.Array2D.manual(array=pre_cti_data, pixel_scales=pixel_scales)


class ElectronicsCI:
    def __init__(
        self,
        injection_on: Optional[int] = None,
        injection_off: Optional[int] = None,
        injection_start: Optional[int] = None,
        injection_end: Optional[int] = None,
        ig_1: Optional[float] = None,
        ig_2: Optional[float] = None,
    ):
        """
        Stores the electronics parameters contained for a charge injection line image, with this class currently
        specific to those in Euclid.

        These are extracted from the .fits file header of a charge injection image, with the original .fits
        headers included in the `as_ext_header_dict` property.

        The `injection_on` and `injection_off` parameters determine the number of pixels the charge injection is
        held on and then off for. The `v_start` and `v_end` parameters define the pixels where the charge injection
        starts and ends.

        For example, take a CCD which has 1820 rows of pixels, where:

        - `injection_on=100`
        - `injection_off=200`
        - `v_start=10`
        - `v_end`=1810

        Starting from row 10, for every 300 rows of pixels there first 100 pixels will contain charge injection and
        the remaining 200 rows will not (they will contain EPER trails). This pattern will be repeated 6 times over
        the next 1800 pixels of the CCD with the charge injection ending at 1810.

        NOTE: The charge injection electrons have the following four parameters:

        VSTART_CHJ_INJ
        VEND_CHJ_INJ
        VSTART
        VEND

        I do not yet know which of these maps to which fits header. I am currently assuming all 4 correspond to
        `v_start` and `v_end`, albeit their functionality is not used specifically.

        Parameters
        ----------
        injection_on
            The number of rows of pixels the charge injection is held on for per charge injection region.
        injection_off
            The number of rows of pixels the charge injection is held off for per charge injection region.
        injection_start
            The pixel row where the charge injection begins.
        injection_end
            The pixel row where the charge injection ends.
        ig_1
            The voltage of injection gate 1.
        ig_2
            The voltage of injection gate 2.
        """
        self.injection_on = injection_on
        self.injection_off = injection_off
        self.injection_start = injection_start
        self.injection_end = injection_end
        self.ig_1 = ig_1
        self.ig_2 = ig_2

    @classmethod
    def from_ext_header(cls, ext_header: Dict) -> "ElectronicsCI":
        """
        Creates the charge injection electronics from a Euclid charge injection imaging .fits header.

        Parameters
        ----------
        ext_header
            The .fits header dictionary of a Euclid charge injection image.
        """
        injection_on = ext_header["CI_IJON"]
        injection_off = ext_header["CI_IJOFF"]
        injection_start = ext_header["CI_VSTAR"]
        injection_end = ext_header["CI_VEND"]

        return ElectronicsCI(
            injection_on=injection_on,
            injection_off=injection_off,
            injection_start=injection_start,
            injection_end=injection_end,
        )

    @property
    def as_ext_header_dict(self) -> Dict:
        """
        Returns the charge injection electronics as a dictionary which is representative of the parameter values
        stored in a Euclid charge injection .fits image.
        """
        return {
            "CI_IJON": self.injection_on,
            "CI_IJOFF": self.injection_off,
            "CI_VSTAR": self.injection_start,
            "CI_VEND": self.injection_end,
        }

    @property
    def injection_total(self) -> int:
        """
        The total number of charge injection regions for these electronics settings.
        """

        injection_range = self.injection_end - self.injection_start

        for injection_total in range(10):

            total_pixels = math.floor((injection_total+1)*(self.injection_on) + injection_total*self.injection_off)

            if total_pixels > injection_range:

                return injection_total
