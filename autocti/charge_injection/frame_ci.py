import numpy as np
from autoarray.structures.arrays import abstract_array
from autoarray.structures.arrays.two_d import array_2d_util
from autocti.charge_injection import layout_ci as pattern
from autocti.mask.mask_2d import Mask2D
from autoarray.layout import layout_util
from autoarray.instruments import euclid
from autoarray.geometry import geometry_util


class AbstractCIFrame(abstract_frame.AbstractFrame2D):
    def __new__(
        cls,
        array,
        mask,
        layout_ci,
        original_roe_corner=(1, 0),
        exposure_info=None,
        scans=None,
    ):
        """
        Class which represents the CCD quadrant of a charge injection image (e.g. the location of the parallel and
        serial front edge, trails).

        frame_geometry : CIFrame.CIQuadGeometry
            The quadrant geometry of the image, defining where the parallel / serial overscans are and
            therefore the direction of clocking and rotations before input into the cti algorithm.
        layout_ci : Layout2DCI.Layout2DCI
            The charge injection layout_ci (scans, normalization, etc.) of the charge injection image.
        """

        if type(array) is list:
            array = np.asarray(array)

        obj = array.view(cls)
        obj.mask = mask
        obj.zoom_for_plot = False
        obj.original_roe_corner = original_roe_corner
        obj.exposure_info = exposure_info
        obj.scans = scans or abstract_frame.Scans()
        obj.layout_ci = layout_ci

        return obj

    def _new_structure(self, array, mask):
        return self.__class__(
            array=array,
            mask=mask,
            layout_ci=self.layout_ci,
            original_roe_corner=self.original_roe_corner,
            scans=self.scans,
            exposure_info=self.exposure_info,
        )


class CIFrame(AbstractCIFrame):
    @classmethod
    def from_frame_ci(cls, frame_ci, mask):

        frame_ci[mask == True] = 0.0

        return CIFrame(
            array=frame_ci,
            mask=mask,
            layout_ci=frame_ci.layout_ci,
            original_roe_corner=frame_ci.original_roe_corner,
            exposure_info=frame_ci.exposure_info,
            scans=abstract_frame.layout.from_frame(frame=frame_ci),
        )


class CIFrameEuclid(CIFrame):
    @classmethod
    def from_fits_header(cls, array, ext_header):
        """
        Use an input array of a Euclid quadrant and its corresponding .fits file header to rotate the quadrant to
        the correct orientation for arCTIc clocking.

        See the docstring of the `from_ccd_and_quadrant_id` classmethod for a complete description of the Euclid FPA,
        quadrants and rotations.
        """

        ccd_id = ext_header["CCDID"]
        quadrant_id = ext_header["QUADID"]

        parallel_overscan_size = ext_header.get("PAROVRX", default=None)
        if parallel_overscan_size is None:
            parallel_overscan_size = 0
        serial_overscan_size = ext_header.get("OVRSCANX", default=None)
        serial_prescan_size = ext_header.get("PRESCANX", default=None)
        serial_size = ext_header.get("NAXIS1", default=None)
        parallel_size = ext_header.get("NAXIS2", default=None)

        injection_on = ext_header["INJON"]
        injection_off = ext_header["INJOFF"]
        injection_total = ext_header["INJTOTAL"]

        roe_corner = euclid.roe_corner_from(ccd_id=ccd_id, quadrant_id=quadrant_id)

        regions_ci = pattern.regions_ci_from(
            injection_on=injection_on,
            injection_off=injection_off,
            injection_total=injection_total,
            parallel_size=parallel_size,
            serial_prescan_size=serial_prescan_size,
            serial_overscan_size=serial_overscan_size,
            serial_size=serial_size,
            roe_corner=roe_corner,
        )

        normalization = ext_header["INJNORM"]

        layout_ci = pattern.Layout2DCIUniform(
            normalization=normalization, region_list=regions_ci
        )

        return cls.from_ccd_and_quadrant_id(
            array=array,
            ccd_id=ccd_id,
            quadrant_id=quadrant_id,
            layout_ci=layout_ci,
            parallel_size=parallel_size,
            serial_size=serial_size,
            serial_prescan_size=serial_prescan_size,
            serial_overscan_size=serial_overscan_size,
            parallel_overscan_size=parallel_overscan_size,
        )

    @classmethod
    def from_ccd_and_quadrant_id(
        cls,
        array,
        ccd_id,
        quadrant_id,
        layout_ci,
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=51,
        serial_overscan_size=29,
        parallel_overscan_size=20,
    ):
        """
        In the Euclid FPA, the quadrant id ('E', 'F', 'G', 'H') depends on whether the CCD is located
        on the left side (rows 1-3) or right side (rows 4-6) of the FPA:

        LEFT SIDE ROWS 1-2-3
        --------------------

         <--------S-----------   ---------S----------->
        [] [========= 2 =========] [========= 3 =========] []          |
        /    [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  /          |
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | Direction arctic
        P   [xxxxxxxxx H xxxxxxxxx] [xxxxxxxxx G xxxxxxxxx]  P         | clocks an image
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | without any rotation
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | (e.g. towards row 0
                                                                       | of the ndarrays)
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
        P   [xxxxxxxxx E xxxxxxxxx] [xxxxxxxxx F xxxxxxxxx] P          |
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
            [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]            |

        [] [========= 0 =========] [========= 1 =========] []
            <---------S----------   ----------S----------->


        RIGHT SIDE ROWS 4-5-6
        ---------------------

         <--------S-----------   ---------S----------->
        [] [========= 2 =========] [========= 3 =========] []          |
        /    [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  /          |
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | Direction arctic
        P   [xxxxxxxxx F xxxxxxxxx] [xxxxxxxxx E xxxxxxxxx]  P         | clocks an image
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | without any rotation
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | (e.g. towards row 0
                                                                       | of the ndarrays)
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
        P   [xxxxxxxxx G xxxxxxxxx] [xxxxxxxxx H xxxxxxxxx] P          |
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
            [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]            |

        [] [========= 0 =========] [========= 1 =========] []
            <---------S----------   ----------S----------->

        Therefore, to setup a quadrant image with the correct frame_geometry using its CCD id (from which
        we can extract its row number) and quadrant id, we need to first determine if the CCD is on the left / right
        side and then use its quadrant id ('E', 'F', 'G' or 'H') to pick the correct quadrant.
        """

        row_index = ccd_id[-1]

        if (row_index in "123") and (quadrant_id == "E"):
            return CIFrameEuclid.bottom_left(
                array=array,
                layout_ci=layout_ci,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "123") and (quadrant_id == "F"):
            return CIFrameEuclid.bottom_right(
                array=array,
                layout_ci=layout_ci,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "123") and (quadrant_id == "G"):
            return CIFrameEuclid.top_right(
                array=array,
                layout_ci=layout_ci,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "123") and (quadrant_id == "H"):
            return CIFrameEuclid.top_left(
                array=array,
                layout_ci=layout_ci,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "456") and (quadrant_id == "E"):
            return CIFrameEuclid.top_right(
                array=array,
                layout_ci=layout_ci,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "456") and (quadrant_id == "F"):
            return CIFrameEuclid.top_left(
                array=array,
                layout_ci=layout_ci,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "456") and (quadrant_id == "G"):
            return CIFrameEuclid.bottom_left(
                array=array,
                layout_ci=layout_ci,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )
        elif (row_index in "456") and (quadrant_id == "H"):
            return CIFrameEuclid.bottom_right(
                array=array,
                layout_ci=layout_ci,
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                serial_overscan_size=serial_overscan_size,
                parallel_overscan_size=parallel_overscan_size,
            )

    @classmethod
    def top_left(
        cls,
        array,
        layout_ci,
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=51,
        serial_overscan_size=29,
        parallel_overscan_size=20,
    ):

        scans = euclid.ScansEuclid.top_left(
            parallel_size=parallel_size,
            serial_size=serial_size,
            serial_prescan_size=serial_prescan_size,
            serial_overscan_size=serial_overscan_size,
            parallel_overscan_size=parallel_overscan_size,
        )

        return CIFrame.manual(
            array=array,
            pixel_scales=0.1,
            layout_ci=layout_ci,
            roe_corner=(0, 0),
            scans=scans,
        )

    @classmethod
    def top_right(
        cls,
        array,
        layout_ci,
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=51,
        serial_overscan_size=29,
        parallel_overscan_size=20,
    ):

        scans = euclid.ScansEuclid.top_right(
            parallel_size=parallel_size,
            serial_size=serial_size,
            serial_prescan_size=serial_prescan_size,
            serial_overscan_size=serial_overscan_size,
            parallel_overscan_size=parallel_overscan_size,
        )

        return CIFrame.manual(
            array=array,
            pixel_scales=0.1,
            layout_ci=layout_ci,
            roe_corner=(0, 1),
            scans=scans,
        )

    @classmethod
    def bottom_left(
        cls,
        array,
        layout_ci,
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=51,
        serial_overscan_size=29,
        parallel_overscan_size=20,
    ):

        scans = euclid.ScansEuclid.bottom_left(
            parallel_size=parallel_size,
            serial_size=serial_size,
            serial_prescan_size=serial_prescan_size,
            serial_overscan_size=serial_overscan_size,
            parallel_overscan_size=parallel_overscan_size,
        )

        return CIFrame.manual(
            array=array,
            pixel_scales=0.1,
            layout_ci=layout_ci,
            roe_corner=(1, 0),
            scans=scans,
        )

    @classmethod
    def bottom_right(
        cls,
        array,
        layout_ci,
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=51,
        serial_overscan_size=29,
        parallel_overscan_size=20,
    ):

        scans = euclid.ScansEuclid.bottom_right(
            parallel_size=parallel_size,
            serial_size=serial_size,
            serial_prescan_size=serial_prescan_size,
            serial_overscan_size=serial_overscan_size,
            parallel_overscan_size=parallel_overscan_size,
        )

        return CIFrame.manual(
            array=array,
            pixel_scales=0.1,
            layout_ci=layout_ci,
            roe_corner=(1, 1),
            scans=scans,
        )
