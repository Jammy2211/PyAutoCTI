import logging
import os
from os import path

from autoarray import exc
from autoarray.structures.arrays import array_2d_util

from autocti.instruments.acs.array_2d import Array2DACS
from autocti.instruments.acs.header import HeaderACS

from autocti.instruments.acs import acs_util

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel("INFO")


class ImageACS(Array2DACS):
    """
    The layout of an ACS array and image is given in `FrameACS`.

    This class handles specifically the image of an ACS observation, assuming that it contains specific
    header info.
    """

    @classmethod
    def from_fits(
        cls,
        file_path,
        quadrant_letter,
        bias_subtract_via_bias_file=False,
        bias_subtract_via_prescan=False,
        bias_file_path=None,
        use_calibrated_gain=True,
    ):
        """
        Use the input .fits file and quadrant letter to extract the quadrant from the full CCD, perform
        the rotations required to give correct arctic clocking and convert the image from units of COUNTS / CPS to
        ELECTRONS.

        See the docstring of the `FrameACS` class for a complete description of the HST FPA, quadrants and
        rotations.

        Also see https://github.com/spacetelescope/hstcal/blob/main/pkg/acs/calacs/acscte/dopcte-gen2.c#L418

        Parameters
        ----------
        file_path
            The full path of the file that the image is loaded from, including the file name and ``.fits`` extension.
        quadrant_letter
            The letter of the ACS quadrant the image is extracted from and loaded.
        bias_subtract_via_bias_file
            If True, the corresponding bias file of the image is loaded (via the name of the file in the fits header).
        bias_subtract_via_prescan
            If True, the prescan on the image is used to estimate a component of bias that is subtracted from the image.
        bias_file_path
            If `bias_subtract_via_bias_file=True`, this overwrites the path to the bias file instead of the default
            behaviour of using the .fits header.
        use_calibrated_gain
            If True, the calibrated gain values are used to convert from COUNTS to ELECTRONS.
        """

        hdu = acs_util.fits_hdu_via_quadrant_letter_from(
            quadrant_letter=quadrant_letter
        )

        header_sci_obj = array_2d_util.header_obj_from(file_path=file_path, hdu=0)
        header_hdu_obj = array_2d_util.header_obj_from(file_path=file_path, hdu=hdu)

        header = HeaderACS(
            header_sci_obj=header_sci_obj,
            header_hdu_obj=header_hdu_obj,
            hdu=hdu,
            quadrant_letter=quadrant_letter,
        )

        if header.header_sci_obj["TELESCOP"] != "HST":
            raise exc.ArrayException(
                f"The file {file_path} does not point to a valid HST ACS dataset."
            )

        if header.header_sci_obj["INSTRUME"] != "ACS":
            raise exc.ArrayException(
                f"The file {file_path} does not point to a valid HST ACS dataset."
            )

        array = array_2d_util.numpy_array_2d_via_fits_from(
            file_path=file_path, hdu=hdu, do_not_scale_image_data=True
        )

        array = header.array_original_to_electrons(
            array=array, use_calibrated_gain=use_calibrated_gain
        )

        if bias_subtract_via_bias_file:
            if bias_file_path is None:
                file_dir = os.path.split(file_path)[0]
                bias_file_path = path.join(file_dir, header.bias_file)

            bias = array_2d_util.numpy_array_2d_via_fits_from(
                file_path=bias_file_path, hdu=hdu, do_not_scale_image_data=True
            )

            header_sci_obj = array_2d_util.header_obj_from(
                file_path=bias_file_path, hdu=0
            )
            header_hdu_obj = array_2d_util.header_obj_from(
                file_path=bias_file_path, hdu=hdu
            )

            bias_header = HeaderACS(
                header_sci_obj=header_sci_obj,
                header_hdu_obj=header_hdu_obj,
                hdu=hdu,
                quadrant_letter=quadrant_letter,
            )

            if bias_header.original_units != "COUNTS":
                raise exc.ArrayException("Cannot use bias frame not in counts.")

            bias = bias * bias_header.calibrated_gain

        else:
            bias = None

        return cls.from_ccd(
            array_electrons=array,
            quadrant_letter=quadrant_letter,
            header=header,
            bias_subtract_via_prescan=bias_subtract_via_prescan,
            bias=bias,
        )
