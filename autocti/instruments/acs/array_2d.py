from astropy.io import fits
import logging
import numpy as np
import os
import shutil

from autoarray.structures.arrays.uniform_2d import Array2D

from autoarray import exc
from autoarray.structures.arrays import array_2d_util
from autoarray.layout import layout_util

from autocti.instruments.acs import acs_util

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel("INFO")


class Array2DACS(Array2D):
    """
    An ACS array consists of four quadrants ('A', 'B', 'C', 'D') which have the following layout (which are described
    at the following STScI
    link https://github.com/spacetelescope/hstcal/blob/main/pkg/acs/calacs/acscte/dopcte-gen2.c#L418).

       <--------S-----------   ---------S----------->
    [] [========= 2 =========] [========= 3 =========] []          /\
    /    [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  /        |
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | Direction arctic
    P   [xxxxxxxxx B/C xxxxxxx] [xxxxxxxxx A/D xxxxxxx]  P         | clocks an image
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | without any rotation
    \/  [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  \/        | (e.g. towards row 0
                                                                   | of the NumPy arrays)

    For a ACS .fits file:

    - The images contained in hdu 1 correspond to quadrants B (left) and A (right).
    - The images contained in hdu 4 correspond to quadrants C (left) and D (right).
    """

    @classmethod
    def from_fits(cls, file_path, quadrant_letter):
        """
        Use the input .fits file and quadrant letter to extract the quadrant from the full CCD, perform
        the rotations required to give correct arctic clocking and convert the image from units of COUNTS / CPS to
        ELECTRONS.

        See the docstring of the `FrameACS` class for a complete description of the HST FPA, quadrants and
        rotations.

        Also see https://github.com/spacetelescope/hstcal/blob/main/pkg/acs/calacs/acscte/dopcte-gen2.c#L418
        """

        hdu = acs_util.fits_hdu_via_quadrant_letter_from(
            quadrant_letter=quadrant_letter
        )

        array = array_2d_util.numpy_array_2d_via_fits_from(file_path=file_path, hdu=hdu)

        return cls.from_ccd(array_electrons=array, quadrant_letter=quadrant_letter)

    @classmethod
    def from_ccd(
        cls,
        array_electrons,
        quadrant_letter,
        header=None,
        bias_subtract_via_prescan=False,
        bias=None,
    ):
        """
        Using an input array of both quadrants in electrons, use the quadrant letter to extract the quadrant from the
        full CCD and perform the rotations required to give correct arctic.

        See the docstring of the `FrameACS` class for a complete description of the HST FPA, quadrants and
        rotations.

        Also see https://github.com/spacetelescope/hstcal/blob/main/pkg/acs/calacs/acscte/dopcte-gen2.c#L418
        """
        if quadrant_letter == "A":
            array_electrons = array_electrons[0:2068, 0:2072]
            roe_corner = (1, 0)
            use_flipud = True

            if bias is not None:
                bias = bias[0:2068, 0:2072]

        elif quadrant_letter == "B":
            array_electrons = array_electrons[0:2068, 2072:4144]
            roe_corner = (1, 1)
            use_flipud = True

            if bias is not None:
                bias = bias[0:2068, 2072:4144]

        elif quadrant_letter == "C":
            array_electrons = array_electrons[0:2068, 0:2072]

            roe_corner = (1, 0)
            use_flipud = False

            if bias is not None:
                bias = bias[0:2068, 0:2072]

        elif quadrant_letter == "D":
            array_electrons = array_electrons[0:2068, 2072:4144]

            roe_corner = (1, 1)
            use_flipud = False

            if bias is not None:
                bias = bias[0:2068, 2072:4144]

        else:
            raise exc.ArrayException(
                "Quadrant letter for FrameACS must be A, B, C or D."
            )

        return cls.quadrant_a(
            array_electrons=array_electrons,
            header=header,
            roe_corner=roe_corner,
            use_flipud=use_flipud,
            bias_subtract_via_prescan=bias_subtract_via_prescan,
            bias=bias,
        )

    @classmethod
    def quadrant_a(
        cls,
        array_electrons,
        roe_corner,
        use_flipud,
        header=None,
        bias_subtract_via_prescan=False,
        bias=None,
    ):
        """
        Use an input array of the left quadrant in electrons and perform the rotations required to give correct
        arctic clocking.

        See the docstring of the `FrameACS` class for a complete description of the HST FPA, quadrants and
        rotations.

        Also see https://github.com/spacetelescope/hstcal/blob/main/pkg/acs/calacs/acscte/dopcte-gen2.c#L418
        """

        array_electrons = layout_util.rotate_array_via_roe_corner_from(
            array=array_electrons, roe_corner=roe_corner
        )

        if use_flipud:
            array_electrons = np.flipud(array_electrons)

        if bias_subtract_via_prescan:
            bias_serial_prescan_value = acs_util.prescan_fitted_bias_column(
                array_electrons[:, 18:24]
            )

            array_electrons -= bias_serial_prescan_value

            header.bias_serial_prescan_column = bias_serial_prescan_value

        if bias is not None:
            bias = layout_util.rotate_array_via_roe_corner_from(
                array=bias, roe_corner=roe_corner
            )

            if use_flipud:
                bias = np.flipud(bias)

            array_electrons -= bias

            header.bias = Array2DACS.no_mask(values=bias, pixel_scales=0.05)

        return cls.no_mask(values=array_electrons, header=header, pixel_scales=0.05)

    @classmethod
    def quadrant_b(
        cls, array_electrons, header=None, bias_subtract_via_prescan=False, bias=None
    ):
        """
        Use an input array of the right quadrant in electrons and perform the rotations required to give correct
        arctic clocking.

        See the docstring of the `FrameACS` class for a complete description of the HST FPA, quadrants and
        rotations.

        Also see https://github.com/spacetelescope/hstcal/blob/main/pkg/acs/calacs/acscte/dopcte-gen2.c#L418
        """

        array_electrons = layout_util.rotate_array_via_roe_corner_from(
            array=array_electrons, roe_corner=(1, 1)
        )

        array_electrons = np.flipud(array_electrons)

        if bias_subtract_via_prescan:
            bias_serial_prescan_value = acs_util.prescan_fitted_bias_column(
                array_electrons[:, 18:24]
            )

            array_electrons -= bias_serial_prescan_value

            header.bias_serial_prescan_column = bias_serial_prescan_value

        if bias is not None:
            bias = layout_util.rotate_array_via_roe_corner_from(
                array=bias, roe_corner=(1, 1)
            )

            bias = np.flipud(bias)

            array_electrons -= bias

            header.bias = Array2DACS.no_mask(values=bias, pixel_scales=0.05)

        return cls.no_mask(values=array_electrons, header=header, pixel_scales=0.05)

    @classmethod
    def quadrant_c(
        cls, array_electrons, header=None, bias_subtract_via_prescan=False, bias=None
    ):
        """
        Use an input array of the left quadrant in electrons and perform the rotations required to give correct
        arctic clocking.

        See the docstring of the `FrameACS` class for a complete description of the HST FPA, quadrants and
        rotations.

        Also see https://github.com/spacetelescope/hstcal/blob/main/pkg/acs/calacs/acscte/dopcte-gen2.c#L418
        """

        array_electrons = layout_util.rotate_array_via_roe_corner_from(
            array=array_electrons, roe_corner=(1, 0)
        )

        if bias_subtract_via_prescan:
            bias_serial_prescan_value = acs_util.prescan_fitted_bias_column(
                array_electrons[:, 18:24]
            )

            array_electrons -= bias_serial_prescan_value

            header.bias_serial_prescan_column = bias_serial_prescan_value

        if bias is not None:
            bias = layout_util.rotate_array_via_roe_corner_from(
                array=bias, roe_corner=(1, 0)
            )

            array_electrons -= bias

            header.bias = Array2DACS.no_mask(values=bias, pixel_scales=0.05)

        return cls.no_mask(values=array_electrons, header=header, pixel_scales=0.05)

    @classmethod
    def quadrant_d(
        cls, array_electrons, header=None, bias_subtract_via_prescan=False, bias=None
    ):
        """
        Use an input array of the right quadrant in electrons and perform the rotations required to give correct
        arctic clocking.

        See the docstring of the `FrameACS` class for a complete description of the HST FPA, quadrants and
        rotations.

        Also see https://github.com/spacetelescope/hstcal/blob/main/pkg/acs/calacs/acscte/dopcte-gen2.c#L418
        """

        array_electrons = layout_util.rotate_array_via_roe_corner_from(
            array=array_electrons, roe_corner=(1, 1)
        )

        if bias_subtract_via_prescan:
            bias_serial_prescan_value = acs_util.prescan_fitted_bias_column(
                array_electrons[:, 18:24]
            )

            array_electrons -= bias_serial_prescan_value

            header.bias_serial_prescan_column = bias_serial_prescan_value

        if bias is not None:
            bias = layout_util.rotate_array_via_roe_corner_from(
                array=bias, roe_corner=(1, 1)
            )

            array_electrons -= bias

            header.bias = Array2DACS.no_mask(values=bias, pixel_scales=0.05)

        return cls.no_mask(values=array_electrons, header=header, pixel_scales=0.05)

    def update_fits(self, original_file_path, new_file_path):
        """
        Output the array to a .fits file.

        Parameters
        ----------
        file_path
            The path the file is output to, including the filename and the ``.fits`` extension,
            e.g. '/path/to/filename.fits'
        """

        new_file_dir = os.path.split(new_file_path)[0]

        if not os.path.exists(new_file_dir):
            os.makedirs(new_file_dir)

        if not os.path.exists(new_file_path):
            shutil.copy(original_file_path, new_file_path)

        hdulist = fits.open(new_file_path)

        hdulist[self.header.hdu].data = self.layout_2d.original_orientation_from(
            array=self
        )

        ext_header = hdulist[4].header
        bscale = ext_header["BSCALE"]

        os.remove(new_file_path)

        hdulist.writeto(new_file_path)
