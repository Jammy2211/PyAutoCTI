from autocti.structures.frame import abstract_frame
from autocti.structures import frame as f
from autocti.structures import region as reg
from autocti.util import array_util
from autocti import exc

from astropy.io import fits


def array_converted_to_electrons_from_fits(file_path, quadrant_letter):

    if quadrant_letter is "A" or quadrant_letter is "B":
        hdu = 1
    elif quadrant_letter is "C" or quadrant_letter is "D":
        hdu = 4
    else:
        raise exc.FrameException("Quadrant letter for HSTFrame must be A, B, C or D.")

    array = array_util.numpy_array_2d_from_fits(
        file_path=file_path, hdu=hdu, do_not_scale_image_data=True
    )

    hdulist = fits.open(file_path)
    ext_header = hdulist[hdu].header

    units = ext_header["BUNIT"]
    bscale = ext_header["BSCALE"]
    bzero = ext_header["BZERO"]
    exposure_time = ext_header["EXPTIME"]

    if units in "COUNTS":
        return (array * bscale) + bzero, bscale, bzero, exposure_time
    elif units in "CPS":
        return (array * exposure_time * bscale) + bzero, bscale, bzero, exposure_time


class HSTFrame(f.Frame):
    """An ACS frame consists of four quadrants ('A', 'B', 'C', 'D') which have the following layout:

       <--------S-----------   ---------S----------->
    [] [========= 2 =========] [========= 3 =========] []          /\
    /    [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  /        |
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | Direction arctic
    P   [xxxxxxxxx B/C xxxxxxx] [xxxxxxxxx A/D xxxxxxx]  P         | clocks an image
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | without any rotation
    \/  [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  \/        | (e.g. towards row 0
                                                                   | of the NumPy arrays)

    For a HST .fits file:

    - The images contained in hdu 1 correspond to quadrants B (left) and A (right).
    - The images contained in hdu 4 correspond to quadrants C (left) and D (right).
    """

    @classmethod
    def from_fits(cls, file_path, quadrant_letter):
        """
        Use the input .fits file and quadrant letter to extract the quadrant from the full CCD, perform
        the rotations required to give correct arctic clocking and convert the image from units of COUNTS / CPS to
        ELECTRONS.

        See the docstring of the _HSTFrame_ class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """

        array, gain, gain_zero, exposure_time = array_converted_to_electrons_from_fits(
            file_path=file_path, quadrant_letter=quadrant_letter
        )

        return cls.from_ccd(
            array_electrons=array,
            quadrant_letter=quadrant_letter,
            gain=gain,
            gain_zero=gain_zero,
            exposure_time=exposure_time,
        )

    @classmethod
    def from_ccd(
        cls,
        array_electrons,
        quadrant_letter,
        parallel_size=2068,
        serial_size=2072,
        serial_prescan_size=24,
        parallel_overscan_size=20,
        gain=None,
        gain_zero=0.0,
        exposure_time=None,
    ):
        """
        Using an input array of both quadrants in electrons, use the quadrant letter to extract the quadrant from the
        full CCD and perform the rotations required to give correct arctic.

        See the docstring of the _HSTFrame_ class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """
        if quadrant_letter is "B" or quadrant_letter is "C":

            return cls.left(
                array_electrons=array_electrons[0:parallel_size, 0:serial_size],
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                parallel_overscan_size=parallel_overscan_size,
                gain=gain,
                gain_zero=gain_zero,
                exposure_time=exposure_time,
            )
        elif quadrant_letter is "A" or quadrant_letter is "D":
            return cls.right(
                array=array_electrons[0:parallel_size, serial_size : serial_size * 2],
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                parallel_overscan_size=parallel_overscan_size,
                gain=gain,
                gain_zero=gain_zero,
                exposure_time=exposure_time,
            )
        else:
            raise exc.FrameException(
                "Quadrant letter for HSTFrame must be A, B, C or D."
            )

    @classmethod
    def left(
        cls,
        array_electrons,
        parallel_size=2068,
        serial_size=2072,
        serial_prescan_size=24,
        parallel_overscan_size=20,
        gain=None,
        gain_zero=0.0,
        exposure_time=None,
    ):
        """
        Use an input array of the left quadrant in electrons and perform the rotations required to give correct
        arctic clocking.

        See the docstring of the _HSTFrame_ class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """
        parallel_overscan = reg.Region(
            (
                parallel_size - parallel_overscan_size,
                parallel_size,
                serial_prescan_size,
                serial_size,
            )
        )

        serial_prescan = reg.Region((0, parallel_size, 0, serial_prescan_size))

        return f.Frame.manual(
            array=array_electrons,
            roe_corner=(1, 0),
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            gain=gain,
            gain_zero=gain_zero,
            exposure_time=exposure_time,
            pixel_scales=0.05,
        )

    @classmethod
    def right(
        cls,
        array,
        parallel_size=2068,
        serial_size=2072,
        parallel_overscan_size=20,
        serial_prescan_size=51,
        gain=None,
        gain_zero=0.0,
        exposure_time=None,
    ):
        """
        Use an input array of the right quadrant in electrons and perform the rotations required to give correct
        arctic clocking.

        See the docstring of the _HSTFrame_ class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """
        parallel_overscan = reg.Region(
            (
                parallel_size - parallel_overscan_size,
                parallel_size,
                0,
                serial_size - serial_prescan_size,
            )
        )

        serial_prescan = reg.Region(
            (0, parallel_size, serial_size - serial_prescan_size, serial_size)
        )

        return f.Frame.manual(
            array=array,
            roe_corner=(1, 1),
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            gain=gain,
            gain_zero=gain_zero,
            exposure_time=exposure_time,
            pixel_scales=0.05,
        )


class MaskedHSTFrame(abstract_frame.AbstractFrame):
    @classmethod
    def from_fits(cls, file_path, quadrant_letter, mask):
        """
        Use the input .fits file and quadrant letter to extract the quadrant from the full CCD, perform
        the rotations required to give correct arctic clocking and convert the image from units of COUNTS / CPS to
        ELECTRONS.

        A mask is input which is subject to the same extraction and rotations.

        See the docstring of the _HSTFrame_ class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """
        array = array_converted_to_electrons_from_fits(
            file_path=file_path, quadrant_letter=quadrant_letter
        )

        return cls.from_ccd(
            array=array,
            quadrant_letter=quadrant_letter,
            mask=mask,
            gain=gain,
            gain_zero=gain_zero,
            exposure_time=exposure_time,
        )

    @classmethod
    def from_ccd(
        cls,
        array,
        mask,
        quadrant_letter,
        parallel_size=2068,
        serial_size=2072,
        serial_prescan_size=24,
        parallel_overscan_size=20,
        gain=None,
        gain_zero=0.0,
        exposure_time=None,
    ):
        """
        Using an input array of both quadrants in electrons, use the quadrant letter to extract the quadrant from the
        full CCD and perform the rotations required to give correct arctic.

        A mask is input which is subject to the same extraction and rotations.

        See the docstring of the _HSTFrame_ class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """
        if quadrant_letter is "B" or quadrant_letter is "C":

            return cls.left(
                array=array[0:parallel_size, 0:serial_size],
                mask=mask[0:parallel_size, 0:serial_size],
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                parallel_overscan_size=parallel_overscan_size,
                gain=gain,
                gain_zero=gain_zero,
                exposure_time=exposure_time,
            )
        elif quadrant_letter is "A" or quadrant_letter is "D":
            return cls.right(
                array=array[0:parallel_size, serial_size : serial_size * 2],
                mask=mask[0:parallel_size, serial_size : serial_size * 2],
                parallel_size=parallel_size,
                serial_size=serial_size,
                serial_prescan_size=serial_prescan_size,
                parallel_overscan_size=parallel_overscan_size,
                gain=gain,
                gain_zero=gain_zero,
                exposure_time=exposure_time,
            )
        else:
            raise exc.FrameException(
                "Quadrant letter for HSTFrame must be A, B, C or D."
            )

    @classmethod
    def left(
        cls,
        array,
        mask,
        parallel_size=2068,
        serial_size=2072,
        serial_prescan_size=24,
        parallel_overscan_size=20,
        gain=None,
        gain_zero=0.0,
        exposure_time=None,
    ):
        """
        Use an input array of the left quadrant in electrons and perform the rotations required to give correct
        arctic clocking.

        A mask is input which is subject to the same extraction and rotations.

        See the docstring of the _HSTFrame_ class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """
        parallel_overscan = reg.Region(
            (
                parallel_size - parallel_overscan_size,
                parallel_size,
                serial_prescan_size,
                serial_size,
            )
        )

        serial_prescan = reg.Region((0, parallel_size, 0, serial_prescan_size))

        return f.MaskedFrame.manual(
            array=array,
            mask=mask,
            roe_corner=(1, 0),
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            gain=gain,
            gain_zero=gain_zero,
            exposure_time=exposure_time,
        )

    @classmethod
    def right(
        cls,
        array,
        mask,
        parallel_size=2068,
        serial_size=2072,
        parallel_overscan_size=20,
        serial_prescan_size=51,
        gain=None,
        gain_zero=0.0,
        exposure_time=None,
    ):
        """
        Use an input array of the right quadrant in electrons and perform the rotations required to give correct
        arctic clocking.

        A mask is input which is subject to the same extraction and rotations.

        See the docstring of the _HSTFrame_ class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """
        parallel_overscan = reg.Region(
            (
                parallel_size - parallel_overscan_size,
                parallel_size,
                0,
                serial_size - serial_prescan_size,
            )
        )

        serial_prescan = reg.Region(
            (0, parallel_size, serial_size - serial_prescan_size, serial_size)
        )

        return f.MaskedFrame.manual(
            array=array,
            mask=mask,
            roe_corner=(1, 1),
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            gain=gain,
            gain_zero=gain_zero,
            exposure_time=exposure_time,
        )
