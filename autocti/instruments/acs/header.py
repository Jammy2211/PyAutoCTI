import logging

from autoarray.structures.header import Header

from autoarray import exc

from autocti.instruments.acs import acs_util

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel("INFO")


class HeaderACS(Header):
    def __init__(
        self,
        header_sci_obj,
        header_hdu_obj,
        quadrant_letter=None,
        hdu=None,
        bias=None,
        bias_serial_prescan_column=None,
    ):
        super().__init__(header_sci_obj=header_sci_obj, header_hdu_obj=header_hdu_obj)

        self.bias = bias
        self.bias_serial_prescan_column = bias_serial_prescan_column
        self.quadrant_letter = quadrant_letter
        self.hdu = hdu

    @property
    def bscale(self):
        return self.header_hdu_obj["BSCALE"]

    @property
    def bzero(self):
        return self.header_hdu_obj["BZERO"]

    @property
    def gain(self):
        return self.header_sci_obj["CCDGAIN"]

    @property
    def calibrated_gain(self):
        if round(self.gain) == 1:
            calibrated_gain = [0.99989998, 0.97210002, 1.01070000, 1.01800000]
        elif round(self.gain) == 2:
            calibrated_gain = [2.002, 1.945, 2.028, 1.994]
        elif round(self.gain) == 4:
            calibrated_gain = [4.011, 3.902, 4.074, 3.996]
        else:
            raise exc.ArrayException(
                "Calibrated gain of ACS file does not round to 1, 2 or 4."
            )

        if self.quadrant_letter == "A":
            return calibrated_gain[0]
        elif self.quadrant_letter == "B":
            return calibrated_gain[1]
        elif self.quadrant_letter == "C":
            return calibrated_gain[2]
        elif self.quadrant_letter == "D":
            return calibrated_gain[3]

    @property
    def original_units(self):
        return self.header_hdu_obj["BUNIT"]

    @property
    def bias_file(self):
        return self.header_sci_obj["BIASFILE"].replace("jref$", "")

    def array_eps_to_counts(self, array_eps):
        return acs_util.array_eps_to_counts(
            array_eps=array_eps, bscale=self.bscale, bzero=self.bzero
        )

    def array_original_to_electrons(self, array, use_calibrated_gain):
        if self.original_units in "COUNTS":
            array = (array * self.bscale) + self.bzero
        elif self.original_units in "CPS":
            array = (array * self.exposure_time * self.bscale) + self.bzero

        if use_calibrated_gain:
            return array * self.calibrated_gain
        else:
            return array * self.gain

    def array_electrons_to_original(self, array, use_calibrated_gain):
        if use_calibrated_gain:
            array /= self.calibrated_gain
        else:
            array /= self.gain

        if self.original_units in "COUNTS":
            return (array - self.bzero) / self.bscale
        elif self.original_units in "CPS":
            return (array - self.bzero) / (self.exposure_time * self.bscale)
