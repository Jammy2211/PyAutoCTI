import arcticpy


class Clocker(object):
    def __init__(
        self,
        iterations=1,
        parallel_roe=None,
        parallel_express=0,
        parallel_charge_injection_mode=False,
        parallel_offset=0,
        parallel_window=None,
        serial_roe=None,
        serial_express=0,
        serial_offset=0,
        serial_window=None,
        time_window=[0, 1],
    ):
        """
        The CTI Clock for arctic clocking.

        Parameters
        ----------
        parallel_sequence : [float]
            The array or single value of the time between clock shifts.
        iterations : int
            If CTI is being corrected, iterations determines the number of times clocking is run to perform the \
            correction via forward modeling. For adding CTI only one run is required and iterations is ignored.
        parallel_express : int
            The factor by which pixel-to-pixel transfers are combined for efficiency.
        parallel_charge_injection_mode : bool
            If True, clocking is performed in charge injection line mode, where each pixel is clocked and therefore \
             trailed by traps over the entire CCD (as opposed to its distance from the CCD register).
        parallel_readout_offset : int
            Introduces an offset which increases the number of transfers each pixel takes in the parallel direction.
        """

        self.iterations = iterations

        self.parallel_roe = parallel_roe
        self.parallel_express = parallel_express
        self.parallel_charge_injection_mode = parallel_charge_injection_mode
        self.parallel_offset = parallel_offset
        self.parallel_window = parallel_window

        self.serial_roe = serial_roe
        self.serial_express = serial_express
        self.serial_offset = serial_offset
        self.serial_window = serial_window

        self.time_window = time_window

    def add_cti(
        self,
        image,
        parallel_ccd=None,
        parallel_traps=None,
        serial_ccd=None,
        serial_traps=None,
    ):
        return arcticpy.add_cti(
            image=image,
            parallel_ccd=parallel_ccd,
            parallel_roe=self.parallel_roe,
            parallel_traps=parallel_traps,
            parallel_express=self.parallel_express,
            parallel_offset=self.parallel_offset,
            parallel_window=self.parallel_window,
            serial_ccd=serial_ccd,
            serial_roe=self.serial_roe,
            serial_traps=serial_traps,
            serial_express=self.serial_express,
            serial_offset=self.serial_offset,
            serial_window=self.serial_window,
            time_window=self.time_window,
        )

    def remove_cti(
        self,
        image,
        parallel_ccd=None,
        parallel_traps=None,
        serial_ccd=None,
        serial_traps=None,
    ):
        return arcticpy.remove_cti(
            image=image,
            parallel_ccd=parallel_ccd,
            parallel_roe=self.parallel_roe,
            parallel_traps=parallel_traps,
            parallel_express=self.parallel_express,
            parallel_offset=self.parallel_offset,
            parallel_window=self.parallel_window,
            serial_ccd=serial_ccd,
            serial_roe=self.serial_roe,
            serial_traps=serial_traps,
            serial_express=self.serial_express,
            serial_offset=self.serial_offset,
            serial_window=self.serial_window,
            time_window=self.time_window,
        )


class CCD(arcticpy.CCD):
    def __init__(self, full_well_depth=1e4, well_notch_depth=0.0, well_fill_power=0.58):
        super().__init__(
            fraction_of_traps_per_phase=[1],
            full_well_depth=full_well_depth,
            well_fill_power=well_fill_power,
            well_notch_depth=well_notch_depth,
            well_bloom_level=None,
        )


class TrapInstantCapture(arcticpy.TrapInstantCapture):
    """ For the old C++ style release-then-instant-capture algorithm. """

    def __init__(self, density=0.13, release_timescale=0.25):
        """ The parameters for a single trap species.

        Parameters
        ----------
        density : float
            The density of the trap species in a pixel.
        release_timescale : float
            The release timescale of the trap, in the same units as the time
            spent in each pixel or phase (Clocker sequence).
        surface : bool
            ###
        """
        super().__init__(
            density=density, release_timescale=release_timescale, surface=False
        )
