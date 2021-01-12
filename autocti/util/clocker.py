from arcticpy import main


class Clocker(object):
    def __init__(
        self,
        iterations=1,
        parallel_roe=None,
        parallel_express=0,
        parallel_charge_injection_mode=False,
        parallel_window_range=None,
        serial_roe=None,
        serial_express=0,
        serial_window_range=None,
        time_window_range=None,
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
        self.parallel_window_range = parallel_window_range

        self.serial_roe = serial_roe
        self.serial_express = serial_express
        self.serial_window_range = serial_window_range

        self.time_window_range = time_window_range

    def add_cti(
        self,
        image,
        parallel_ccd=None,
        parallel_traps=None,
        serial_ccd=None,
        serial_traps=None,
    ):

        return main.add_cti(
            image=image,
            parallel_ccd=parallel_ccd,
            parallel_roe=self.parallel_roe,
            parallel_traps=parallel_traps,
            parallel_express=self.parallel_express,
            parallel_offset=image.readout_offsets[0],
            parallel_window_range=self.parallel_window_range,
            serial_ccd=serial_ccd,
            serial_roe=self.serial_roe,
            serial_traps=serial_traps,
            serial_express=self.serial_express,
            serial_offset=image.readout_offsets[1],
            serial_window_range=self.serial_window_range,
            time_window_range=self.time_window_range,
        )

    def remove_cti(
        self,
        image,
        parallel_ccd=None,
        parallel_traps=None,
        serial_ccd=None,
        serial_traps=None,
    ):
        return main.remove_cti(
            image=image,
            iterations=self.iterations,
            parallel_ccd=parallel_ccd,
            parallel_roe=self.parallel_roe,
            parallel_traps=parallel_traps,
            parallel_express=self.parallel_express,
            parallel_offset=image.readout_offsets[0],
            parallel_window_range=self.parallel_window_range,
            serial_ccd=serial_ccd,
            serial_roe=self.serial_roe,
            serial_traps=serial_traps,
            serial_express=self.serial_express,
            serial_offset=image.readout_offsets[1],
            serial_window_range=self.serial_window_range,
            time_window_range=self.time_window_range,
        )
