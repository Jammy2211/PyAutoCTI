from autoarray.structures.arrays.two_d import array_2d
from arcticpy.src import cti, ccd, roe


class Clocker(object):
    def __init__(
        self,
        iterations=1,
        parallel_roe=roe.ROE(),
        parallel_express=0,
        parallel_charge_injection_mode=False,
        parallel_window_start=0,
        parallel_window_stop=-1,
        serial_roe=roe.ROE(),
        serial_express=0,
        serial_window_start=0,
        serial_window_stop=-1,
        verbosity=0,
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
             trailed by traps over the entire CCDPhase (as opposed to its distance from the CCDPhase register).
        parallel_readout_offset : int
            Introduces an offset which increases the number of transfers each pixel takes in the parallel direction.
        """

        self.iterations = iterations

        self.parallel_roe = parallel_roe
        self.parallel_express = parallel_express
        self.parallel_charge_injection_mode = parallel_charge_injection_mode
        self.parallel_window_start = parallel_window_start
        self.parallel_window_stop = parallel_window_stop

        self.serial_roe = serial_roe
        self.serial_express = serial_express
        self.serial_window_start = serial_window_start
        self.serial_window_stop = serial_window_stop

        self.verbosity = verbosity

    def add_cti(
        self,
        image_pre_cti,
        parallel_ccd=None,
        parallel_traps=None,
        serial_ccd=None,
        serial_traps=None,
    ):

        if parallel_ccd is not None:
            parallel_ccd = ccd.CCD(
                phases=[parallel_ccd], fraction_of_traps_per_phase=[1.0]
            )

        if serial_ccd is not None:
            serial_ccd = ccd.CCD(phases=[serial_ccd], fraction_of_traps_per_phase=[1.0])

        image_post_cti = cti.add_cti(
            image=image_pre_cti,
            parallel_ccd=parallel_ccd,
            parallel_roe=self.parallel_roe,
            parallel_traps=parallel_traps,
            parallel_express=self.parallel_express,
            parallel_offset=image_pre_cti.readout_offsets[0],
            parallel_window_start=self.parallel_window_start,
            parallel_window_stop=self.parallel_window_stop,
            serial_ccd=serial_ccd,
            serial_roe=self.serial_roe,
            serial_traps=serial_traps,
            serial_express=self.serial_express,
            serial_offset=image_pre_cti.readout_offsets[1],
            serial_window_start=self.serial_window_start,
            serial_window_stop=self.serial_window_stop,
            verbosity=self.verbosity,
        )

        return array_2d.Array2D.manual_mask(
            array=image_post_cti, mask=image_pre_cti.mask
        ).native

    def remove_cti(
        self,
        image,
        parallel_ccd=None,
        parallel_traps=None,
        serial_ccd=None,
        serial_traps=None,
    ):
        return cti.remove_cti(
            image=image,
            n_iterations=self.iterations,
            parallel_ccd=parallel_ccd,
            parallel_roe=self.parallel_roe,
            parallel_traps=parallel_traps,
            parallel_express=self.parallel_express,
            parallel_offset=image.readout_offsets[0],
            parallel_window_start=self.parallel_window_start,
            parallel_window_stop=self.parallel_window_stop,
            serial_ccd=serial_ccd,
            serial_roe=self.serial_roe,
            serial_traps=serial_traps,
            serial_express=self.serial_express,
            serial_offset=image.readout_offsets[1],
            serial_window_start=self.serial_window_start,
            serial_window_stop=self.serial_window_stop,
        )
