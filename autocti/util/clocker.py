import numpy as np

from autoarray.structures.arrays.one_d.array_1d import Array1D
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autocti import exc

from arcticpy.src import cti
from arcticpy.src.ccd import CCD
from arcticpy.src import roe


class AbstractClocker:
    def __init__(self, iterations=1, verbosity=0):

        self.iterations = iterations
        self.verbosity = verbosity

    def ccd_from(self, ccd_phase):
        if ccd_phase is not None:
            return CCD(phases=[ccd_phase], fraction_of_traps_per_phase=[1.0])


class Clocker1D(AbstractClocker):
    def __init__(
        self,
        iterations=1,
        roe=roe.ROE(),
        express=0,
        charge_injection_mode=False,
        window_start=0,
        window_stop=-1,
        verbosity=0,
    ):
        """
        The CTI Clock for arctic clocking.
        Parameters
        ----------
        parallel_sequence
            The array or single value of the time between clock shifts.
        iterations
            If CTI is being corrected, iterations determines the number of times clocking is run to perform the \
            correction via forward modeling. For adding CTI only one run is required and iterations is ignored.
        parallel_express
            The factor by which pixel-to-pixel transfers are combined for efficiency.
        parallel_charge_injection_mode
            If True, clocking is performed in charge injection line mode, where each pixel is clocked and therefore \
             trailed by traps over the entire CCDPhase (as opposed to its distance from the CCDPhase register).
        parallel_readout_offset
            Introduces an offset which increases the number of transfers each pixel takes in the parallel direction.
        """

        super().__init__(iterations=iterations, verbosity=verbosity)

        self.roe = roe
        self.express = express
        self.charge_injection_mode = charge_injection_mode
        self.window_start = window_start
        self.window_stop = window_stop

    def add_cti(self, data, ccd=None, trap_list=None):

        if not any([trap_list]):
            raise exc.ClockerException(
                "No Trap species were passed to the add_cti method"
            )

        if not any([ccd]):
            raise exc.ClockerException("No CCD object was passed to the add_cti method")

        image_pre_cti_2d = Array2D.zeros(
            shape_native=(data.shape_native[0], 1), pixel_scales=data.pixel_scales
        ).native

        image_pre_cti_2d[:, 0] = data

        ccd = self.ccd_from(ccd_phase=ccd)

        image_post_cti = cti.add_cti(
            image=image_pre_cti_2d,
            parallel_ccd=ccd,
            parallel_roe=self.roe,
            parallel_traps=trap_list,
            parallel_express=self.express,
            parallel_offset=data.readout_offsets[0],
            parallel_window_start=self.window_start,
            parallel_window_stop=self.window_stop,
            verbosity=self.verbosity,
        )

        return Array1D.manual_native(
            array=image_post_cti.flatten(), pixel_scales=data.pixel_scales
        )


class Clocker2D(AbstractClocker):
    def __init__(
        self,
        iterations=1,
        parallel_roe=roe.ROE(),
        parallel_express=0,
        parallel_charge_injection_mode=False,
        parallel_window_start=0,
        parallel_window_stop=-1,
        parallel_poisson_traps=False,
        serial_roe=roe.ROE(),
        serial_express=0,
        serial_window_start=0,
        serial_window_stop=-1,
        verbosity=0,
        poisson_seed=-1,
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

        super().__init__(iterations=iterations, verbosity=verbosity)

        self.parallel_roe = parallel_roe
        self.parallel_express = parallel_express
        self.parallel_charge_injection_mode = parallel_charge_injection_mode
        self.parallel_window_start = parallel_window_start
        self.parallel_window_stop = parallel_window_stop
        self.parallel_poisson_traps = parallel_poisson_traps

        self.serial_roe = serial_roe
        self.serial_express = serial_express
        self.serial_window_start = serial_window_start
        self.serial_window_stop = serial_window_stop

        self.poisson_seed = poisson_seed

    def add_cti(
        self,
        data,
        parallel_ccd=None,
        parallel_trap_list=None,
        serial_ccd=None,
        serial_trap_list=None,
    ):

        if self.parallel_poisson_traps:
            return self.add_cti_poisson_traps(
                data=data,
                parallel_ccd=parallel_ccd,
                parallel_trap_list=parallel_trap_list,
                serial_ccd=serial_ccd,
                serial_trap_list=serial_trap_list
            )

        if not any([parallel_trap_list, serial_trap_list]):
            raise exc.ClockerException(
                "No Trap species (parallel or serial) were passed to the add_cti method"
            )

        if not any([parallel_ccd, serial_ccd]):
            raise exc.ClockerException(
                "No CCD object(parallel or serial) was passed to the add_cti method"
            )

        parallel_ccd = self.ccd_from(ccd_phase=parallel_ccd)
        serial_ccd = self.ccd_from(ccd_phase=serial_ccd)

        try:
            parallel_offset = data.readout_offsets[0]
            serial_offset = data.readout_offsets[1]
        except AttributeError:
            parallel_offset = 0
            serial_offset = 0

        image_post_cti = cti.add_cti(
            image=data,
            parallel_ccd=parallel_ccd,
            parallel_roe=self.parallel_roe,
            parallel_traps=parallel_trap_list,
            parallel_express=self.parallel_express,
            parallel_offset=parallel_offset,
            parallel_window_start=self.parallel_window_start,
            parallel_window_stop=self.parallel_window_stop,
            serial_ccd=serial_ccd,
            serial_roe=self.serial_roe,
            serial_traps=serial_trap_list,
            serial_express=self.serial_express,
            serial_offset=serial_offset,
            serial_window_start=self.serial_window_start,
            serial_window_stop=self.serial_window_stop,
            verbosity=self.verbosity,
        )

        try:
            return Array2D.manual_mask(array=image_post_cti, mask=data.mask).native
        except AttributeError:
            return image_post_cti

    def remove_cti(
        self,
        data,
        parallel_ccd=None,
        parallel_trap_list=None,
        serial_ccd=None,
        serial_trap_list=None,
    ):

        if not any([parallel_trap_list, serial_trap_list]):
            raise exc.ClockerException(
                "No Trap species (parallel or serial) were passed to the add_cti method"
            )

        if not any([parallel_ccd, serial_ccd]):
            raise exc.ClockerException(
                "No CCD object(parallel or serial) was passed to the add_cti method"
            )

        parallel_ccd = self.ccd_from(ccd_phase=parallel_ccd)
        serial_ccd = self.ccd_from(ccd_phase=serial_ccd)

        image_cti_removed = cti.remove_cti(
            image=data,
            n_iterations=self.iterations,
            parallel_ccd=parallel_ccd,
            parallel_roe=self.parallel_roe,
            parallel_traps=parallel_trap_list,
            parallel_express=self.parallel_express,
            parallel_offset=data.readout_offsets[0],
            parallel_window_start=self.parallel_window_start,
            parallel_window_stop=self.parallel_window_stop,
            serial_ccd=serial_ccd,
            serial_roe=self.serial_roe,
            serial_traps=serial_trap_list,
            serial_express=self.serial_express,
            serial_offset=data.readout_offsets[1],
            serial_window_start=self.serial_window_start,
            serial_window_stop=self.serial_window_stop,
        )

        return Array2D.manual_mask(
            array=image_cti_removed, mask=data.mask, header=data.header
        ).native

    def add_cti_poisson_traps(
        self,
        data,
        parallel_ccd=None,
        parallel_trap_list=None,
        serial_ccd=None,
        serial_trap_list=None,
    ):

        if not any([parallel_trap_list, serial_trap_list]):
            raise exc.ClockerException(
                "No Trap species (parallel or serial) were passed to the add_cti method"
            )

        if not any([parallel_ccd, serial_ccd]):
            raise exc.ClockerException(
                "No CCD object(parallel or serial) was passed to the add_cti method"
            )

        parallel_ccd = self.ccd_from(ccd_phase=parallel_ccd)

        try:
            parallel_offset = data.readout_offsets[0]
        except AttributeError:
            parallel_offset = 0

        image_pre_cti = data.native
        image_post_cti = np.zeros(data.shape_native)

        total_rows = image_post_cti.shape[0]
        total_columns = image_post_cti.shape[1]

        for column in range(total_columns):

            parallel_trap_poisson_list = [
                parallel_trap.poisson_density_from(total_pixels=total_rows, seed=self.poisson_seed)
                for parallel_trap in parallel_trap_list
            ]

            image_pre_cti_pass = np.zeros(shape=(total_rows, 1))
            image_pre_cti_pass[:, 0] = image_pre_cti[:, column]

            image_post_cti[:, column] = cti.add_cti(
                image=image_pre_cti_pass,
                parallel_ccd=parallel_ccd,
                parallel_roe=self.parallel_roe,
                parallel_traps=parallel_trap_poisson_list,
                parallel_express=self.parallel_express,
                parallel_offset=parallel_offset,
                parallel_window_start=self.parallel_window_start,
                parallel_window_stop=self.parallel_window_stop,
                verbosity=self.verbosity,
            )[:, 0]

        serial_ccd = self.ccd_from(ccd_phase=serial_ccd)

        try:
            serial_offset = data.readout_offsets[1]
        except AttributeError:
            serial_offset = 0

        image_post_cti = cti.add_cti(
            image=image_post_cti,
            serial_ccd=serial_ccd,
            serial_roe=self.serial_roe,
            serial_traps=serial_trap_list,
            serial_express=self.serial_express,
            serial_offset=serial_offset,
            serial_window_start=self.serial_window_start,
            serial_window_stop=self.serial_window_stop,
            verbosity=self.verbosity,
        )

        try:
            return Array2D.manual_mask(array=image_post_cti, mask=data.mask).native
        except AttributeError:
            return image_post_cti
