import copy
import numpy as np
from typing import List, Optional, Tuple

import autoarray as aa
from autocti import exc

from arcticpy.src import cti
from arcticpy.src.ccd import CCD
from arcticpy.src.ccd import CCDPhase
from arcticpy.src.roe import ROE
from arcticpy.src.traps import AbstractTrap


class AbstractClocker:
    def __init__(self, iterations: int = 1, verbosity: int = 0):
        """
        An abstract clocker, which wraps the c++ arctic CTI clocking algorithm in **PyAutoCTI**.

        Parameters
        ----------
        iterations
            The number of iterations used to correct CTI from an image.
        verbosity
            Whether to silence print statements and output from the c++ arctic call.
        """

        self.iterations = iterations
        self.verbosity = verbosity

    def ccd_from(self, ccd_phase: CCDPhase) -> CCD:
        """
        Returns a `CCD` object from a `CCDPhase` object.

        The `CCDPhase` describes the volume-filling behaviour of the CCD and is therefore used as a model-component
        in CTI calibration. To call arctic it needs converting to a `CCD` object.

        Parameters
        ----------
        ccd_phase
            The ccd phase describing the volume-filling behaviour of the CCD.

        Returns
        -------
        A `CCD` object based on the input phase which is passed to the c++ arctic.
        """
        if ccd_phase is not None:
            return CCD(phases=[ccd_phase], fraction_of_traps_per_phase=[1.0])

    def check_traps(
        self,
        trap_list_0: Optional[List[AbstractTrap]],
        trap_list_1: Optional[List[AbstractTrap]] = None,
    ):
        """
        Checks that there are trap species passed to the clocking algorithm and raises an exception if not.

        Parameters
        ----------
        trap_list
            A list of the trap species on the CCD which capture and release electrons during clock to as to add CTI.
        """
        if not any([trap_list_0, trap_list_1]):
            raise exc.ClockerException(
                "No Trap species were passed to the add_cti method"
            )

    def check_ccd(self, ccd_list: List[CCDPhase]):
        """
        Checks that there are trap species passed to the clocking algorithm and raises an exception if not.

        Parameters
        ----------
        ccd_list
            The ccd phase settings describing the volume-filling behaviour of the CCD which characterises the capture
            and release of electrons and therefore CTI.
        """
        if not any(ccd_list):
            raise exc.ClockerException("No CCD object was passed to the add_cti method")


class Clocker1D(AbstractClocker):
    def __init__(
        self,
        iterations: int = 1,
        roe: ROE = ROE(),
        express: int = 0,
        window_start: int = 0,
        window_stop: int = -1,
        verbosity: int = 0,
    ):
        """
        Performs clocking of a 1D signal via the c++ arctic algorithm.

        This corresponds to a single row or column of a CCD in the parallel or serial direction. Given the notion of
        parallel and serial are not relevent in 1D, these prefixes are dropped from parameters (unlike the `Clocker2D`)
        object.

        Parameters
        ----------
        iterations
            The number of iterations used to correct CTI from an image.
        roe
            Contains parameters describing the read-out electronics of the CCD (e.g. CCD dwell times, charge injection
            clocking, etc.).
        express
            An integer factor describing how pixel-to-pixel transfers are combined into single transfers for
            efficiency (see: https://academic.oup.com/mnras/article/401/1/371/1006825).
        window_start
            The pixel index of the input image where arCTIc clocking begins, for example if `window_start=10` the
            first 10 pixels are omitted and not clocked.
        window_start
            The pixel index of the input image where arCTIc clocking ends, for example if `window_start=20` any
            pixels after the 20th pixel are omitted and not clocked.
        verbosity
            Whether to silence print statements and output from the c++ arctic call.
        """

        super().__init__(iterations=iterations, verbosity=verbosity)

        self.roe = roe
        self.express = express
        self.window_start = window_start
        self.window_stop = window_stop

    def add_cti(
        self,
        data: aa.Array1D,
        ccd: Optional[CCDPhase] = None,
        trap_list: Optional[List[AbstractTrap]] = None,
    ) -> aa.Array1D:
        """
        Add CTI to a 1D dataset by passing it from the c++ arctic clocking algorithm.

        The c++ arctic wrapper takes as input a 2D ndarray, therefore this function converts the input 1D data to
        a 2D ndarray where the second dimension is 1, passes this from arctic and then flattens the 2D ndarray that
        is returned by to a 1D `Array1D` object with the input data's dimensions.

        Parameters
        ----------
        data
            The 1D data that is clocked via arctic and has CTI added to it.
        ccd
            The ccd phase settings describing the volume-filling behaviour of the CCD which characterises the capture
            and release of electrons and therefore CTI.
        trap_list
            A list of the trap species on the CCD which capture and release electrons during clock to as to add CTI.
        """
        self.check_traps(trap_list_0=trap_list)
        self.check_ccd(ccd_list=[ccd])

        image_pre_cti_2d = aa.Array2D.zeros(
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

        return aa.Array1D.manual_native(
            array=image_post_cti.flatten(), pixel_scales=data.pixel_scales
        )


class Clocker2D(AbstractClocker):
    def __init__(
        self,
        iterations: int = 1,
        parallel_roe: ROE = ROE(),
        parallel_express: int = 0,
        parallel_window_start: int = 0,
        parallel_window_stop: int = -1,
        parallel_poisson_traps: bool = False,
        parallel_fast_pixels: Optional[Tuple[int, int]] = None,
        serial_roe: ROE = ROE(),
        serial_express: int = 0,
        serial_window_start: int = 0,
        serial_window_stop: int = -1,
        verbosity: int = 0,
        poisson_seed: int = -1,
    ):
        """
        Performs clocking of a 2D image via the c++ arctic algorithm.

        This corresponds to a full read out of a CCD including clocking in the parallel and / or serial direction.

        Parameters
        ----------
        iterations
            The number of iterations used to correct CTI from an image.
        parallel_roe
            Contains parameters describing the read-out electronics of the CCD (e.g. CCD dwell times, charge injection
            clocking, etc.) for clocking in the parallel direction.
        parallel_express
            An integer factor describing how parallel pixel-to-pixel transfers are combined into single transfers for
            efficiency (see: https://academic.oup.com/mnras/article/401/1/371/1006825).
        parallel_window_start
            The pixel index of the input image where parallel arCTIc clocking begins, for example
            if `window_start=10` the first 10 pixels are omitted and not clocked.
        parallel_window_start
            The pixel index of the input image where parallel arCTIc clocking ends, for example if `window_start=20`
            any pixels after the 20th pixel are omitted and not clocked.
        parallel_poisson_traps
            If `True`, the density of every trap species (which in the `Trap` objects is defined as the mean density
            of that species) are drawn from a Poisson distribution for every column of data.
        serial_roe
            Contains parameters describing the read-out electronics of the CCD (e.g. CCD dwell times, charge injection
            clocking, etc.) for clocking in the serial direction.
        serial_express
            An integer factor describing how serial pixel-to-pixel transfers are combined into single transfers for
            efficiency (see: https://academic.oup.com/mnras/article/401/1/371/1006825).
        serial_window_start
            The pixel index of the input image where serial arCTIc clocking begins, for example
            if `window_start=10` the first 10 pixels are omitted and not clocked.
        serial_window_start
            The pixel index of the input image where serial arCTIc clocking ends, for example if `window_start=20`
            any pixels after the 20th pixel are omitted and not clocked.            
        verbosity
            Whether to silence print statements and output from the c++ arctic call.
        poisson_seed
            A seed for the random number generator which draws the Poisson trap densities from a Poisson distribution.
        """

        super().__init__(iterations=iterations, verbosity=verbosity)

        self.parallel_roe = parallel_roe
        self.parallel_express = parallel_express
        self.parallel_window_start = parallel_window_start
        self.parallel_window_stop = parallel_window_stop
        self.parallel_poisson_traps = parallel_poisson_traps
        self.parallel_fast_pixels = parallel_fast_pixels

        self.serial_roe = serial_roe
        self.serial_express = serial_express
        self.serial_window_start = serial_window_start
        self.serial_window_stop = serial_window_stop

        self.poisson_seed = poisson_seed

    def add_cti(
        self,
        data: aa.Array2D,
        parallel_ccd: Optional[CCDPhase] = None,
        parallel_trap_list: Optional[List[AbstractTrap]] = None,
        serial_ccd: Optional[CCDPhase] = None,
        serial_trap_list: Optional[List[AbstractTrap]] = None,
    ) -> aa.Array2D:
        """
        Add CTI to a 2D dataset by passing it to the c++ arctic clocking algorithm.

        Clocking is performed towards the readout register and electronics, with parallel CTI added first followed
        by serial CTI. If both parallel and serial CTI are added, parallel CTI is added and the post-CTI image
        (therefore including trailing after parallel clocking) is used to perform serial clocking and add serial CTI.

        If the flag `parallel_poisson_traps=True` the parallel clocking algorithm is sent through the alternative
        function `add_cti_poisson_traps`.  This adds CTI via arctic in the same way, however for parallel clocking
        the density of traps in every column is drawn from a Poisson distribution to represent the stochastic
        nature of how many traps are in each column of a real CCD.

        Parameters
        ----------
        data
            The 1D data that is clocked via arctic and has CTI added to it.
        parallel_ccd
            The ccd phase settings describing the volume-filling behaviour of the CCD which characterises the capture
            and release of electrons and therefore CTI for clocking in the parallel direction.
        parallel_trap_list
            A list of the parallel trap species on the CCD which capture and release electrons during parallel clocking 
            which adds CTI.
        serial_ccd
            The ccd phase settings describing the volume-filling behaviour of the CCD which characterises the capture
            and release of electrons and therefore CTI for clocking in the serial direction.
        serial_trap_list
            A list of the serial trap species on the CCD which capture and release electrons during serial clocking 
            which adds CTI.            
        """

        if self.parallel_poisson_traps:
            return self.add_cti_poisson_traps(
                data=data,
                parallel_ccd=parallel_ccd,
                parallel_trap_list=parallel_trap_list,
                serial_ccd=serial_ccd,
                serial_trap_list=serial_trap_list,
            )

        if self.parallel_fast_pixels is not None:
            return self.add_cti_parallel_fast(
                data=data,
                parallel_ccd=parallel_ccd,
                parallel_trap_list=parallel_trap_list,
            )

        self.check_traps(trap_list_0=parallel_trap_list, trap_list_1=serial_trap_list)
        self.check_ccd(ccd_list=[parallel_ccd, serial_ccd])

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
            return aa.Array2D.manual_mask(array=image_post_cti, mask=data.mask).native
        except AttributeError:
            return image_post_cti

    def add_cti_poisson_traps(
        self,
        data: aa.Array2D,
        parallel_ccd: Optional[CCDPhase] = None,
        parallel_trap_list: Optional[List[AbstractTrap]] = None,
        serial_ccd: Optional[CCDPhase] = None,
        serial_trap_list: Optional[List[AbstractTrap]] = None,
    ) -> aa.Array2D:
        """
        Add CTI to a 2D dataset by passing it to the c++ arctic clocking algorithm.

        Clocking is performed towards the readout register and electronics, with parallel CTI added first followed
        by serial CTI. If both parallel and serial CTI are added, parallel CTI is added and the post-CTI image
        (therefore including trailing after parallel clocking) is used to perform serial clocking and add serial CTI.

        This algorithm adds CTI via arctic in the same way as the normal `add_cti` function, however for parallel
        clocking the density of traps in every column is drawn from a Poisson distribution to represent the stochastic
        nature of how many traps are in each column of a real CCD.

        Parameters
        ----------
        data
            The 1D data that is clocked via arctic and has CTI added to it.
        parallel_ccd
            The ccd phase settings describing the volume-filling behaviour of the CCD which characterises the capture
            and release of electrons and therefore CTI for clocking in the parallel direction.
        parallel_trap_list
            A list of the parallel trap species on the CCD which capture and release electrons during parallel clocking
            which adds CTI.
        serial_ccd
            The ccd phase settings describing the volume-filling behaviour of the CCD which characterises the capture
            and release of electrons and therefore CTI for clocking in the serial direction.
        serial_trap_list
            A list of the serial trap species on the CCD which capture and release electrons during serial clocking
            which adds CTI.
        """
        self.check_traps(trap_list_0=parallel_trap_list, trap_list_1=serial_trap_list)
        self.check_ccd(ccd_list=[parallel_ccd, serial_ccd])

        parallel_ccd = self.ccd_from(ccd_phase=parallel_ccd)

        try:
            parallel_offset = data.readout_offsets[0]
        except AttributeError:
            parallel_offset = 0

        image_pre_cti = data.native
        image_post_cti = np.zeros(data.shape_native)

        total_rows = image_post_cti.shape[0]
        total_columns = image_post_cti.shape[1]

        parallel_trap_column_list = []

        for column in range(total_columns):

            parallel_trap_poisson_list = [
                parallel_trap.poisson_density_from(
                    total_pixels=total_rows, seed=self.poisson_seed
                )
                for parallel_trap in parallel_trap_list
            ]

            parallel_trap_column_list.append(parallel_trap_poisson_list)

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

        self.parallel_trap_column_list = parallel_trap_column_list

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
            return aa.Array2D.manual_mask(array=image_post_cti, mask=data.mask).native
        except AttributeError:
            return image_post_cti

    def _check_parallel_fast(self, image_pre_cti: aa.Array2D):
        """
        Checks the input 2D image to make sure it fits criteria that makes the parallel fast speed up valid
        (see `add_cti_parallel_fast()`.
        """
        start_column = self.parallel_fast_pixels[0]
        end_column = self.parallel_fast_pixels[1]

        if (
            np.any(image_pre_cti[:, 0:start_column]) != 0
            or np.any(image_pre_cti[:, end_column:]) != 0
        ):
            raise exc.ClockerException(
                "Cannot use fast Clocker2D in parallel when there are non-zero entries outside the columns defined by"
                "the `parallel_fast_pixels` tuple."
            )

        if np.any(image_pre_cti[:, start_column] != image_pre_cti[:, end_column - 1]):

            raise exc.ClockerException(
                "Cannot use fast Clocker2D in parallel when there are non-zero entries outside the columns defined by"
                "the `parallel_fast_pixels` tuple."
            )

    def add_cti_parallel_fast(
        self,
        data: aa.Array2D,
        parallel_ccd: Optional[CCDPhase] = None,
        parallel_trap_list: Optional[List[AbstractTrap]] = None,
        perform_checks: bool = True,
    ):
        """
        Add CTI to a 2D dataset by passing it to the c++ arctic clocking algorithm.

        Clocking is performed towards the readout register and electronics, with parallel CTI added first followed
        by serial CTI. If both parallel and serial CTI are added, parallel CTI is added and the post-CTI image
        (therefore including trailing after parallel clocking) is used to perform serial clocking and add serial CTI.

        For 2D images where the same signal is repeated over all columns (e.g. uniform charge injection imaging)
        the CTI added to each column via arctic is identical. Therefore, this function speeds up CTI addition by
        only a single column to arcitc once and copying the output column the NumPy array to construct the the final
        post-cti image.

        By default, checks are performed which ensure that the input data fits the criteria for this speed up.

        Parameters
        ----------
        data
            The 1D data that is clocked via arctic and has CTI added to it.
        parallel_ccd
            The ccd phase settings describing the volume-filling behaviour of the CCD which characterises the capture
            and release of electrons and therefore CTI for clocking in the parallel direction.
        parallel_trap_list
            A list of the parallel trap species on the CCD which capture and release electrons during parallel clocking
            which adds CTI.
        """

        image_pre_cti = data.native
        if perform_checks:
            self._check_parallel_fast(image_pre_cti=image_pre_cti)

        self.check_traps(trap_list_0=parallel_trap_list)
        self.check_ccd(ccd_list=[parallel_ccd])

        parallel_ccd = self.ccd_from(ccd_phase=parallel_ccd)

        image_pre_cti_pass = np.zeros(shape=(data.shape[0], 1))
        image_pre_cti_pass[:, 0] = image_pre_cti[:, self.parallel_fast_pixels[0]]

        image_post_cti_pass = cti.add_cti(
            image=image_pre_cti_pass,
            parallel_ccd=parallel_ccd,
            parallel_roe=self.parallel_roe,
            parallel_traps=parallel_trap_list,
            parallel_express=self.parallel_express,
            parallel_offset=0,
            verbosity=self.verbosity,
        )

        image_post_cti = copy.copy(data.native)

        image_post_cti[
            :, self.parallel_fast_pixels[0] : self.parallel_fast_pixels[1]
        ] = image_post_cti_pass

        return image_post_cti

    def remove_cti(
        self,
        data: aa.Array2D,
        parallel_ccd: Optional[CCDPhase] = None,
        parallel_trap_list: Optional[List[AbstractTrap]] = None,
        serial_ccd: Optional[CCDPhase] = None,
        serial_trap_list: Optional[List[AbstractTrap]] = None,
    ) -> aa.Array2D:
        """
        Remove CTI from a 2D dataset by passing it to the c++ arctic clocking algorithm. The removal of CTI is
        performed by adding CTI to the data to understand how electrons are moved on the CCD, and using this
        image to then move them back to their original pixel.

        Clocking is performed towards the readout register and electronics, with parallel CTI added first followed
        by serial CTI. If both parallel and serial CTI are added, parallel CTI is added and the post-CTI image
        (therefore including trailing after parallel clocking) is used to perform serial clocking and add serial CTI.

        Parameters
        ----------
        data
            The 1D data that is clocked via arctic and has CTI added to it.
        parallel_ccd
            The ccd phase settings describing the volume-filling behaviour of the CCD which characterises the capture
            and release of electrons and therefore CTI for clocking in the parallel direction.
        parallel_trap_list
            A list of the parallel trap species on the CCD which capture and release electrons during parallel clocking
            which adds CTI.
        serial_ccd
            The ccd phase settings describing the volume-filling behaviour of the CCD which characterises the capture
            and release of electrons and therefore CTI for clocking in the serial direction.
        serial_trap_list
            A list of the serial trap species on the CCD which capture and release electrons during serial clocking
            which adds CTI.
        """
        self.check_traps(trap_list_0=parallel_trap_list, trap_list_1=serial_trap_list)
        self.check_ccd(ccd_list=[parallel_ccd, serial_ccd])

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

        return aa.Array2D.manual_mask(
            array=image_cti_removed, mask=data.mask, header=data.header
        ).native
