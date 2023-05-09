import numpy as np
from typing import Optional

from arcticpy import add_cti
from arcticpy import remove_cti

from arcticpy import ROE

import autoarray as aa

from autocti.clocker.abstract import AbstractClocker
from autocti.model.model_util import CTI2D
from autocti.preloads import Preloads


class Clocker2D(AbstractClocker):
    def __init__(
        self,
        iterations: int = 5,
        parallel_roe: Optional[ROE] = None,
        parallel_express: int = 0,
        parallel_window_offset: int = 0,
        parallel_window_start: int = 0,
        parallel_window_stop: int = -1,
        parallel_time_start=0,
        parallel_time_stop=-1,
        parallel_prune_n_electrons=1e-18,
        parallel_prune_frequency=0,
        parallel_poisson_traps: bool = False,
        parallel_fast_mode: Optional[bool] = False,
        serial_roe: Optional[ROE] = None,
        serial_express: int = 0,
        serial_window_offset: int = 0,
        serial_window_start: int = 0,
        serial_window_stop: int = -1,
        serial_time_start=0,
        serial_time_stop=-1,
        serial_prune_n_electrons=1e-18,
        serial_prune_frequency=0,
        serial_fast_mode: Optional[bool] = None,
        allow_negative_pixels=1,
        pixel_bounce=None,
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
        parallel_window_offset
            The number of pixels before parallel clocking begins, thereby extending the length over which clocking
            is performed in the parallel direction and increasing CTI.
        parallel_window_start
            The pixel index of the input image where parallel arCTIc clocking begins, for example
            if `window_start=10` the first 10 pixels are omitted and not clocked.
        parallel_window_start
            The pixel index of the input image where parallel arCTIc clocking ends, for example if `window_start=20`
            any pixels after the 20th pixel are omitted and not clocked.
        parallel_poisson_traps
            If `True`, the density of every trap species (which in the `Trap` objects is defined as the mean density
            of that species) are drawn from a Poisson distribution for every column of data.
        parallel_fast_mode
            If input, parallel CTI is added via arctic efficiently by determining all unique columns in the input data,
            calling arctic once per unique column and mapping the 1D output over the full 2D image. This requires
            many column in the image to have the same signal to be efficient, for example charge injection
            calibration data.
        serial_roe
            Contains parameters describing the read-out electronics of the CCD (e.g. CCD dwell times, charge injection
            clocking, etc.) for clocking in the serial direction.
        serial_express
            An integer factor describing how serial pixel-to-pixel transfers are combined into single transfers for
            efficiency (see: https://academic.oup.com/mnras/article/401/1/371/1006825).
        serial_window_offset
            The number of pixels before serial clocking begins, thereby extending the length over which clocking
            is performed in the serial direction and increasing CTI.
        serial_window_start
            The pixel index of the input image where serial arCTIc clocking begins, for example
            if `window_start=10` the first 10 pixels are omitted and not clocked.
        serial_window_start
            The pixel index of the input image where serial arCTIc clocking ends, for example if `window_start=20`
            any pixels after the 20th pixel are omitted and not clocked.
        serial_fast_mode
            If input, serial CTI is added via arctic efficiently by calling arctic once and mapping the 1D output over
            the full 2D image. This requires every row in the image has the same signal (such that each column gives
            an identical arctic output).
        allow_negative_pixels
            If True, negative electrons in a pixel are allowed and modeled via arCTIc, if Falss they are explicitly
            not allowed.
        verbosity
            Whether to silence print statements and output from the c++ arctic call.
        poisson_seed
            A seed for the random number generator which draws the Poisson trap densities from a Poisson distribution.
        """

        super().__init__(iterations=iterations, verbosity=verbosity)

        self.parallel_roe = parallel_roe or ROE()
        self.parallel_express = parallel_express
        self.parallel_window_offset = parallel_window_offset
        self.parallel_window_start = parallel_window_start
        self.parallel_window_stop = parallel_window_stop
        self.parallel_time_start = parallel_time_start
        self.parallel_time_stop = parallel_time_stop
        self.parallel_prune_n_electrons = parallel_prune_n_electrons
        self.parallel_prune_frequency = parallel_prune_frequency
        self.parallel_poisson_traps = parallel_poisson_traps
        self.parallel_fast_mode = parallel_fast_mode

        self.serial_roe = serial_roe or ROE()
        self.serial_express = serial_express
        self.serial_window_offset = serial_window_offset
        self.serial_window_start = serial_window_start
        self.serial_window_stop = serial_window_stop
        self.serial_time_start = serial_time_start
        self.serial_time_stop = serial_time_stop
        self.serial_prune_n_electrons = serial_prune_n_electrons
        self.serial_prune_frequency = serial_prune_frequency
        self.serial_fast_mode = serial_fast_mode

        self.allow_negative_pixels = allow_negative_pixels
        self.pixel_bounce = pixel_bounce

        self.poisson_seed = poisson_seed

    def _parallel_traps_ccd_from(self, cti: CTI2D):
        """
        Unpack the `CTI1D` object to retries its traps and ccd.
        """
        if cti.parallel_trap_list is not None:
            trap_list = list(cti.parallel_trap_list)
        else:
            trap_list = None

        return trap_list, cti.parallel_ccd

    def _serial_traps_ccd_from(self, cti: CTI2D):
        """
        Unpack the `CTI1D` object to retries its traps and ccd.
        """
        if cti.serial_trap_list is not None:
            trap_list = list(cti.serial_trap_list)
        else:
            trap_list = None

        return trap_list, cti.serial_ccd

    def add_cti(
        self, data: aa.Array2D, cti: CTI2D, preloads: Optional[Preloads] = Preloads()
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
        cti
            An object which represents the CTI properties of 2D clocking, including the trap species which capture
            and release electrons and the volume-filling behaviour of the CCD for parallel and serial clocking.
        """

        if self.parallel_poisson_traps:
            return self.add_cti_poisson_traps(data=data, cti=cti)

        if self.parallel_fast_mode:
            return self.add_cti_parallel_fast(data=data, cti=cti, preloads=preloads)

        if self.serial_fast_mode:
            return self.add_cti_serial_fast(data=data, cti=cti, preloads=preloads)

        data = data.native_skip_mask

        parallel_trap_list, parallel_ccd = self._parallel_traps_ccd_from(cti=cti)
        serial_trap_list, serial_ccd = self._serial_traps_ccd_from(cti=cti)

        self.check_traps(trap_list_0=parallel_trap_list, trap_list_1=serial_trap_list)
        self.check_ccd(ccd_list=[parallel_ccd, serial_ccd])

        parallel_ccd = self.ccd_from(ccd_phase=parallel_ccd)
        serial_ccd = self.ccd_from(ccd_phase=serial_ccd)

        try:
            parallel_window_offset = data.readout_offsets[0]
            serial_window_offset = data.readout_offsets[1]
        except AttributeError:
            parallel_window_offset = self.parallel_window_offset
            serial_window_offset = self.serial_window_offset

        image_post_cti = add_cti(
            image=data,
            parallel_ccd=parallel_ccd,
            parallel_roe=self.parallel_roe,
            parallel_traps=parallel_trap_list,
            parallel_express=self.parallel_express,
            parallel_window_offset=parallel_window_offset,
            parallel_window_start=self.parallel_window_start,
            parallel_window_stop=self.parallel_window_stop,
            parallel_time_start=self.parallel_time_start,
            parallel_time_stop=self.parallel_time_stop,
            parallel_prune_n_electrons=self.parallel_prune_n_electrons,
            parallel_prune_frequency=self.parallel_prune_frequency,
            serial_ccd=serial_ccd,
            serial_roe=self.serial_roe,
            serial_traps=serial_trap_list,
            serial_express=self.serial_express,
            serial_window_offset=serial_window_offset,
            serial_window_start=self.serial_window_start,
            serial_window_stop=self.serial_window_stop,
            serial_time_start=self.serial_time_start,
            serial_time_stop=self.serial_time_stop,
            serial_prune_n_electrons=self.serial_prune_n_electrons,
            serial_prune_frequency=self.serial_prune_frequency,
            allow_negative_pixels=self.allow_negative_pixels,
            pixel_bounce=self.pixel_bounce,
            verbosity=self.verbosity,
        )

        try:
            return aa.Array2D(
                values=image_post_cti, mask=data.mask, store_native=True, skip_mask=True
            )
        except AttributeError:
            return image_post_cti

    def add_cti_poisson_traps(self, data: aa.Array2D, cti: CTI2D) -> aa.Array2D:
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
        cti
            An object which represents the CTI properties of 2D clocking, including the trap species which capture
            and release electrons and the volume-filling behaviour of the CCD for parallel and serial clocking.
        """

        parallel_trap_list, parallel_ccd = self._parallel_traps_ccd_from(cti=cti)
        serial_trap_list, serial_ccd = self._serial_traps_ccd_from(cti=cti)

        self.check_traps(trap_list_0=parallel_trap_list, trap_list_1=serial_trap_list)
        self.check_ccd(ccd_list=[parallel_ccd, serial_ccd])

        parallel_ccd = self.ccd_from(ccd_phase=parallel_ccd)

        try:
            parallel_window_offset = data.readout_offsets[0]
        except AttributeError:
            parallel_window_offset = self.parallel_window_offset

        image_pre_cti = data.native_skip_mask
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

            image_post_cti[:, column] = add_cti(
                image=image_pre_cti_pass,
                parallel_ccd=parallel_ccd,
                parallel_roe=self.parallel_roe,
                parallel_traps=parallel_trap_poisson_list,
                parallel_express=self.parallel_express,
                parallel_window_offset=parallel_window_offset,
                parallel_window_start=self.parallel_window_start,
                parallel_window_stop=self.parallel_window_stop,
                allow_negative_pixels=self.allow_negative_pixels,
                pixel_bounce=self.pixel_bounce,
                verbosity=self.verbosity,
            )[:, 0]

        self.parallel_trap_column_list = parallel_trap_column_list

        serial_ccd = self.ccd_from(ccd_phase=serial_ccd)

        try:
            serial_window_offset = data.readout_offsets[1]
        except AttributeError:
            serial_window_offset = self.serial_window_offset

        image_post_cti = add_cti(
            image=image_post_cti,
            serial_ccd=serial_ccd,
            serial_roe=self.serial_roe,
            serial_traps=serial_trap_list,
            serial_express=self.serial_express,
            serial_window_offset=serial_window_offset,
            serial_window_start=self.serial_window_start,
            serial_window_stop=self.serial_window_stop,
            serial_time_start=self.serial_time_start,
            serial_time_stop=self.serial_time_stop,
            serial_prune_n_electrons=self.serial_prune_n_electrons,
            serial_prune_frequency=self.serial_prune_frequency,
            allow_negative_pixels=self.allow_negative_pixels,
            pixel_bounce=self.pixel_bounce,
            verbosity=self.verbosity,
        )

        try:
            return aa.Array2D(
                values=image_post_cti, mask=data.mask, store_native=True, skip_mask=True
            )
        except AttributeError:
            return image_post_cti

    def fast_indexes_from(self, data: aa.Array2D, for_parallel: bool):
        """
        For 2D images where the same signal is repeated over many columns (e.g. uniform charge injection imaging)
        the CTI added to each column for parallel CTI via arctic is identical. The same is true of serial CTI
        addition, where many rows of data are repeated.

        Therefore, this function speeds up CTI addition via arCTIc by extracting all identical columns (for parallel
        clocking) or rows (for serial clocking), adding CTI via arcitc to only these extracted stripes and copying the
        output stripes to construct the final post-cti image.

        This function inspects the input image and extracts the indexes of every unique stripe, alongside
        a list which maps this unique stripe to all other stripes it is identical to. These are used in the functions
        `add_cti_parallel_fast` and `add_cti_serial_fast` to create the extracted image that is passed to arctic and
        rebuild the final post CTI image.

        Parameters
        ----------
        data
            The 1D data that is clocked via arctic and has CTI added to it.
        """

        if for_parallel:
            total_stripes = data.shape[1]
        else:
            total_stripes = data.shape[0]

        fast_index_list = []
        fast_column_lists = []

        unchecked_list = range(0, total_stripes)

        for stripe_index in range(total_stripes):
            paired = False

            pair_list = []
            unchecked_list_new = []

            if stripe_index in unchecked_list:
                for pair_index in unchecked_list:
                    if for_parallel:
                        residual_map = np.abs(
                            data[:, stripe_index] - data[:, pair_index]
                        )
                    else:
                        residual_map = np.abs(
                            data[stripe_index, :] - data[pair_index, :]
                        )

                    if np.all(residual_map < 1.0e-8):
                        if not paired:
                            fast_index_list.append(stripe_index)

                        paired = True

                        pair_list.append(pair_index)

                    else:
                        unchecked_list_new.append(pair_index)

                fast_column_lists.append(pair_list)
                unchecked_list = unchecked_list_new

        return fast_index_list, fast_column_lists

    def add_cti_parallel_fast(
        self, data: aa.Array2D, cti: CTI2D, preloads: Preloads = Preloads()
    ):
        """
        Add CTI to a 2D dataset by passing it to the c++ arctic clocking algorithm.

        Clocking is performed towards the readout register and electronics, with parallel CTI added first followed
        by serial CTI. If both parallel and serial CTI are added, parallel CTI is added and the post-CTI image
        (therefore including trailing after parallel clocking) is used to perform serial clocking and add serial CTI.

        For 2D images where the same signal is repeated over many columns (e.g. uniform charge injection imaging)
        the CTI added to each column via arctic is identical. Therefore, this function speeds up CTI addition by
        extracting all identical columns, adding CTI via arcitc to only these columns and copying the output columns
        to construct the final post-cti image.

        Parameters
        ----------
        data
            The 1D data that is clocked via arctic and has CTI added to it.
        cti
            An object which represents the CTI properties of 2D clocking, including the trap species which capture
            and release electrons and the volume-filling behaviour of the CCD for parallel clocking.
        perform_checks
            Check that it is value for the input data to use the fast clocking speed up (e.g. all columns are identical
            and all entries outside this region are zero).
        """

        parallel_trap_list, parallel_ccd = self._parallel_traps_ccd_from(cti=cti)

        image_pre_cti = data.native_skip_mask

        self.check_traps(trap_list_0=parallel_trap_list)
        self.check_ccd(ccd_list=[parallel_ccd])

        parallel_ccd = self.ccd_from(ccd_phase=parallel_ccd)

        if preloads.parallel_fast_index_list is None:
            fast_index_list, fast_column_lists = self.fast_indexes_from(
                data=image_pre_cti, for_parallel=True
            )
        else:
            fast_index_list = preloads.parallel_fast_index_list
            fast_column_lists = preloads.parallel_fast_column_lists

        image_pre_cti_pass = np.zeros(
            shape=(data.shape_native[0], len(fast_index_list))
        )
        for i, fast_index in enumerate(fast_index_list):
            image_pre_cti_pass[:, i] = image_pre_cti[:, fast_index]

        image_post_cti_pass = add_cti(
            image=image_pre_cti_pass,
            parallel_ccd=parallel_ccd,
            parallel_roe=self.parallel_roe,
            parallel_traps=parallel_trap_list,
            parallel_express=self.parallel_express,
            parallel_window_offset=self.parallel_window_offset,
            parallel_time_start=self.parallel_time_start,
            parallel_time_stop=self.parallel_time_stop,
            parallel_prune_n_electrons=self.parallel_prune_n_electrons,
            parallel_prune_frequency=self.parallel_prune_frequency,
            allow_negative_pixels=self.allow_negative_pixels,
            pixel_bounce=self.pixel_bounce,
            verbosity=self.verbosity,
        )

        image_post_cti = np.zeros(shape=data.shape_native)

        for i, fast_column_list in enumerate(fast_column_lists):
            for fast_column in fast_column_list:
                image_post_cti[:, fast_column] = image_post_cti_pass[:, i]

        if cti.serial_trap_list is None:
            return aa.Array2D(
                values=image_post_cti, mask=data.mask, store_native=True, skip_mask=True
            )

        serial_trap_list, serial_ccd = self._serial_traps_ccd_from(cti=cti)

        self.check_traps(trap_list_0=serial_trap_list)
        self.check_ccd(ccd_list=[serial_ccd])

        serial_ccd = self.ccd_from(ccd_phase=serial_ccd)

        image_post_cti = add_cti(
            image=image_post_cti,
            serial_ccd=serial_ccd,
            serial_roe=self.serial_roe,
            serial_traps=serial_trap_list,
            serial_express=self.serial_express,
            serial_window_offset=self.serial_window_offset,
            serial_time_start=self.serial_time_start,
            serial_time_stop=self.serial_time_stop,
            serial_prune_n_electrons=self.serial_prune_n_electrons,
            serial_prune_frequency=self.serial_prune_frequency,
            allow_negative_pixels=self.allow_negative_pixels,
            pixel_bounce=self.pixel_bounce,
            verbosity=self.verbosity,
        )

        return aa.Array2D(
            values=image_post_cti, mask=data.mask, store_native=True, skip_mask=True
        )

    def add_cti_serial_fast(
        self, data: aa.Array2D, cti: CTI2D, preloads: Preloads = Preloads()
    ):
        """
        Add CTI to a 2D dataset by passing it to the c++ arctic clocking algorithm.

        Clocking is performed towards the readout register and electronics, with parallel CTI added first followed
        by serial CTI. If both parallel and serial CTI are added, serial CTI is added and the post-CTI image
        (therefore including trailing after serial clocking) is used to perform serial clocking and add serial CTI.

        For 2D images where the same signal is repeated over all columns (e.g. uniform charge injection imaging)
        the CTI added to each row via arctic is identical. Therefore, this function speeds up CTI addition by
        only a single column to arcitc once and copying the output column the NumPy array to construct the final
        post-cti image.

        This only works for serial CTI when parallel CTI is omitted.

        By default, checks are performed which ensure that the input data fits the criteria for this speed up.

        Parameters
        ----------
        data
            The 1D data that is clocked via arctic and has CTI added to it.
        cti
            An object which represents the CTI properties of 2D clocking, including the trap species which capture
            and release electrons and the volume-filling behaviour of the CCD for serial clocking.
        perform_checks
            Check that it is value for the input data to use the fast clocking speed up (e.g. all columns are identical
            and all entries outside this region are zero).
        """

        serial_trap_list, serial_ccd = self._serial_traps_ccd_from(cti=cti)

        image_pre_cti = data.native_skip_mask

        self.check_traps(trap_list_0=serial_trap_list)
        self.check_ccd(ccd_list=[serial_ccd])

        serial_ccd = self.ccd_from(ccd_phase=serial_ccd)

        if preloads.serial_fast_index_list is None:
            fast_index_list, fast_row_lists = self.fast_indexes_from(
                data=image_pre_cti, for_parallel=False
            )
        else:
            fast_index_list = preloads.serial_fast_index_list
            fast_row_lists = preloads.serial_fast_row_lists

        image_pre_cti_pass = np.zeros(
            shape=(len(fast_index_list), data.shape_native[1])
        )
        for i, fast_index in enumerate(fast_index_list):
            image_pre_cti_pass[i, :] = image_pre_cti[fast_index, :]

        image_post_cti_pass = add_cti(
            image=image_pre_cti_pass,
            serial_ccd=serial_ccd,
            serial_roe=self.serial_roe,
            serial_traps=serial_trap_list,
            serial_express=self.serial_express,
            serial_window_offset=self.serial_window_offset,
            serial_time_start=self.serial_time_start,
            serial_time_stop=self.serial_time_stop,
            serial_prune_n_electrons=self.serial_prune_n_electrons,
            serial_prune_frequency=self.serial_prune_frequency,
            allow_negative_pixels=self.allow_negative_pixels,
            pixel_bounce=self.pixel_bounce,
            verbosity=self.verbosity,
        )

        image_post_cti = np.zeros(shape=data.shape_native)

        for i, fast_row_list in enumerate(fast_row_lists):
            for fast_row in fast_row_list:
                image_post_cti[fast_row, :] = image_post_cti_pass[i, :]

        return aa.Array2D(
            values=image_post_cti, mask=data.mask, store_native=True, skip_mask=True
        )

    def remove_cti(self, data: aa.Array2D, cti: CTI2D) -> aa.Array2D:
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
        cti
            An object which represents the CTI properties of 2D clocking, including the trap species which capture
            and release electrons and the volume-filling behaviour of the CCD for parallel and serial clocking.
        """

        parallel_trap_list, parallel_ccd = self._parallel_traps_ccd_from(cti=cti)
        serial_trap_list, serial_ccd = self._serial_traps_ccd_from(cti=cti)

        self.check_traps(trap_list_0=parallel_trap_list, trap_list_1=serial_trap_list)
        self.check_ccd(ccd_list=[parallel_ccd, serial_ccd])

        parallel_ccd = self.ccd_from(ccd_phase=parallel_ccd)
        serial_ccd = self.ccd_from(ccd_phase=serial_ccd)

        image_cti_removed = remove_cti(
            image=data,
            n_iterations=self.iterations,
            parallel_ccd=parallel_ccd,
            parallel_roe=self.parallel_roe,
            parallel_traps=parallel_trap_list,
            parallel_express=self.parallel_express,
            parallel_window_offset=data.readout_offsets[0],
            parallel_window_start=self.parallel_window_start,
            parallel_window_stop=self.parallel_window_stop,
            parallel_time_start=self.parallel_time_start,
            parallel_time_stop=self.parallel_time_stop,
            parallel_prune_n_electrons=self.parallel_prune_n_electrons,
            parallel_prune_frequency=self.parallel_prune_frequency,
            serial_ccd=serial_ccd,
            serial_roe=self.serial_roe,
            serial_traps=serial_trap_list,
            serial_express=self.serial_express,
            serial_window_offset=data.readout_offsets[1],
            serial_window_start=self.serial_window_start,
            serial_window_stop=self.serial_window_stop,
            serial_time_start=self.serial_time_start,
            serial_time_stop=self.serial_time_stop,
            serial_prune_n_electrons=self.serial_prune_n_electrons,
            serial_prune_frequency=self.serial_prune_frequency,
            allow_negative_pixels=self.allow_negative_pixels,
            pixel_bounce=self.pixel_bounce,
        )

        return aa.Array2D(
            values=image_cti_removed,
            mask=data.mask,
            header=data.header,
            store_native=True,
            skip_mask=True,
        )
