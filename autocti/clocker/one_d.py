from arcticpy.src import cti as arctic
from arcticpy.src.roe import ROE

import autoarray as aa

from autocti.clocker.abstract import AbstractClocker
from autocti.model.model_util import CTI1D


class Clocker1D(AbstractClocker):
    def __init__(
        self,
        iterations: int = 5,
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

    def _traps_ccd_from(self, cti: CTI1D):
        """
        Unpack the `CTI1D` object to retries its traps and ccd.
        """
        if cti.trap_list is not None:
            trap_list = list(cti.trap_list)
        else:
            trap_list = None

        return trap_list, cti.ccd

    def add_cti(self, data: aa.Array1D, cti: CTI1D) -> aa.Array1D:
        """
        Add CTI to a 1D dataset by passing it from the c++ arctic clocking algorithm.

        The c++ arctic wrapper takes as input a 2D ndarray, therefore this function converts the input 1D data to
        a 2D ndarray where the second dimension is 1, passes this from arctic and then flattens the 2D ndarray that
        is returned by to a 1D `Array1D` object with the input data's dimensions.

        Parameters
        ----------
        data
            The 1D data that is clocked via arctic and has CTI added to it.
        cti
            An object which represents the CTI properties of 1D clocking, including the trap species which capture
            and release electrons and the volume-filling behaviour of the CCD.
        """

        trap_list, ccd = self._traps_ccd_from(cti=cti)

        self.check_traps(trap_list_0=trap_list)
        self.check_ccd(ccd_list=[ccd])

        image_pre_cti_2d = aa.Array2D.zeros(
            shape_native=(data.shape_native[0], 1), pixel_scales=data.pixel_scales
        ).native

        image_pre_cti_2d[:, 0] = data

        ccd = self.ccd_from(ccd_phase=ccd)

        image_post_cti = arctic.add_cti(
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

    def remove_cti(self, data: aa.Array1D, cti: CTI1D) -> aa.Array1D:
        """
        Add CTI to a 1D dataset by passing it from the c++ arctic clocking algorithm.

        The c++ arctic wrapper takes as input a 2D ndarray, therefore this function converts the input 1D data to
        a 2D ndarray where the second dimension is 1, passes this from arctic and then flattens the 2D ndarray that
        is returned by to a 1D `Array1D` object with the input data's dimensions.

        Parameters
        ----------
        data
            The 1D data that is clocked via arctic and has CTI removeed to it.
        cti
            An object which represents the CTI properties of the clocking, including the trap species which capture
            and release electrons and the volume-filling behaviour of the CCD.
        """

        trap_list, ccd = self._traps_ccd_from(cti=cti)

        self.check_traps(trap_list_0=trap_list)
        self.check_ccd(ccd_list=[ccd])

        image_pre_cti_2d = aa.Array2D.zeros(
            shape_native=(data.shape_native[0], 1), pixel_scales=data.pixel_scales
        ).native

        image_pre_cti_2d[:, 0] = data

        ccd = self.ccd_from(ccd_phase=ccd)

        image_post_cti = arctic.remove_cti(
            image=image_pre_cti_2d,
            n_iterations=self.iterations,
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
