from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import mat_plot
from autoarray.plot.plotters import fit_imaging_plotters
from autoarray.plot.plotters import abstract_plotters
from autocti import charge_injection as ci
from autocti.plot.ci_imaging_plotters import extract_line_from


class CIFitPlotter(fit_imaging_plotters.AbstractFitImagingPlotter):
    def __init__(
        self,
        fit: ci.CIFitImaging,
        mat_plot_2d: mat_plot.MatPlot2D = mat_plot.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
        mat_plot_1d: mat_plot.MatPlot1D = mat_plot.MatPlot1D(),
        visuals_1d: vis.Visuals1D = vis.Visuals1D(),
        include_1d: inc.Include1D = inc.Include1D(),
    ):

        super().__init__(
            fit=fit,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

        self.visuals_1d = visuals_1d
        self.include_1d = include_1d
        self.mat_plot_1d = mat_plot_1d

    @property
    def visuals_with_include_2d(self):

        visuals_2d = super().visuals_with_include_2d

        return visuals_2d + self.visuals_2d.__class__(
            parallel_overscan=self.extract_2d(
                "parallel_overscan",
                self.fit.masked_ci_imaging.image.scans.parallel_overscan,
            ),
            serial_prescan=self.extract_2d(
                "serial_prescan", self.fit.masked_ci_imaging.image.scans.serial_prescan
            ),
            serial_overscan=self.extract_2d(
                "serial_overscan",
                self.fit.masked_ci_imaging.image.scans.serial_overscan,
            ),
        )

    @abstract_plotters.for_figure
    def figure_ci_pre_cti(self):
        """Plot the observed ci_pre_cti of the ccd simulator.

        Set *autocti.simulator.plotter.plotter* for a description of all input parameters not described below.

        Parameters
        -----------
        ci_pre_cti : CIFrame
            The ci_pre_cti of the dataset.
        """

        self.mat_plot_2d.plot_frame(frame=self.fit.ci_pre_cti)

    @abstract_plotters.for_figure
    def figure_ci_post_cti(self):
        """Plot the observed ci_post_cti of the ccd simulator.

        Set *autocti.simulator.plotter.plotter* for a description of all input parameters not described below.

        Parameters
        -----------
        ci_post_cti : CIFrame
            The ci_post_cti of the dataset.
        """

        self.mat_plot_2d.plot_frame(frame=self.fit.ci_post_cti)

    @abstract_plotters.for_figure
    def figure_noise_scaling_maps(self):
        """Plot the observed chi_squared_map of the ccd simulator.

        Set *autocti.simulator.plotter.plotter* for a description of all input parameters not described below.

        Parameters
        -----------
        chi_squared_map : CIFrame
            The chi_squared_map of the dataset.
        """

        number_subplots = len(self.fit.noise_scaling_maps)

        self.open_subplot_figure(number_subplots=number_subplots)

        for index in range(len(self.fit.noise_scaling_maps)):

            self.setup_subplot(number_subplots=number_subplots, subplot_index=index + 1)
            self.mat_plot_2d.plot_frame(frame=self.fit.noise_scaling_maps[index])

        self.mat_plot_2d.output.subplot_to_figure()
        self.mat_plot_2d.figure.close()

    @abstract_plotters.for_figure
    def figure_image_line(self, line_region):
        """Plot the observed image of the ccd simulator.

        Set *autocti.simulator.plotter.plotter* for a description of all input parameters not described below.

        Parameters
        -----------
        image : CIFrame
            The image of the dataset.
        """
        line = extract_line_from(ci_frame=self.fit.image, line_region=line_region)

        self.mat_plot_1d.plot_line(y=line, x=range(len(line)))

    @abstract_plotters.for_figure
    def figure_noise_map_line(self, line_region):
        """Plot the observed noise_map of the ccd simulator.

        Set *autocti.simulator.plotter.plotter* for a description of all input parameters not described below.

        Parameters
        -----------
        noise_map : CIFrame
            The noise_map of the dataset.
        """
        line = extract_line_from(ci_frame=self.fit.noise_map, line_region=line_region)

        self.mat_plot_1d.plot_line(y=line, x=range(len(line)))

    @abstract_plotters.for_figure
    def figure_signal_to_noise_map_line(self, line_region):
        """Plot the observed signal_to_noise_map of the ccd simulator.

        Set *autocti.simulator.plotter.plotter* for a description of all input parameters not described below.

        Parameters
        -----------
        signal_to_noise_map : CIFrame
            The signal_to_noise_map of the dataset.
        """
        line = extract_line_from(
            ci_frame=self.fit.signal_to_noise_map, line_region=line_region
        )

        self.mat_plot_1d.plot_line(y=line, x=range(len(line)))

    @abstract_plotters.for_figure
    def figure_ci_pre_cti_line(self, line_region):
        """Plot the observed ci_pre_cti of the ccd simulator.

        Set *autocti.simulator.plotter.plotter* for a description of all input parameters not described below.

        Parameters
        -----------
        ci_pre_cti : CIFrame
            The ci_pre_cti of the dataset.
        """
        line = extract_line_from(ci_frame=self.fit.ci_pre_cti, line_region=line_region)

        self.mat_plot_1d.plot_line(y=line, x=range(len(line)))

    @abstract_plotters.for_figure
    def figure_ci_post_cti_line(self, line_region):
        """Plot the observed ci_post_cti of the ccd simulator.

        Set *autocti.simulator.plotter.plotter* for a description of all input parameters not described below.

        Parameters
        -----------
        ci_post_cti : CIFrame
            The ci_post_cti of the dataset.
        """
        line = extract_line_from(ci_frame=self.fit.ci_post_cti, line_region=line_region)

        self.mat_plot_1d.plot_line(y=line, x=range(len(line)))

    @abstract_plotters.for_figure
    def figure_residual_map_line(self, line_region):
        """Plot the observed residual_map of the ccd simulator.

        Set *autocti.simulator.plotter.plotter* for a description of all input parameters not described below.

        Parameters
        -----------
        residual_map : CIFrame
            The residual_map of the dataset.
        """
        line = extract_line_from(
            ci_frame=self.fit.residual_map, line_region=line_region
        )

        self.mat_plot_1d.plot_line(y=line, x=range(len(line)))

    @abstract_plotters.for_figure
    def figure_normalized_residual_map_line(self, line_region):
        """Plot the observed normalized_residual_map of the ccd simulator.

        Set *autocti.simulator.plotter.plotter* for a description of all input parameters not described below.

        Parameters
        -----------
        normalized_residual_map : CIFrame
            The normalized_residual_map of the dataset.
        """
        line = extract_line_from(
            ci_frame=self.fit.normalized_residual_map, line_region=line_region
        )

        self.mat_plot_1d.plot_line(y=line, x=range(len(line)))

    @abstract_plotters.for_figure
    def figure_chi_squared_map_line(self, line_region):
        """Plot the observed chi_squared_map of the ccd simulator.

        Set *autocti.simulator.plotter.plotter* for a description of all input parameters not described below.

        Parameters
        -----------
        chi_squared_map : CIFrame
            The chi_squared_map of the dataset.
        """
        line = extract_line_from(
            ci_frame=self.fit.chi_squared_map, line_region=line_region
        )

        self.mat_plot_1d.plot_line(y=line, x=range(len(line)))

    def figure_individuals(
        self,
        plot_image=False,
        plot_noise_map=False,
        plot_signal_to_noise_map=False,
        plot_ci_pre_cti=False,
        plot_ci_post_cti=False,
        plot_residual_map=False,
        plot_normalized_residual_map=False,
        plot_chi_squared_map=False,
    ):

        super().figure_individuals(
            plot_image=plot_image,
            plot_noise_map=plot_noise_map,
            plot_signal_to_noise_map=plot_signal_to_noise_map,
            plot_residual_map=plot_residual_map,
            plot_normalized_residual_map=plot_normalized_residual_map,
            plot_chi_squared_map=plot_chi_squared_map,
        )

        if plot_ci_pre_cti:
            self.figure_ci_pre_cti()

        if plot_ci_post_cti:
            self.figure_ci_post_cti()

    def figure_individuals_lines(
        self,
        line_region,
        plot_image=False,
        plot_noise_map=False,
        plot_signal_to_noise_map=False,
        plot_ci_pre_cti=False,
        plot_ci_post_cti=False,
        plot_residual_map=False,
        plot_normalized_residual_map=False,
        plot_chi_squared_map=False,
    ):

        if plot_image:
            self.figure_image_line(line_region=line_region)
        if plot_noise_map:
            self.figure_noise_map_line(line_region=line_region)
        if plot_signal_to_noise_map:
            self.figure_signal_to_noise_map_line(line_region=line_region)
        if plot_ci_pre_cti:
            self.figure_ci_pre_cti_line(line_region=line_region)
        if plot_ci_post_cti:
            self.figure_ci_post_cti_line(line_region=line_region)
        if plot_residual_map:
            self.figure_residual_map_line(line_region=line_region)
        if plot_normalized_residual_map:
            self.figure_normalized_residual_map_line(line_region=line_region)
        if plot_chi_squared_map:
            self.figure_chi_squared_map_line(line_region=line_region)

    @abstract_plotters.for_subplot
    def subplot_ci_fit(self):
        """Plot the model datas_ of an analysis, using the *Fitter* class object.

        The visualization and output type can be fully customized.

        Parameters
        -----------
        """

        number_subplots = 9

        self.open_subplot_figure(number_subplots=number_subplots)

        self.setup_subplot(number_subplots=number_subplots, subplot_index=1)
        self.figure_image()

        self.setup_subplot(number_subplots=number_subplots, subplot_index=2)
        self.figure_noise_map()

        self.setup_subplot(number_subplots=number_subplots, subplot_index=3)
        self.figure_signal_to_noise_map()

        self.setup_subplot(number_subplots=number_subplots, subplot_index=4)
        self.figure_ci_pre_cti()

        self.setup_subplot(number_subplots=number_subplots, subplot_index=5)
        self.figure_ci_post_cti()

        self.setup_subplot(number_subplots=number_subplots, subplot_index=7)
        self.figure_residual_map()

        self.setup_subplot(number_subplots=number_subplots, subplot_index=8)
        self.figure_chi_squared_map()

        self.mat_plot_2d.output.subplot_to_figure()
        self.mat_plot_2d.figure.close()

    # @abstract_plotters.for_figure
    # def subplot_residual_maps(self, fits):
    #     """Plot the model datas_ of an analysis, using the *Fitter* class object.
    #
    #     The visualization and output type can be fully customied.
    #
    #     """
    #
    #     number_subplots = len(fits)
    #
    #     self.open_subplot_figure(number_subplots=number_subplots)
    #
    #     for index, fit in enumerate(fits):
    #
    #         self.setup_subplot(
    #             number_subplots=number_subplots, subplot_index=index + 1
    #         )
    #
    #         residual_map()
    #
    #     self.mat_plot_2d.output.subplot_to_figure()
    #
    #     self.mat_plot_2d.figure.close()
    #
    # @abstract_plotters.for_figure
    # def subplot_normalized_residual_maps(fits):
    #     """Plot the model datas_ of an analysis, using the *Fitter* class object.
    #
    #     The visualization and output type can be fully customied.
    #
    #     """
    #
    #     number_subplots = len(fits)
    #
    #     self.open_subplot_figure(number_subplots=number_subplots)
    #
    #     for index, fit in enumerate(fits):
    #
    #         self.setup_subplot(
    #             number_subplots=number_subplots, subplot_index=index + 1
    #         )
    #
    #         normalized_residual_map()
    #
    #     self.mat_plot_2d.output.subplot_to_figure()
    #
    #     self.mat_plot_2d.figure.close()
    #
    # @abstract_plotters.for_figure
    # def subplot_chi_squared_maps(fits):
    #     """Plot the model datas_ of an analysis, using the *Fitter* class object.
    #
    #     The visualization and output type can be fully customized.
    #
    #     """
    #
    #     number_subplots = len(fits)
    #
    #     self.open_subplot_figure(number_subplots=number_subplots)
    #
    #     for index, fit in enumerate(fits):
    #
    #         self.setup_subplot(
    #             number_subplots=number_subplots, subplot_index=index + 1
    #         )
    #
    #         chi_squared_map()
    #
    #     self.mat_plot_2d.output.subplot_to_figure()
    #
    #     self.mat_plot_2d.figure.close()

    @abstract_plotters.for_subplot
    def subplot_fit_lines(self, line_region):
        """Plot the model datas_ of an analysis, using the *Fitter* class object.

        The visualization and output type can be fully customized.

        """

        number_subplots = 9

        self.open_subplot_figure(number_subplots=number_subplots)

        self.setup_subplot(number_subplots=number_subplots, subplot_index=1)
        self.figure_image_line(line_region=line_region)

        self.setup_subplot(number_subplots=number_subplots, subplot_index=2)
        self.figure_noise_map_line(line_region=line_region)

        self.setup_subplot(number_subplots=number_subplots, subplot_index=3)
        self.figure_signal_to_noise_map_line(line_region=line_region)

        self.setup_subplot(number_subplots=number_subplots, subplot_index=4)
        self.figure_ci_pre_cti_line(line_region=line_region)

        self.setup_subplot(number_subplots=number_subplots, subplot_index=5)
        self.figure_ci_post_cti_line(line_region=line_region)

        self.setup_subplot(number_subplots=number_subplots, subplot_index=7)
        self.figure_residual_map_line(line_region=line_region)

        self.setup_subplot(number_subplots=number_subplots, subplot_index=8)
        self.figure_chi_squared_map_line(line_region=line_region)

        self.mat_plot_2d.output.subplot_to_figure()
        self.mat_plot_2d.figure.close()

    #
    # @abstract_plotters.for_figure
    # def subplot_residual_map_lines(self, fits):
    #     """Plot the model datas_ of an analysis, using the *Fitter* class object.
    #
    #     The visualization and output type can be fully customized.
    #     """
    #
    #     number_subplots = len(fits)
    #
    #     self.open_subplot_figure(number_subplots=number_subplots)
    #
    #     for index, fit in enumerate(fits):
    #         self.setup_subplot(
    #             number_subplots=number_subplots, subplot_index=index + 1
    #         )
    #
    #         residual_map_line(
    #             line_region=line_region
    #         )
    #
    #     self.mat_plot_2d.output.subplot_to_figure()
    #
    #     self.mat_plot_2d.figure.close()
    #
    # @abstract_plotters.for_figure
    # def subplot_normalized_residual_map_lines(
    #     self, fits
    # ):
    #     """Plot the model datas_ of an analysis, using the *Fitter* class object.
    #
    #     The visualization and output type can be fully customized.
    #     """
    #
    #     number_subplots = len(fits)
    #
    #     self.open_subplot_figure(number_subplots=number_subplots)
    #
    #     for index, fit in enumerate(fits):
    #         self.setup_subplot(
    #             number_subplots=number_subplots, subplot_index=index + 1
    #         )
    #
    #         normalized_residual_map_line(
    #             line_region=line_region
    #         )
    #
    #     self.mat_plot_2d.output.subplot_to_figure()
    #
    #     self.mat_plot_2d.figure.close()
    #
    # @abstract_plotters.for_figure
    # def subplot_chi_squared_map_lines(self, fits):
    #     """Plot the model datas_ of an analysis, using the *Fitter* class object.
    #
    #     The visualization and output type can be fully customized.
    #     """
    #
    #     number_subplots = len(fits)
    #
    #     self.open_subplot_figure(number_subplots=number_subplots)
    #
    #     for index, fit in enumerate(fits):
    #
    #         self.setup_subplot(
    #             number_subplots=number_subplots, subplot_index=index + 1
    #         )
    #
    #         chi_squared_map_line(
    #             line_region=line_region
    #         )
    #
    #     self.mat_plot_2d.output.subplot_to_figure()
    #
    #     self.mat_plot_2d.figure.close()
