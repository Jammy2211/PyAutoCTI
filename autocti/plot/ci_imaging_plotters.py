from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import mat_plot
from autoarray.plot.plotters import imaging_plotters
from autoarray.plot.plotters import abstract_plotters
from autocti import charge_injection as ci
from autocti import exc


class CIImagingPlotter(imaging_plotters.AbstractImagingPlotter):
    def __init__(
        self,
        imaging: ci.CIImaging,
        mat_plot_2d: mat_plot.MatPlot2D = mat_plot.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
        mat_plot_1d: mat_plot.MatPlot1D = mat_plot.MatPlot1D(),
        visuals_1d: vis.Visuals1D = vis.Visuals1D(),
        include_1d: inc.Include1D = inc.Include1D(),
    ):

        super().__init__(
            imaging=imaging,
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
                "parallel_overscan", self.imaging.image.scans.parallel_overscan
            ),
            serial_prescan=self.extract_2d(
                "serial_prescan", self.imaging.image.scans.serial_prescan
            ),
            serial_overscan=self.extract_2d(
                "serial_overscan", self.imaging.image.scans.serial_overscan
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
        self.mat_plot_2d.plot_array(
            array=self.imaging.ci_pre_cti, visuals_2d=self.visuals_with_include_2d
        )

    @abstract_plotters.for_figure
    def figure_cosmic_ray_map(self):
        """Plot the observed ci_pre_cti of the ccd simulator.

        Set *autocti.simulator.plotter.plotter* for a description of all input parameters not described below.

        Parameters
        -----------
        ci_pre_cti : CIFrame
            The ci_pre_cti of the dataset.
        """
        self.mat_plot_2d.plot_array(
            array=self.imaging.cosmic_ray_map, visuals_2d=self.visuals_with_include_2d
        )

    @abstract_plotters.for_figure
    def figure_image_line(self, line_region):
        """Plot the observed image of the ccd simulator.

        Set *autocti.simulator.plotter.plotter* for a description of all input parameters not described below.

        Parameters
        -----------
        ci_imaging : CIFrame
            The image of the dataset.
        """
        line = extract_line_from(ci_frame=self.imaging.image, line_region=line_region)

        self.mat_plot_1d.plot_line(y=line, x=range(len(line)))

    @abstract_plotters.for_figure
    def figure_noise_map_line(self, line_region):
        """Plot the observed noise_map of the ccd simulator.

        Set *autocti.simulator.plotter.plotter* for a description of all input parameters not described below.

        Parameters
        -----------
        ci_imaging : CIFrame
            The noise_map of the dataset.
        """
        line = extract_line_from(
            ci_frame=self.imaging.noise_map, line_region=line_region
        )

        self.mat_plot_1d.plot_line(y=line, x=range(len(line)))

    @abstract_plotters.for_figure
    def figure_ci_pre_cti_line(self, line_region):
        """Plot the observed ci_pre_cti of the ccd simulator.

        Set *autocti.simulator.plotter.plotter* for a description of all input parameters not described below.

        Parameters
        -----------
        ci_imaging : CIFrame
            The ci_pre_cti of the dataset.
        """
        line = extract_line_from(
            ci_frame=self.imaging.ci_pre_cti, line_region=line_region
        )

        self.mat_plot_1d.plot_line(y=line, x=range(len(line)))

    @abstract_plotters.for_figure
    def figure_signal_to_noise_map_line(self, line_region):
        """Plot the observed signal_to_noise_map of the ccd simulator.

        Set *autocti.simulator.plotter.plotter* for a description of all input parameters not described below.

        Parameters
        -----------
        ci_imaging : CIFrame
            The signal_to_noise_map of the dataset.
        """
        line = extract_line_from(
            ci_frame=self.imaging.signal_to_noise_map, line_region=line_region
        )

        self.mat_plot_1d.plot_line(y=line, x=range(len(line)))

    def figure_individuals(
        self,
        plot_image=False,
        plot_noise_map=False,
        plot_inverse_noise_map=False,
        plot_signal_to_noise_map=False,
        plot_absolute_signal_to_noise_map=False,
        plot_potential_chi_squared_map=False,
        plot_ci_pre_cti=False,
        plot_cosmic_ray_map=False,
    ):
        """
        Plot each attribute of the imaging data_type as individual figures one by one (e.g. the dataset, noise_map, PSF, \
         Signal-to_noise-map, etc).

        Set *autocti.data_type.array.plotter.plotter* for a description of all innput parameters not described below.

        Parameters
        -----------
        ci_imaging : data_type.ImagingData
            The imaging data_type, which includes the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
        origin : True
            If true, the origin of the dataset's coordinate system is plotted as a 'x'.
        """

        super().figure_individuals(
            plot_image=plot_image,
            plot_noise_map=plot_noise_map,
            plot_inverse_noise_map=plot_inverse_noise_map,
            plot_signal_to_noise_map=plot_signal_to_noise_map,
            plot_absolute_signal_to_noise_map=plot_absolute_signal_to_noise_map,
            plot_potential_chi_squared_map=plot_potential_chi_squared_map,
        )

        if plot_ci_pre_cti:
            self.figure_ci_pre_cti()

        if plot_cosmic_ray_map:
            self.figure_cosmic_ray_map()

    def figure_individual_ci_lines(
        self,
        line_region,
        plot_image=False,
        plot_noise_map=False,
        plot_ci_pre_cti=False,
        plot_signal_to_noise_map=False,
    ):
        """Plot each attribute of the ci simulator as individual figures one by one (e.g. the dataset, noise_map, PSF, \
         Signal-to_noise-map, etc).

        Set *autocti.simulator.arrays.plotter.plotter* for a description of all innput parameters not described below.

        Parameters
        -----------
        ci_imaging : simulator.CCDData
            The ci simulator, which includes the observed dataset, noise_map, PSF, signal-to-noise_map, etc.
        origin : True
            If true, the origin of the dataset's coordinate system is plotted as a 'x'.
        """

        if plot_image:
            self.figure_image_line(line_region=line_region)

        if plot_noise_map:
            self.figure_noise_map_line(line_region=line_region)

        if plot_ci_pre_cti:
            self.figure_ci_pre_cti_line(line_region=line_region)

        if plot_signal_to_noise_map:
            self.figure_signal_to_noise_map_line(line_region=line_region)

    @abstract_plotters.for_subplot
    def subplot_ci_imaging(self):
        """Plot the imaging data_type as a sub-plotter of all its quantites (e.g. the dataset, noise_map, PSF, Signal-to_noise-map, \
         etc).

        Set *autocti.data_type.array.plotter.plotter* for a description of all innput parameters not described below.

        Parameters
        -----------
        ci_imaging : data_type.ImagingData
            The imaging data_type, which includes the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
        origin : True
            If true, the origin of the dataset's coordinate system is plotted as a 'x'.
        image_plane_pix_grid : np.ndarray or data_type.array.grid_stacks.PixGrid
            If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
            over the immage.
        ignore_config : bool
            If `False`, the config file general.ini is used to determine whether the subpot is plotted. If `True`, the \
            config file is ignored.
        """

        number_subplots = 4

        self.open_subplot_figure(number_subplots=number_subplots)

        self.setup_subplot(number_subplots=number_subplots, subplot_index=1)
        self.figure_image()

        self.setup_subplot(number_subplots=number_subplots, subplot_index=2)
        self.figure_noise_map()

        self.setup_subplot(number_subplots=number_subplots, subplot_index=3)
        self.figure_ci_pre_cti()

        self.setup_subplot(number_subplots=number_subplots, subplot_index=4)
        self.figure_signal_to_noise_map()

        self.mat_plot_2d.output.subplot_to_figure()
        self.mat_plot_2d.figure.close()

    @abstract_plotters.for_subplot
    def subplot_ci_lines(self, line_region):
        """Plot the ci simulator as a sub-plotter of all its quantites (e.g. the dataset, noise_map, PSF, Signal-to_noise-map, \
         etc).

        Set *autocti.simulator.arrays.plotter.plotter* for a description of all innput parameters not described below.

        Parameters
        -----------
        ci_imaging : simulator.CCDData
            The ci simulator, which includes the observed dataset, noise_map, PSF, signal-to-noise_map, etc.
        origin : True
            If true, the origin of the dataset's coordinate system is plotted as a 'x'.
        image_plane_pix_grid : np.ndarray or simulator.arrays.grid_lines.PixGrid
            If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
            over the immage.
        ignore_config : bool
            If `False`, the config file general.ini is used to determine whether the subpot is plotted. If `True`, the \
            config file is ignored.
        """

        number_subplots = 4

        self.open_subplot_figure(number_subplots=number_subplots)

        self.setup_subplot(number_subplots=number_subplots, subplot_index=1)
        self.figure_image_line(line_region=line_region)

        self.setup_subplot(number_subplots=number_subplots, subplot_index=2)
        self.figure_noise_map_line(line_region=line_region)

        self.setup_subplot(number_subplots=number_subplots, subplot_index=3)
        self.figure_ci_pre_cti_line(line_region=line_region)

        self.setup_subplot(number_subplots=number_subplots, subplot_index=4)
        self.figure_signal_to_noise_map_line(line_region=line_region)

        self.mat_plot_1d.output.subplot_to_figure()
        self.mat_plot_1d.figure.close()


def extract_line_from(ci_frame, line_region):

    if line_region == "parallel_front_edge":
        return ci_frame.parallel_front_edge_line_binned_over_columns()
    elif line_region == "parallel_trails":
        return ci_frame.parallel_trails_line_binned_over_columns()
    elif line_region == "serial_front_edge":
        return ci_frame.serial_front_edge_line_binned_over_rows()
    elif line_region == "serial_trails":
        return ci_frame.serial_trails_line_binned_over_rows()
    else:
        raise exc.PlottingException(
            "The line region specified for the plotting of a line was invalid"
        )
