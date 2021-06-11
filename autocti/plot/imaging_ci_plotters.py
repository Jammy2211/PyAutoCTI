from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import mat_plot as mp
from autoarray.plot import imaging_plotters
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autocti import charge_injection as ci
from autocti import exc

from typing import Union


class ImagingCIPlotter(imaging_plotters.AbstractImagingPlotter):
    def __init__(
        self,
        imaging: ci.ImagingCI,
        mat_plot_2d: mp.MatPlot2D = mp.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
        mat_plot_1d: mp.MatPlot1D = mp.MatPlot1D(),
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
    def visuals_with_include_1d(self) -> vis.Visuals1D:
        """
        Extracts from the `ImagingCI` attributes that can be plotted in 1D and return them in a `Visuals1D` object.

        Only attributes with `True` entries in the `Include1D` object are extracted for plotting.

        From a `ImagingCI` the following 1D attributes can be extracted for plotting:

        - N/A

        Returns
        -------
        vis.Visuals1D
            The collection of attributes that can be plotted by a `Plotter1D` object.
        """
        return self.visuals_1d + self.visuals_1d.__class__()

    @property
    def visuals_with_include_2d(self):

        visuals_2d = super().visuals_with_include_2d

        return visuals_2d + self.visuals_2d.__class__(
            parallel_overscan=self.extract_2d(
                "parallel_overscan", self.imaging.layout.parallel_overscan
            ),
            serial_prescan=self.extract_2d(
                "serial_prescan", self.imaging.layout.serial_prescan
            ),
            serial_overscan=self.extract_2d(
                "serial_overscan", self.imaging.layout.serial_overscan
            ),
        )

    @property
    def extract_line_from(self):
        return self.imaging.layout.extract_line_from

    def figures_2d(
        self,
        image=False,
        noise_map=False,
        inverse_noise_map=False,
        signal_to_noise_map=False,
        absolute_signal_to_noise_map=False,
        potential_chi_squared_map=False,
        pre_cti_image=False,
        cosmic_ray_map=False,
    ):
        """
        Plot each attribute of the imaging data_type as individual figures one by one (e.g. the dataset, noise_map, PSF, \
         Signal-to_noise-map, etc).

        Set *autocti.data_type.array.plotter.plotter* for a description of all innput parameters not described below.

        Parameters
        -----------
        imaging_ci : data_type.ImagingData
            The imaging data_type, which includes the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
        origin : True
            If true, the origin of the dataset's coordinate system is plotted as a 'x'.
        """

        super().figures_2d(
            image=image,
            noise_map=noise_map,
            inverse_noise_map=inverse_noise_map,
            signal_to_noise_map=signal_to_noise_map,
            absolute_signal_to_noise_map=absolute_signal_to_noise_map,
            potential_chi_squared_map=potential_chi_squared_map,
        )

        if pre_cti_image:

            self.mat_plot_2d.plot_array(
                array=self.imaging.pre_cti_image,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(
                    title="CI Pre CTI Image", filename="pre_cti_image"
                ),
            )

        if cosmic_ray_map:

            self.mat_plot_2d.plot_array(
                array=self.imaging.cosmic_ray_map,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=mp.AutoLabels(
                    title="Cosmic Ray Map", filename="cosmic_ray_map"
                ),
            )

    def figures_1d_ci_line_region(
        self,
        line_region,
        image=False,
        noise_map=False,
        pre_cti_image=False,
        signal_to_noise_map=False,
    ):
        """Plot each attribute of the ci simulator as individual figures one by one (e.g. the dataset, noise_map, PSF, \
         Signal-to_noise-map, etc).

        Set *autocti.simulator.arrays.plotter.plotter* for a description of all innput parameters not described below.

        Parameters
        -----------
        imaging_ci : simulator.CCDData
            The ci simulator, which includes the observed dataset, noise_map, PSF, signal-to-noise_map, etc.
        origin : True
            If true, the origin of the dataset's coordinate system is plotted as a 'x'.
        """

        if image:

            line = self.extract_line_from(
                array=self.imaging.image, line_region=line_region
            )

            self.mat_plot_1d.plot_yx(
                y=line,
                x=range(len(line)),
                visuals_1d=self.visuals_with_include_1d,
                auto_labels=mp.AutoLabels(
                    title=f"Image Line {line_region}",
                    ylabel="Image",
                    xlabel="Pixel No.",
                    filename=f"image_{line_region}",
                ),
            )

        if noise_map:

            line = self.extract_line_from(
                array=self.imaging.noise_map, line_region=line_region
            )

            self.mat_plot_1d.plot_yx(
                y=line,
                x=range(len(line)),
                visuals_1d=self.visuals_1d,
                auto_labels=mp.AutoLabels(
                    title=f"Noise Map Line {line_region}",
                    ylabel="Image",
                    xlabel="Pixel No.",
                    filename=f"noise_map_{line_region}",
                ),
            )

        if pre_cti_image:

            line = self.extract_line_from(
                array=self.imaging.pre_cti_image, line_region=line_region
            )

            self.mat_plot_1d.plot_yx(
                y=line,
                x=range(len(line)),
                visuals_1d=self.visuals_1d,
                auto_labels=mp.AutoLabels(
                    title=f"CI Pre CTI Line {line_region}",
                    ylabel="Image",
                    xlabel="Pixel No.",
                    filename=f"pre_cti_image_{line_region}",
                ),
            )

        if signal_to_noise_map:

            line = self.extract_line_from(
                array=self.imaging.signal_to_noise_map, line_region=line_region
            )

            self.mat_plot_1d.plot_yx(
                y=line,
                x=range(len(line)),
                visuals_1d=self.visuals_1d,
                auto_labels=mp.AutoLabels(
                    title=f"Signal To Noise Map {line_region}",
                    ylabel="Image",
                    xlabel="Pixel No.",
                    filename=f"signal_to_noise_map_{line_region}",
                ),
            )

    def subplot(
        self,
        image=False,
        noise_map=False,
        inverse_noise_map=False,
        signal_to_noise_map=False,
        absolute_signal_to_noise_map=False,
        potential_chi_squared_map=False,
        pre_cti_image=False,
        cosmic_ray_map=False,
        auto_filename="subplot_imaging_ci",
    ):

        self._subplot_custom_plot(
            image=image,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            inverse_noise_map=inverse_noise_map,
            pre_cti_image=pre_cti_image,
            cosmic_ray_map=cosmic_ray_map,
            auto_labels=mp.AutoLabels(filename=auto_filename),
        )

    def subplot_imaging_ci(self):
        """Plot the imaging data_type as a sub-plotter of all its quantites (e.g. the dataset, noise_map, PSF, Signal-to_noise-map, \
         etc).

        Set *autocti.data_type.array.plotter.plotter* for a description of all innput parameters not described below.

        Parameters
        -----------
        imaging_ci : data_type.ImagingData
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

        self.subplot(
            image=True,
            noise_map=True,
            signal_to_noise_map=True,
            pre_cti_image=True,
            inverse_noise_map=True,
            cosmic_ray_map=True,
        )

    def subplot_1d_ci_line_region(self, line_region):
        """Plot the ci simulator as a sub-plotter of all its quantites (e.g. the dataset, noise_map, PSF, Signal-to_noise-map, \
         etc).

        Set *autocti.simulator.arrays.plotter.plotter* for a description of all innput parameters not described below.

        Parameters
        -----------
        imaging_ci : simulator.CCDData
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

        self.open_subplot_figure(number_subplots=4)

        self.figures_1d_ci_line_region(image=True, line_region=line_region)
        self.figures_1d_ci_line_region(noise_map=True, line_region=line_region)
        self.figures_1d_ci_line_region(pre_cti_image=True, line_region=line_region)
        self.figures_1d_ci_line_region(
            signal_to_noise_map=True, line_region=line_region
        )

        self.mat_plot_1d.output.subplot_to_figure(
            auto_filename=f"subplot_1d_ci_{line_region}"
        )
        self.close_subplot_figure()