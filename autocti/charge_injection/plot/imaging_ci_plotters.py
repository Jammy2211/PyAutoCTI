from typing import Callable

import autoarray.plot as aplt

from autoarray.plot.mat_wrap.mat_plot import AutoLabels
from autoarray.dataset.plot.imaging_plotters import ImagingPlotterMeta

from autocti.plot.abstract_plotters import Plotter
from autocti.charge_injection.imaging.imaging import ImagingCI


class ImagingCIPlotter(Plotter):
    def __init__(
        self,
        imaging: ImagingCI,
        mat_plot_2d: aplt.MatPlot2D = aplt.MatPlot2D(),
        visuals_2d: aplt.Visuals2D = aplt.Visuals2D(),
        include_2d: aplt.Include2D = aplt.Include2D(),
        mat_plot_1d: aplt.MatPlot1D = aplt.MatPlot1D(),
        visuals_1d: aplt.Visuals1D = aplt.Visuals1D(),
        include_1d: aplt.Include1D = aplt.Include1D(),
    ):
        """
        Plots the attributes of `Imaging` objects using the matplotlib method `imshow()` and many other matplotlib
        functions which customize the plot's appearance.

        The `mat_plot_2d` attribute wraps matplotlib function calls to make the figure. By default, the settings
        passed to every matplotlib function called are those specified in the `config/visualize/mat_wrap/*.ini` files,
        but a user can manually input values into `MatPlot2d` to customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals2D` object. Attributes may be extracted from
        the `Imaging` and plotted via the visuals object, if the corresponding entry is `True` in the `Include2D`
        object or the `config/visualize/include.ini` file.

        Parameters
        ----------
        imaging
            The charge injection line imaging dataset the plotter plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        include_2d
            Specifies which attributes of the `Imaging` are extracted and plotted as visuals for 2D plots.
        mat_plot_1d
            Contains objects which wrap the matplotlib function calls that make 1D plots.
        visuals_1d
            Contains 1D visuals that can be overlaid on 1D plots.
        include_1d
            Specifies which attributes of the `ImagingCI` are extracted and plotted as visuals for 1D plots.
        """
        super().__init__(
            mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
        )

        self.visuals_1d = visuals_1d
        self.include_1d = include_1d
        self.mat_plot_1d = mat_plot_1d

        self.imaging = imaging

        self._imaging_meta_plotter = ImagingPlotterMeta(
            imaging=self.imaging,
            get_visuals_2d=self.get_visuals_2d,
            mat_plot_2d=self.mat_plot_2d,
            include_2d=self.include_2d,
            visuals_2d=self.visuals_2d,
        )

    def get_visuals_1d(self):
        return self.visuals_1d

    def get_visuals_2d(self):
        return self.get_2d.via_mask_from(mask=self.imaging.mask)

    @property
    def extract_line_from(self) -> Callable:
        return self.imaging.layout.extract_line_from

    def figures_2d(
        self,
        image=False,
        noise_map=False,
        inverse_noise_map=False,
        signal_to_noise_map=False,
        absolute_signal_to_noise_map=False,
        potential_chi_squared_map=False,
        pre_cti_data=False,
        cosmic_ray_map=False,
    ):
        """
        Plots the individual attributes of the plotter's `ImagingCI` object in 2D.

        The API is such that every plottable attribute of the `Imaging` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        image
            Whether or not to make a 2D plot (via `imshow`) of the image data.
        noise_map
            Whether or not to make a 2D plot (via `imshow`) of the noise map.
        inverse_noise_map
            Whether or not to make a 2D plot (via `imshow`) of the inverse noise map.
        signal_to_noise_map
            Whether or not to make a 2D plot (via `imshow`) of the signal-to-noise map.
        absolute_signal_to_noise_map
            Whether or not to make a 2D plot (via `imshow`) of the absolute signal to noise map.
        potential_chi_squared_map
            Whether or not to make a 2D plot (via `imshow`) of the potential chi squared map.
        pre_cti_data
            Whether or not to make a 2D plot (via `imshow`) of the pre-cti data.
        cosmic_ray_map
            Whether or not to make a 2D plot (via `imshow`) of the cosmic ray map.
        """

        self._imaging_meta_plotter.figures_2d(
            image=image,
            noise_map=noise_map,
            inverse_noise_map=inverse_noise_map,
            signal_to_noise_map=signal_to_noise_map,
            absolute_signal_to_noise_map=absolute_signal_to_noise_map,
            potential_chi_squared_map=potential_chi_squared_map,
        )

        if pre_cti_data:

            self.mat_plot_2d.plot_array(
                array=self.imaging.pre_cti_data,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=AutoLabels(
                    title="CI Pre CTI Image", filename="pre_cti_data"
                ),
            )

        if cosmic_ray_map:

            self.mat_plot_2d.plot_array(
                array=self.imaging.cosmic_ray_map,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=AutoLabels(
                    title="Cosmic Ray Map", filename="cosmic_ray_map"
                ),
            )

    def figures_1d_ci_line_region(
        self,
        line_region,
        image=False,
        noise_map=False,
        pre_cti_data=False,
        signal_to_noise_map=False,
    ):
        """
        Plots the individual attributes of the plotter's `ImagingCI` object in 1D.
        
        These 1D plots correspond to a region in 2D on the charge injection image, which is binned up over the parallel
        or serial direction to produce a 1D plot. For example, for the input `line_region=parallel_front_edge`, this
        function extracts the FPR over each charge injection region and bins such that the 1D plot shows the FPR
        in the parallel direction.

        The API is such that every plottable attribute of the `Imaging` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        line_region
            The region on the charge injection image where data is extracted and binned over the parallel or serial
            direction {"parallel_front_edge", "parallel_epers", "serial_front_edge", "serial_trails"}
        image
            Whether or not to make a 1D plot (via `plot`) of the image data extracted and binned over the line region.
        noise_map
            Whether or not to make a 1D plot (via `plot`) of the noise-map extracted and binned over the line region.
        pre_cti_data
            Whether or not to make a 1D plot (via `plot`) of the pre-cti data extracted and binned over the line region.        
        signal_to_noise_map
            Whether or not to make a 1D plot (via `plot`) of the signal-to-noise map data extracted and binned over 
            the line region.
        """

        if image:

            line = self.extract_line_from(
                array=self.imaging.image, line_region=line_region
            )

            self.mat_plot_1d.plot_yx(
                y=line,
                x=range(len(line)),
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
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
                auto_labels=AutoLabels(
                    title=f"Noise Map Line {line_region}",
                    ylabel="Image",
                    xlabel="Pixel No.",
                    filename=f"noise_map_{line_region}",
                ),
            )

        if pre_cti_data:

            line = self.extract_line_from(
                array=self.imaging.pre_cti_data, line_region=line_region
            )

            self.mat_plot_1d.plot_yx(
                y=line,
                x=range(len(line)),
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title=f"CI Pre CTI Line {line_region}",
                    ylabel="Image",
                    xlabel="Pixel No.",
                    filename=f"pre_cti_data_{line_region}",
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
                auto_labels=AutoLabels(
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
        pre_cti_data=False,
        cosmic_ray_map=False,
        auto_filename="subplot_imaging_ci",
    ):
        """
        Plots the individual attributes of the plotter's `ImagingCI` object in 2D on a subplot.

        The API is such that every plottable attribute of the `Imaging` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is included on the subplot.

        Parameters
        ----------
        image
            Whether or not to include a 2D plot (via `imshow`) of the image data.
        noise_map
            Whether or not to include a 2D plot (via `imshow`) of the noise map.
        inverse_noise_map
            Whether or not to include a 2D plot (via `imshow`) of the inverse noise map.
        signal_to_noise_map
            Whether or not to include a 2D plot (via `imshow`) of the signal-to-noise map.
        absolute_signal_to_noise_map
            Whether or not to include a 2D plot (via `imshow`) of the absolute signal to noise map.
        potential_chi_squared_map
            Whether or not to include a 2D plot (via `imshow`) of the potential chi squared map.
        pre_cti_data
            Whether or not to include a 2D plot (via `imshow`) of the pre-cti data.
        cosmic_ray_map
            Whether or not to include a 2D plot (via `imshow`) of the cosmic ray map.
        auto_filename
            The default filename of the output subplot if written to hard-disk.
        """
        self._subplot_custom_plot(
            image=image,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            inverse_noise_map=inverse_noise_map,
            absolute_signal_to_noise_map=absolute_signal_to_noise_map,
            potential_chi_squared_map=potential_chi_squared_map,
            pre_cti_data=pre_cti_data,
            cosmic_ray_map=cosmic_ray_map,
            auto_labels=AutoLabels(filename=auto_filename),
        )

    def subplot_imaging_ci(self):
        """
        Standard subplot of the attributes of the plotter's `ImagingCI` object.
        """
        self.subplot(
            image=True,
            noise_map=True,
            signal_to_noise_map=True,
            pre_cti_data=True,
            inverse_noise_map=True,
            cosmic_ray_map=True,
        )

    def subplot_1d_ci_line_region(self, line_region):
        """
        Plots the individual attributes of the plotter's `ImagingCI` object in 1D on a subplot.

        These 1D plots correspond to a region in 2D on the charge injection image, which is binned up over the parallel
        or serial direction to produce a 1D plot. For example, for the input `line_region=parallel_front_edge`, this
        function extracts the FPR over each charge injection region and bins such that the 1D plot shows the FPR
        in the parallel direction.

        The function plots the image, noise map, pre-cti data and signal to noise map in 1D on the subplot.

        Parameters
        ----------
        line_region
            The region on the charge injection image where data is extracted and binned over the parallel or serial
            direction {"parallel_front_edge", "parallel_epers", "serial_front_edge", "serial_trails"}
        """

        self.open_subplot_figure(number_subplots=4)

        self.figures_1d_ci_line_region(image=True, line_region=line_region)
        self.figures_1d_ci_line_region(noise_map=True, line_region=line_region)
        self.figures_1d_ci_line_region(pre_cti_data=True, line_region=line_region)
        self.figures_1d_ci_line_region(
            signal_to_noise_map=True, line_region=line_region
        )

        self.mat_plot_1d.output.subplot_to_figure(
            auto_filename=f"subplot_1d_ci_{line_region}"
        )
        self.close_subplot_figure()
