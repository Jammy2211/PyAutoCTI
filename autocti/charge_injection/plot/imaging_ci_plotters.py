import numpy as np
from typing import Callable

import autoarray.plot as aplt

from autoarray.mask.mask_2d import Mask2D
from autoarray.plot.auto_labels import AutoLabels
from autoarray.dataset.plot.imaging_plotters import ImagingPlotterMeta

from autocti.plot.abstract_plotters import Plotter
from autocti.charge_injection.imaging.imaging import ImagingCI


class ImagingCIPlotter(Plotter):
    def __init__(
        self,
        dataset: ImagingCI,
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
        dataset
            The charge injection imaging dataset the plotter plots.
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

        self.dataset = dataset

        self._imaging_meta_plotter = ImagingPlotterMeta(
            imaging=self.imaging,
            get_visuals_2d=self.get_visuals_2d,
            mat_plot_2d=self.mat_plot_2d,
            include_2d=self.include_2d,
            visuals_2d=self.visuals_2d,
        )

    @property
    def imaging(self):
        return self.dataset

    def get_visuals_1d(self):
        return self.visuals_1d

    def get_visuals_2d(self):
        return self.get_2d.via_mask_from(mask=self.imaging.mask)

    @property
    def extract_region_from(self) -> Callable:
        return self.imaging.layout.extract_region_from

    @property
    def extract_region_noise_map_from(self) -> Callable:
        return self.imaging.layout.extract_region_noise_map_from

    def figures_2d(
        self,
        image: bool = False,
        noise_map: bool = False,
        inverse_noise_map: bool = False,
        signal_to_noise_map: bool = False,
        absolute_signal_to_noise_map: bool = False,
        potential_chi_squared_map: bool = False,
        pre_cti_data: bool = False,
        cosmic_ray_map: bool = False,
    ):
        """
        Plots the individual attributes of the plotter's `ImagingCI` object in 2D.

        The API is such that every plottable attribute of the `Imaging` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        image
            Whether to make a 2D plot (via `imshow`) of the image data.
        noise_map
            Whether to make a 2D plot (via `imshow`) of the noise map.
        inverse_noise_map
            Whether to make a 2D plot (via `imshow`) of the inverse noise map.
        signal_to_noise_map
            Whether to make a 2D plot (via `imshow`) of the signal-to-noise map.
        absolute_signal_to_noise_map
            Whether to make a 2D plot (via `imshow`) of the absolute signal to noise map.
        potential_chi_squared_map
            Whether to make a 2D plot (via `imshow`) of the potential chi squared map.
        pre_cti_data
            Whether to make a 2D plot (via `imshow`) of the pre-cti data.
        cosmic_ray_map
            Whether to make a 2D plot (via `imshow`) of the cosmic ray map.
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

    def figures_1d_of_region(
        self,
        region: str,
        image: bool = False,
        noise_map: bool = False,
        pre_cti_data: bool = False,
        signal_to_noise_map: bool = False,
        data_with_noise_map: bool = False,
        data_with_noise_map_logy: bool = False,
    ):
        """
        Plots the individual attributes of the plotter's `ImagingCI` object in 1D.

        These 1D plots correspond to a region in 2D on the charge injection image, which is binned up over the parallel
        or serial direction to produce a 1D plot. For example, for the input `region=parallel_fpr`, this
        function extracts the FPR over each charge injection region and bins such that the 1D plot shows the FPR
        in the parallel direction.

        The API is such that every plottable attribute of the `Imaging` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        region
            The region on the charge injection image where data is extracted and binned over the parallel or serial
            direction {"parallel_fpr", "parallel_eper", "serial_fpr", "serial_eper"}
        image
            Whether to make a 1D plot (via `plot`) of the image data extracted and binned over the region.
        noise_map
            Whether to make a 1D plot (via `plot`) of the noise-map extracted and binned over the region.
        pre_cti_data
            Whether to make a 1D plot (via `plot`) of the pre-cti data extracted and binned over the region.
        signal_to_noise_map
            Whether to make a 1D plot (via `plot`) of the signal-to-noise map data extracted and binned over
            the region.
        data_with_noise_map
            Whether to make a 1D plot (via `plot`) of the image data extracted and binned over the region, with the
            noise-map values included as error bars.
        data_with_noise_map_logy
            Whether to make a 1D plot (via `plot`) of the image data extracted and binned over the region, with the
            noise-map values included as error bars and the y-axis on a log10 scale.
        """

        if image:

            y = self.extract_region_from(array=self.imaging.image, region=region)

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title=f"Data {region}",
                    ylabel="Data (e-)",
                    xlabel="Pixel No.",
                    filename=f"image_{region}",
                ),
            )

        if noise_map:

            y = self.extract_region_from(array=self.imaging.noise_map, region=region)

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title=f"Noise Map {region}",
                    ylabel="Noise (e-)",
                    xlabel="Pixel No.",
                    filename=f"noise_map_{region}",
                ),
            )

        if pre_cti_data:

            y = self.extract_region_from(array=self.imaging.pre_cti_data, region=region)

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title=f"CI Pre CTI {region}",
                    ylabel="Pre CTI Data (e-)",
                    xlabel="Pixel No.",
                    filename=f"pre_cti_data_{region}",
                ),
            )

        if signal_to_noise_map:

            y = self.extract_region_from(
                array=self.imaging.signal_to_noise_map, region=region
            )

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title=f"Signal To Noise Map {region}",
                    ylabel="Signal To Noise (e-)",
                    xlabel="Pixel No.",
                    filename=f"signal_to_noise_map_{region}",
                ),
            )

        if data_with_noise_map:

            y = self.extract_region_from(array=self.dataset.data, region=region)
            y_errors = self.extract_region_noise_map_from(
                array=self.dataset.noise_map, region=region
            )

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                plot_axis_type_override="errorbar",
                y_errors=y_errors,
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title=f"Data w/ Noise {region} (FPR = {self.dataset.fpr_value} e-)",
                    ylabel="Data (e-)",
                    xlabel="Pixel No.",
                    filename=f"data_with_noise_map_{region}",
                ),
            )

        if data_with_noise_map_logy:

            y = self.extract_region_from(array=self.dataset.data, region=region)
            y_errors = self.extract_region_noise_map_from(
                array=self.dataset.noise_map, region=region
            )

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                plot_axis_type_override="errorbar_logy",
                y_errors=y_errors,
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title=f"Data w/ Noise {region} [log10 y] (FPR = {self.dataset.fpr_value} e-)",
                    ylabel="Data (e-)",
                    xlabel="Pixel No.",
                    filename=f"data_with_noise_map_logy_{region}",
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

    def subplot_1d_of_region(self, region: str):
        """
        Plots the individual attributes of the plotter's `ImagingCI` object in 1D on a subplot.

        These 1D plots correspond to a region in 2D on the charge injection image, which is binned up over the parallel
        or serial direction to produce a 1D plot. For example, for the input `region=parallel_fpr`, this
        function extracts the FPR over each charge injection region and bins such that the 1D plot shows the FPR
        in the parallel direction.

        The function plots the image, noise map, pre-cti data and signal to noise map in 1D on the subplot.

        Parameters
        ----------
        region
            The region on the charge injection image where data is extracted and binned over the parallel or serial
            direction {"parallel_fpr", "parallel_eper", "serial_fpr", "serial_eper"}
        """

        self.open_subplot_figure(number_subplots=4)

        self.figures_1d_of_region(image=True, region=region)
        self.figures_1d_of_region(noise_map=True, region=region)
        self.figures_1d_of_region(pre_cti_data=True, region=region)
        self.figures_1d_of_region(signal_to_noise_map=True, region=region)

        self.mat_plot_1d.output.subplot_to_figure(
            auto_filename=f"subplot_1d_ci_{region}"
        )
        self.close_subplot_figure()
