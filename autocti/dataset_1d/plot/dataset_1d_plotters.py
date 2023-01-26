import numpy as np
from typing import Callable

import autoarray.plot as aplt

from autoarray.plot.auto_labels import AutoLabels

from autocti.plot.abstract_plotters import Plotter
from autocti.dataset_1d.dataset_1d.dataset_1d import Dataset1D


class Dataset1DPlotter(Plotter):
    def __init__(
        self,
        dataset: Dataset1D,
        mat_plot_1d: aplt.MatPlot1D = aplt.MatPlot1D(),
        visuals_1d: aplt.Visuals1D = aplt.Visuals1D(),
        include_1d: aplt.Include1D = aplt.Include1D(),
    ):
        """
        Plots the attributes of `Dataset1D` objects using the matplotlib method `line()` and many other matplotlib
        functions which customize the plot's appearance.

        The `mat_plot_1d` attribute wraps matplotlib function calls to make the figure. By default, the settings
        passed to every matplotlib function called are those specified in the `config/visualize/mat_wrap/*.ini` files,
        but a user can manually input values into `MatPlot1d` to customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` object. Attributes may be extracted from
        the `Imaging` and plotted via the visuals object, if the corresponding entry is `True` in the `Include1D`
        object or the `config/visualize/include.ini` file.

        Parameters
        ----------
        dataset
            The dataset 1d the plotter plots.
        mat_plot_1d
            Contains objects which wrap the matplotlib function calls that make 1D plots.
        visuals_1d
            Contains 1D visuals that can be overlaid on 1D plots.
        include_1d
            Specifies which attributes of the `ImagingCI` are extracted and plotted as visuals for 1D plots.
        """
        super().__init__(
            mat_plot_1d=mat_plot_1d, include_1d=include_1d, visuals_1d=visuals_1d
        )

        self.dataset = dataset

    @property
    def dataset_1d(self):
        return self.dataset

    def get_visuals_1d(self) -> aplt.Visuals1D:
        return self.visuals_1d

    @property
    def extract_region_from(self) -> Callable:
        return self.dataset.layout.extract_region_from

    def figures_1d(
        self,
        data: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        pre_cti_data: bool = False,
    ):
        """
        Plots the individual attributes of the plotter's `Dataset1D` object in 1D.

        The API is such that every plottable attribute of the `Imaging` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        image
            Whether or not to make a 1D plot (via `plot`) of the data.
        noise_map
            Whether or not to make a 1D plot (via `plot`) of the noise map.
        signal_to_noise_map
            Whether or not to make a 1D plot (via `plot`) of the signal-to-noise map.
        pre_cti_data
            Whether or not to make a 1D plot (via `plot`) of the pre-cti data.
        """

        if data:

            self.mat_plot_1d.plot_yx(
                y=self.dataset_1d.data,
                x=self.dataset_1d.data.grid_radial,
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(title="Dataset 1D Data", filename="data"),
            )

        if noise_map:

            self.mat_plot_1d.plot_yx(
                y=self.dataset_1d.noise_map,
                x=self.dataset_1d.noise_map.grid_radial,
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title="Dataset 1D Noise Map", filename="noise_map"
                ),
            )

        if signal_to_noise_map:

            self.mat_plot_1d.plot_yx(
                y=self.dataset_1d.signal_to_noise_map,
                x=self.dataset_1d.signal_to_noise_map.grid_radial,
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title="Dataset 1D Signal-To-Noise Map",
                    filename="signal_to_noise_map",
                ),
            )

        if pre_cti_data:

            self.mat_plot_1d.plot_yx(
                y=self.dataset_1d.pre_cti_data,
                x=self.dataset_1d.pre_cti_data.grid_radial,
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title="Dataset 1D Pre CTI Data", filename="pre_cti_data"
                ),
            )

    def figures_1d_of_region(
        self,
        region: str,
        data: bool = False,
        noise_map: bool = False,
        pre_cti_data: bool = False,
        signal_to_noise_map: bool = False,
        data_with_noise_map : bool = False,
        data_with_noise_map_logy: bool = False,
    ):
        """
        Plots the individual attributes of the plotter's `Dataset1D` object in 1D.

        These 1D plots correspond to regions in 1D on the charge injection image, which are binned up to produce a
         1D plot.

         For example, for the input `region=fpr`, this function extracts the FPR over each charge region and bins them
        such that the 1D plot shows the average FPR.

        The API is such that every plottable attribute of the `Dataset1D` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        region
            The region on the 1D dataset where data is extracted and binned {fpr", "eper"}
        image
            Whether or not to make a 1D plot (via `plot`) of the image data extracted and binned over the region.
        noise_map
            Whether or not to make a 1D plot (via `plot`) of the noise-map extracted and binned over the region.
        pre_cti_data
            Whether or not to make a 1D plot (via `plot`) of the pre-cti data extracted and binned over the region.
        signal_to_noise_map
            Whether or not to make a 1D plot (via `plot`) of the signal-to-noise map data extracted and binned over
            the region.
        """

        if data:

            y = self.extract_region_from(array=self.dataset.data, region=region)

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title=f"Data 1D {region}",
                    ylabel="Data (e-)",
                    xlabel="Pixel No.",
                    filename=f"data_{region}",
                ),
            )

        if noise_map:
            y = self.extract_region_from(array=self.dataset.noise_map, region=region)

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
            y = self.extract_region_from(array=self.dataset.pre_cti_data, region=region)

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
                array=self.dataset.signal_to_noise_map, region=region
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
            y_errors = self.extract_region_from(array=self.dataset.noise_map, region=region)

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                plot_axis_type_override="errorbar",
                y_errors=y_errors,
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title=f"Data 1D With Noise {region}",
                    ylabel="Data (e-)",
                    xlabel="Pixel No.",
                    filename=f"data_with_noise_map_{region}",
                ),
            )

        if data_with_noise_map_logy:

            y = self.extract_region_from(array=self.dataset.data, region=region)
            y_errors = self.extract_region_from(array=self.dataset.noise_map, region=region)

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                plot_axis_type_override="errorbar_logy",
                y_errors=y_errors,
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title=f"Data 1D With Noise {region} (log10 y axis)",
                    ylabel="Data (e-)",
                    xlabel="Pixel No.",
                    filename=f"data_with_noise_map_logy_{region}",
                ),
            )

    def subplot(
        self,
        data=False,
        noise_map=False,
        signal_to_noise_map=False,
        pre_cti_data=False,
        auto_filename="subplot_dataset_1d",
    ):
        """
        Plots the individual attributes of the plotter's `Dataset1D` object in 1D on a subplot.

        The API is such that every plottable attribute of the `Imaging` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is included on the subplot.

        Parameters
        ----------
        image
            Whether or not to include a 1D plot (via `plot`) of the data.
        noise_map
            Whether or not to include a 1D plot (via `plot`) of the noise map.
        signal_to_noise_map
            Whether or not to include a 1D plot (via `plot`) of the signal-to-noise map.
        pre_cti_data
            Whether or not to include a 1D plot (via `plot`) of the pre-cti data.
        """
        self._subplot_custom_plot(
            data=data,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            pre_cti_data=pre_cti_data,
            auto_labels=AutoLabels(filename=auto_filename),
        )

    def subplot_dataset_1d(self):
        """
        Standard subplot of the attributes of the plotter's `Dataset1D` object.
        """
        self.subplot(
            data=True, noise_map=True, signal_to_noise_map=True, pre_cti_data=True
        )

    def subplot_1d_of_region(self, region: str):
        """
        Plots the individual attributes of the plotter's `Dataset1D` object in 1D on a subplot.

        These 1D plots correspond to a region in 1D on the dataset, which is binned up to produce a 1D plot.

        For example, for the input `region=fpr`, this function extracts the FPR over each charge region and bins such
        that the 1D plot shows the average FPR.

        The function plots the data, noise map, pre-cti data and signal to noise map in 1D on the subplot.

        Parameters
        ----------
        region
            The region on the charge injection image where data is extracted and binned over the parallel or serial
            direction {"parallel_fpr", "parallel_eper", "serial_fpr", "serial_eper"}
        """

        self.open_subplot_figure(number_subplots=4)

        self.figures_1d_of_region(data=True, region=region)
        self.figures_1d_of_region(noise_map=True, region=region)
        self.figures_1d_of_region(pre_cti_data=True, region=region)
        self.figures_1d_of_region(signal_to_noise_map=True, region=region)

        self.mat_plot_1d.output.subplot_to_figure(auto_filename=f"subplot_1d_{region}")
        self.close_subplot_figure()
