from typing import Callable, Optional

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
        region: Optional[str] = None,
        data_with_noise_map: bool = False,
        data_with_noise_map_logy: bool = False,
        data: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        pre_cti_data: bool = False,
    ):
        """
        Plots the individual attributes of the plotter's `Dataset1D` object in 1D.

        The API is such that every plottable attribute of the `Imaging` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        If a `region` string is input, the 1D plots correspond to regions in 1D on the 1D dataset, which are binned up
        to produce a1D plot.

        For example, for the input `region=fpr`, this function extracts the FPR over each charge region and bins them
        such that the 1D plot shows the average FPR.

        The API is such that every plottable attribute of the `Dataset1D` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        region
            The region on the 1D dataset where data is extracted and binned {fpr", "eper"}
        data_with_noise_map
            Whether to make a 1D plot (via `plot`) of the image data extracted and binned over the region, with the
            noise-map values included as error bars.
        data_with_noise_map_logy
            Whether to make a 1D plot (via `plot`) of the image data extracted and binned over the region, with the
            noise-map values included as error bars and the y-axis on a log10 scale.
        data
            Whether to make a 1D plot (via `plot`) of the image data extracted and binned over the region.
        noise_map
            Whether to make a 1D plot (via `plot`) of the noise-map extracted and binned over the region.
        pre_cti_data
            Whether to make a 1D plot (via `plot`) of the pre-cti data extracted and binned over the region.
        signal_to_noise_map
            Whether to make a 1D plot (via `plot`) of the signal-to-noise map data extracted and binned over
            the region.
        """

        suffix = f"_{region}" if region is not None else ""

        if data_with_noise_map:

            y = self.extract_region_from(array=self.dataset.data, region=region)
            y_errors = self.extract_region_from(
                array=self.dataset.noise_map, region=region
            )

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
                    filename=f"data_with_noise_map{suffix}",
                ),
            )

        if data_with_noise_map_logy:

            y = self.extract_region_from(array=self.dataset.data, region=region)
            y_errors = self.extract_region_from(
                array=self.dataset.noise_map, region=region
            )

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
                    filename=f"data_with_noise_map_logy{suffix}",
                ),
            )

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
                    filename=f"data{suffix}",
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
                    filename=f"noise_map{suffix}",
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
                    filename=f"pre_cti_data{suffix}",
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
                    filename=f"signal_to_noise_map{suffix}",
                ),
            )

    def subplot(
        self,
        data_with_noise_map: bool = False,
        data_with_noise_map_logy: bool = False,
        data=False,
        noise_map=False,
        signal_to_noise_map=False,
        pre_cti_data=False,
        auto_filename="subplot_dataset_1d",
        **kwargs,
    ):
        """
        Plots the individual attributes of the plotter's `Dataset1D` object in 1D on a subplot.

        The API is such that every plottable attribute of the `Imaging` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is included on the subplot.

        Parameters
        ----------
        data/
            Whether or not to include a 1D plot (via `plot`) of the data.
        noise_map
            Whether or not to include a 1D plot (via `plot`) of the noise map.
        signal_to_noise_map
            Whether or not to include a 1D plot (via `plot`) of the signal-to-noise map.
        pre_cti_data
            Whether or not to include a 1D plot (via `plot`) of the pre-cti data.
        """

        region = kwargs.get("region", None)
        suffix = f"_{region}" if region is not None else ""

        self._subplot_custom_plot(
            data_with_noise_map=data_with_noise_map,
            data_with_noise_map_logy=data_with_noise_map_logy,
            data=data,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            pre_cti_data=pre_cti_data,
            auto_labels=AutoLabels(filename=f"{auto_filename}{suffix}"),
            **kwargs,
        )

    def subplot_dataset_1d(self, region: Optional[str] = None):
        """
        Standard subplot of the attributes of the plotter's `Dataset1D` object.
        """
        self.subplot(
            region=region,
            data_with_noise_map=True,
            noise_map=True,
            signal_to_noise_map=True,
            pre_cti_data=True,
        )
