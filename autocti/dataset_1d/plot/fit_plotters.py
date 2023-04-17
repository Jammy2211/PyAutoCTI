from typing import Callable, Optional

import autoarray.plot as aplt

from autoarray.plot.auto_labels import AutoLabels

from autocti.plot.abstract_plotters import Plotter
from autocti.dataset_1d.fit import FitDataset1D


class FitDataset1DPlotter(Plotter):
    def __init__(
        self,
        fit: FitDataset1D,
        mat_plot_1d: aplt.MatPlot1D = aplt.MatPlot1D(),
        visuals_1d: aplt.Visuals1D = aplt.Visuals1D(),
        include_1d: aplt.Include1D = aplt.Include1D(),
    ):
        """
        Plots the attributes of `FitDataset1D` objects using the matplotlib method `line()` and many other matplotlib
        functions which customize the plot's appearance.

        The `mat_plot_1d` attribute wraps matplotlib function calls to make the figure. By default, the settings
        passed to every matplotlib function called are those specified in the `config/visualize/mat_wrap/*.ini` files,
        but a user can manually input values into `MatPlot1d` to customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` object. Attributes may be extracted from
        the `Imaging` and plotted via the visuals object, if the corresponding entry is `True` in the `Include1D`
        object or the `config/visualize/include.ini` file.

        Parameters
        ----------
        fit
            The fit to the dataset of a 1D dataset the plotter plots.
        mat_plot_1d
            Contains objects which wrap the matplotlib function calls that make 1D plots.
        visuals_1d
            Contains 1D visuals that can be overlaid on 1D plots.
        include_1d
            Specifies which attributes of the `ImagingCI` are extracted and plotted as visuals for 1D plots.
        """
        self.fit = fit

        super().__init__(
            mat_plot_1d=mat_plot_1d, include_1d=include_1d, visuals_1d=visuals_1d
        )

    def get_visuals_1d(self) -> aplt.Visuals1D:
        return self.visuals_1d

    @property
    def extract_region_from(self) -> Callable:
        return self.fit.dataset.layout.extract_region_from

    def figures_1d(
        self,
        region: Optional[str] = None,
        data: bool = False,
        data_logy: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        pre_cti_data: bool = False,
        post_cti_data: bool = False,
        residual_map: bool = False,
        normalized_residual_map: bool = False,
        chi_squared_map: bool = False,
    ):
        """
        Plots the individual attributes of the plotter's `FitDataset1D` object in 1D.

        The API is such that every plottable attribute of the `FitDataset1D` object is an input parameter of type bool
        of the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        region
            The region on the 1D dataset where data is extracted and binned {fpr", "eper"}
        data
            Whether to make a 1D plot (via `plot`) of the image data extracted and binned over the region, with the
            noise-map values included as error bars.
        data_logy
            Whether to make a 1D plot (via `plot`) of the image data extracted and binned over the region, with the
            noise-map values included as error bars and the y-axis on a log10 scale.
        noise_map
            Whether to make a 1D plot (via `plot`) of the noise map.
        signal_to_noise_map
            Whether to make a 1D plot (via `plot`) of the signal-to-noise map.
        pre_cti_data
            Whether to make a 1D plot (via `plot`) of the pre-cti data.
        post_cti_data
            Whether to make a 1D plot (via `plot`) of the post-cti data.
        residual_map
            Whether to make a 1D plot (via `plot`) of the residual map.
        normalized_residual_map
            Whether to make a 1D plot (via `plot`) of the normalized residual map.
        chi_squared_map
            Whether to make a 1D plot (via `plot`) of the chi-squared map.
        """

        suffix = f"_{region}" if region is not None else ""

        if data:

            y = self.extract_region_from(array=self.fit.data, region=region)
            y_errors = self.extract_region_from(array=self.fit.noise_map, region=region)
            y_extra = self.extract_region_from(array=self.fit.model_data, region=region)

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                plot_axis_type_override="errorbar",
                y_errors=y_errors,
                y_extra=y_extra,
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title=f"Data 1D With Noise {region}",
                    ylabel="Data (e-)",
                    xlabel="Pixel No.",
                    filename=f"data{suffix}",
                ),
            )

        if data_logy:

            y = self.extract_region_from(array=self.fit.data, region=region)
            y_errors = self.extract_region_from(array=self.fit.noise_map, region=region)
            y_extra = self.extract_region_from(array=self.fit.model_data, region=region)

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                plot_axis_type_override="errorbar_logy",
                y_errors=y_errors,
                y_extra=y_extra,
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title=f"Data 1D With Noise {region} (log10 y axis)",
                    ylabel="Data (e-)",
                    xlabel="Pixel No.",
                    filename=f"data_logy{suffix}",
                ),
            )

        if noise_map:

            y = self.extract_region_from(array=self.fit.noise_map, region=region)

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title="Noise-Map",
                    ylabel="Data (e-)",
                    xlabel="Pixel No.",
                    filename=f"noise_map{suffix}",
                ),
            )

        if signal_to_noise_map:

            y = self.extract_region_from(
                array=self.fit.signal_to_noise_map, region=region
            )

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title="Signal-To-Noise Map",
                    ylabel="Data (e-)",
                    xlabel="Pixel No.",
                    filename=f"signal_to_noise_map{suffix}",
                ),
            )

        if residual_map:

            y = self.extract_region_from(array=self.fit.residual_map, region=region)

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title="Residual Map",
                    ylabel="Data (e-)",
                    xlabel="Pixel No.",
                    filename=f"residual_map{suffix}",
                ),
            )

        if normalized_residual_map:

            y = self.extract_region_from(
                array=self.fit.normalized_residual_map, region=region
            )

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title="Normalized Residual Map",
                    ylabel="Data (e-)",
                    xlabel="Pixel No.",
                    filename=f"normalized_residual_map{suffix}",
                ),
            )

        if chi_squared_map:

            y = self.extract_region_from(array=self.fit.chi_squared_map, region=region)

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title="Chi-Squared Map",
                    ylabel="Data (e-)",
                    xlabel="Pixel No.",
                    filename=f"chi_squared_map{suffix}",
                ),
            )

        if pre_cti_data:

            y = self.extract_region_from(array=self.fit.pre_cti_data, region=region)

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title="CI Pre CTI Image",
                    ylabel="Data (e-)",
                    xlabel="Pixel No.",
                    filename=f"pre_cti_data{suffix}",
                ),
            )

        if post_cti_data:

            y = self.extract_region_from(array=self.fit.post_cti_data, region=region)

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title="CI Post CTI Image",
                    ylabel="Data (e-)",
                    xlabel="Pixel No.",
                    filename=f"post_cti_data{suffix}",
                ),
            )

    def subplot(
        self,
        data: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        pre_cti_data: bool = False,
        post_cti_data: bool = False,
        residual_map: bool = False,
        normalized_residual_map: bool = False,
        chi_squared_map: bool = False,
        auto_filename="subplot_fit_dataset_1d",
        **kwargs,
    ):
        """
        Plots the individual attributes of the plotter's `FitDataset1D` object in 1D on a subplot.

        The API is such that every plottable attribute of the `FitDataset1D` object is an input parameter of type bool
        of the function, which if switched to `True` means that it is included on the subplot.

        Parameters
        ----------
        data
            Whether to make a 1D plot (via `plot`) of the image data extracted and binned over the region, with the
            noise-map values included as error bars.
        noise_map
            Whether or not to include a 1D plot (via `plot`) of the noise map.
        signal_to_noise_map
            Whether or not to include a 1D plot (via `plot`) of the signal-to-noise map.
        pre_cti_data
            Whether or not to include a 1D plot (via `plot`) of the pre-cti data.
        post_cti_data
            Whether or not to include a 1D plot (via `plot`) of the post-cti data.
        residual_map
            Whether or not to include a 1D plot (via `plot`) of the residual map.
        normalized_residual_map
            Whether or not to include a 1D plot (via `plot`) of the normalized residual map.
        chi_squared_map
            Whether or not to include a 1D plot (via `plot`) of the chi-squared map.
        """

        region = kwargs.get("region", None)
        suffix = f"_{region}" if region is not None else ""

        self._subplot_custom_plot(
            data=data,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            pre_cti_data=pre_cti_data,
            post_cti_data=post_cti_data,
            residual_map=residual_map,
            normalized_residual_map=normalized_residual_map,
            chi_squared_map=chi_squared_map,
            auto_labels=AutoLabels(
                ylabel="Data (e-)",
                xlabel="Pixel No.",
                filename=f"{auto_filename}{suffix}",
            ),
        )

    def subplot_fit_dataset_1d(self, region: Optional[str] = None):
        """
        Standard subplot of the attributes of the plotter's `FitDataset1D` object.
        """
        return self.subplot(
            region=region,
            data=True,
            signal_to_noise_map=True,
            pre_cti_data=True,
            post_cti_data=True,
            normalized_residual_map=True,
            chi_squared_map=True,
        )
