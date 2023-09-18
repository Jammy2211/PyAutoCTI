import numpy as np
from typing import Callable

import autoarray.plot as aplt

from autoarray.fit.plot.fit_imaging_plotters import FitImagingPlotterMeta
from autoarray.plot.auto_labels import AutoLabels

from autocti.plot.abstract_plotters import Plotter
from autocti.charge_injection.fit import FitImagingCI


class FitImagingCIPlotter(Plotter):
    def __init__(
        self,
        fit: FitImagingCI,
        mat_plot_2d: aplt.MatPlot2D = aplt.MatPlot2D(),
        visuals_2d: aplt.Visuals2D = aplt.Visuals2D(),
        include_2d: aplt.Include2D = aplt.Include2D(),
        mat_plot_1d: aplt.MatPlot1D = aplt.MatPlot1D(),
        visuals_1d: aplt.Visuals1D = aplt.Visuals1D(),
        include_1d: aplt.Include1D = aplt.Include1D(),
        residuals_symmetric_cmap: bool = True,
    ):
        """
        Plots the attributes of `FitImagingCI` objects using the matplotlib methods `imshow()`, `plot()` and many other
        matplotlib functions which customize the plot's appearance.

        The `mat_plot_1d` and `mat_plot_2d` attribute wraps matplotlib function calls to make the figure. By default,
        the settings passed to every matplotlib function called are those specified in
        the `config/visualize/mat_wrap/*.ini` files, but a user can manually input values into `MatPlot1D` and
        `MatPlot2D` to customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` and `Visuals2D` object. Attributes may be
        extracted from the `FitImagingCI` and plotted via the visuals object, if the corresponding entry
        is `True` in the `Include1D` and `Include2D` object or the `config/visualize/include.ini` file.

        Parameters
        ----------
        fit
            The fit to an imaging dataset the plotter plots.
        get_visuals_2d
            A function which extracts from the `FitImaging` the 2D visuals which are plotted on figures.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make the plot.
        visuals_2d
            Contains visuals that can be overlaid on the plot.
        include_2d
            Specifies which attributes of the `Array2D` are extracted and plotted as visuals.
        mat_plot_1d
            Contains objects which wrap the matplotlib function calls that make 1D plots.
        visuals_1d
            Contains 1D visuals that can be overlaid on 1D plots.
        include_1d
            Specifies which attributes of the `ImagingCI` are extracted and plotted as visuals for 1D plots.
        residuals_symmetric_cmap
            If true, the `residual_map` and `normalized_residual_map` are plotted with a symmetric color map such
            that `abs(vmin) = abs(vmax)`.
        """
        super().__init__(
            dataset=fit.dataset,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

        self.visuals_1d = visuals_1d
        self.include_1d = include_1d
        self.mat_plot_1d = mat_plot_1d

        self.fit = fit

        self._fit_imaging_meta_plotter = FitImagingPlotterMeta(
            fit=self.fit,
            get_visuals_2d=self.get_visuals_2d,
            mat_plot_2d=self.mat_plot_2d,
            include_2d=self.include_2d,
            visuals_2d=self.visuals_2d,
            residuals_symmetric_cmap=residuals_symmetric_cmap,
        )

        self.residuals_symmetric_cmap = residuals_symmetric_cmap

    def get_visuals_2d(self):
        return self.get_2d.via_fit_imaging_ci_from(fit=self.fit)

    @property
    def extract_region_from(self) -> Callable:
        return self.fit.dataset.layout.extract_region_from

    @property
    def extract_region_noise_map_from(self) -> Callable:
        return self.fit.dataset.layout.extract_region_noise_map_from

    def figures_2d(
        self,
        data: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        pre_cti_data: bool = False,
        post_cti_data: bool = False,
        residual_map: bool = False,
        normalized_residual_map: bool = False,
        chi_squared_map: bool = False,
    ):
        """
        Plots the individual attributes of the plotter's `FitImagingCI` object in 2D.

        The API is such that every plottable attribute of the `FitImagingCI` object is an input parameter of type bool
        of the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        data
            Whether to make a 2D plot (via `imshow`) of the image data.
        noise_map
            Whether to make a 2D plot (via `imshow`) of the noise map.
        signal_to_noise_map
            Whether to make a 2D plot (via `imshow`) of the signal-to-noise map.
        pre_cti_data
            Whether to make a 2D plot (via `imshow`) of the pre-cti data.
        post_cti_data
            Whether to make a 2D plot (via `imshow`) of the post-cti data.
        residual_map
            Whether to make a 2D plot (via `imshow`) of the residual map.
        normalized_residual_map
            Whether to make a 2D plot (via `imshow`) of the normalized residual map.
        chi_squared_map
            Whether to make a 2D plot (via `imshow`) of the chi-squared map.
        """
        self._fit_imaging_meta_plotter.figures_2d(
            data=data,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            residual_map=residual_map,
            normalized_residual_map=normalized_residual_map,
            chi_squared_map=chi_squared_map,
        )

        if pre_cti_data:
            self.mat_plot_2d.plot_array(
                array=self.fit.pre_cti_data,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=AutoLabels(title="Pre CTI Data", filename="pre_cti_data"),
            )

        if post_cti_data:
            self.mat_plot_2d.plot_array(
                array=self.fit.post_cti_data,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=AutoLabels(
                    title="CI Post CTI Image", filename="post_cti_data"
                ),
            )

    def figures_1d(
        self,
        region,
        data: bool = False,
        data_logy: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        pre_cti_data: bool = False,
        post_cti_data: bool = False,
        residual_map: bool = False,
        residual_map_logy: bool = False,
        normalized_residual_map: bool = False,
        chi_squared_map: bool = False,
    ):
        """
        Plots the individual attributes of the plotter's `FitImagingCI` object in 1D.

        These 1D plots correspond to a region in 2D on the charge injection image, which is binned up over the parallel
        or serial direction to produce a 1D plot. For example, for the input `region=parallel_fpr`, this
        function extracts the FPR over each charge injection region and bins such that the 1D plot shows the FPR
        in the parallel direction.

        The API is such that every plottable attribute of the `FitImagingCI` object is an input parameter of type bool
        of the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        region
            The region on the charge injection image where data is extracted and binned over the parallel or serial
            direction {"parallel_fpr", "parallel_eper", "serial_fpr", "serial_eper"}.
        data
            Whether to make a 1D plot (via `plot`) of the image data extracted and binned over the region, with the
            noise-map values included as error bars and the model-fit overlaid.
        data_logy
            Whether to make a 1D plot (via `plot`) of the image data extracted and binned over the region, with the
            noise-map values included as error bars and the y-axis on a log10 scale and the model-fit overlaid.
        data
            Whether to make a 1D plot (via `plot`) of the image data extracted and binned over the region.
        noise_map
            Whether to make a 1D plot (via `plot`) of the noise-map extracted and binned over the region.
        signal_to_noise_map
            Whether to make a 1D plot (via `plot`) of the signal-to-noise map data extracted and binned over
            the region.
        pre_cti_data
            Whether to make a 1D plot (via `plot`) of the pre-cti data extracted and binned over the region.
        post_cti_data
            Whether to make a 1D plot (via `plot`) of the post-cti data extracted and binned over the
            region.
        residual_map
            Whether to make a 1D plot (via `plot`) of the residual map extracted and binned over the region, with the
            noise-map values included as error bars and the model-fit overlaid.
        residual_map_logy
            Whether to make a 1D plot (via `plot`) of the residual map extracted and binned over the region, with the
            noise-map values included as error bars and the y-axis on a log10 scale and the model-fit overlaid.
        normalized_residual_map
            Whether to make a 1D plot (via `plot`) of the normalized residual map extracted and binned over the
            region.
        chi_squared_map
            Whether to make a 1D plot (via `plot`) of the chi-squared map extracted and binned over the
            region.
        """

        title_str = self.title_str_from(region=region)

        y = self.extract_region_from(array=self.fit.data, region=region)
        y_errors = self.extract_region_noise_map_from(
            array=self.fit.noise_map, region=region
        )
        y_extra = self.extract_region_from(array=self.fit.model_data, region=region)

        should_plot_zero = self.should_plot_zero_from(region=region)

        if data:
            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                plot_axis_type_override="errorbar",
                y_errors=y_errors,
                y_extra=y_extra,
                should_plot_zero=should_plot_zero,
                text_manual_dict=self.text_manual_dict_from(region=region),
                text_manual_dict_y=self.text_manual_dict_y_from(region=region),
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title=f"Data {title_str}",
                    yunit="e-",
                    filename=f"data_{region}",
                ),
            )

        if data_logy:
            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                plot_axis_type_override="errorbar_logy",
                y_errors=y_errors,
                y_extra=y_extra,
                text_manual_dict=self.text_manual_dict_from(region=region),
                text_manual_dict_y=self.text_manual_dict_y_from(region=region),
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title=f"Data {title_str} [log10]",
                    yunit="e-",
                    filename=f"data_logy_{region}",
                ),
            )

        if noise_map:
            y = self.extract_region_from(array=self.fit.image, region=region)

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title=f"Noise-Map {title_str}",
                    yunit="e-",
                    filename=f"noise_map_{region}",
                ),
            )

        if signal_to_noise_map:
            y = self.extract_region_from(
                array=self.fit.signal_to_noise_map, region=region
            )

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title=f"Signal-To-Noise Map {title_str}",
                    yunit="",
                    filename=f"signal_to_noise_map_{region}",
                ),
            )

        if pre_cti_data:
            y = self.extract_region_from(array=self.fit.pre_cti_data, region=region)

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title=f"CI Pre CTI {title_str}",
                    yunit="e-",
                    filename=f"pre_cti_data_{region}",
                ),
            )

        if post_cti_data:
            y = self.extract_region_from(array=self.fit.post_cti_data, region=region)

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                should_plot_zero=should_plot_zero,
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title=f"CI Post CTI {title_str}",
                    yunit="e-",
                    filename=f"post_cti_data_{region}",
                ),
            )

        if residual_map:
            y = self.extract_region_from(array=self.fit.residual_map, region=region)

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                plot_axis_type_override="errorbar",
                y_errors=y_errors,
                should_plot_zero=should_plot_zero,
                text_manual_dict=self.text_manual_dict_from(region=region),
                text_manual_dict_y=self.text_manual_dict_y_from(region=region),
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title=f"Residual Map {title_str}",
                    yunit="e-",
                    filename=f"residual_map_{region}",
                ),
            )

        if residual_map_logy:
            y = self.extract_region_from(array=self.fit.residual_map, region=region)

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                plot_axis_type_override="errorbar_logy",
                y_errors=y_errors,
                text_manual_dict=self.text_manual_dict_from(region=region),
                text_manual_dict_y=self.text_manual_dict_y_from(region=region),
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title=f"Residual Map {title_str} [log10]",
                    yunit="e-",
                    filename=f"residual_map_logy_{region}",
                ),
            )

        if normalized_residual_map:
            y = self.extract_region_from(
                array=self.fit.normalized_residual_map, region=region
            )

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title=f"Normalized Residual Map {title_str}",
                    yunit=r"$\sigma$",
                    filename=f"normalized_residual_map_{region}",
                ),
            )

        if chi_squared_map:
            y = self.extract_region_from(array=self.fit.chi_squared_map, region=region)

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title=f"Chi-Squared Map {region}",
                    yunit=r"$\chi^2$",
                    filename=f"chi_squared_map_{region}",
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
        auto_filename: str = "subplot_fit",
    ):
        """
        Plots the individual attributes of the plotter's `FitImagingCI` object in 2D on a subplot.

        The API is such that every plottable attribute of the `FitImagingCI` object is an input parameter of type bool
        of the function, which if switched to `True` means that it is included on the subplot.

        Parameters
        ----------
        data
            Whether to include a 2D plot (via `imshow`) of the image data.
        noise_map
            Whether to include a 2D plot (via `imshow`) noise map.
        signal_to_noise_map
            Whether to include a 2D plot (via `imshow`) signal-to-noise map.
        pre_cti_data
            Whether to include a 2D plot (via `imshow`) of the pre-cti data.
        post_cti_data
            Whether to include a 2D plot (via `imshow`) of the post-cti data.
        residual_map
            Whether to include a 2D plot (via `imshow`) residual map.
        normalized_residual_map
            Whether to include a 2D plot (via `imshow`) normalized residual map.
        chi_squared_map
            Whether to include a 2D plot (via `imshow`) chi-squared map.
        auto_filename
            The default filename of the output subplot if written to hard-disk.
        """
        self._subplot_custom_plot(
            data=data,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            pre_cti_data=pre_cti_data,
            post_cti_data=post_cti_data,
            residual_map=residual_map,
            normalized_residual_map=normalized_residual_map,
            chi_squared_map=chi_squared_map,
            auto_labels=AutoLabels(filename=auto_filename),
        )

    def subplot_fit(self):
        """
        Standard subplot of the attributes of the plotter's `FitImaging` object.
        """
        return self.subplot(
            data=True,
            signal_to_noise_map=True,
            pre_cti_data=True,
            post_cti_data=True,
            normalized_residual_map=True,
            chi_squared_map=True,
        )

    def subplot_1d(self, region: str):
        """
        Plots the individual attributes of the plotter's `FitImagingCI` object in 1D on a subplot.

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

        self.open_subplot_figure(number_subplots=6)

        self.figures_1d(data=True, region=region)
        self.figures_1d(signal_to_noise_map=True, region=region)
        self.figures_1d(pre_cti_data=True, region=region)
        self.figures_1d(post_cti_data=True, region=region)
        self.figures_1d(normalized_residual_map=True, region=region)
        self.figures_1d(chi_squared_map=True, region=region)

        self.mat_plot_1d.output.subplot_to_figure(
            auto_filename=f"subplot_1d_fit_ci_{region}"
        )
        self.close_subplot_figure()

    def subplot_noise_scaling_map_dict(self):
        """
        Plots the noise-scaling maps of the plotter's `FitImagingCI` object in 2D on a subplot.
        """

        self.open_subplot_figure(number_subplots=len(self.fit.noise_scaling_map_dict))

        for key, noise_scaling_map in self.fit.noise_scaling_map_dict.items():
            self.mat_plot_2d.plot_array(
                array=noise_scaling_map,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=AutoLabels(title=f"Noise Scaling Map {key}"),
            )

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename=f"subplot_noise_scaling_map_dict"
        )
        self.mat_plot_2d.figure.close()
