import numpy as np
from typing import Callable

from autoconf import conf

import autoarray.plot as aplt

from autoarray.plot.auto_labels import AutoLabels
from autoarray.dataset.plot.imaging_plotters import ImagingPlotterMeta

from autocti.plot.abstract_plotters import Plotter
from autocti.charge_injection.imaging.imaging import ImagingCI
from autocti.extract.settings import SettingsExtract

from autocti import exc


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
        residuals_symmetric_cmap: bool = True,
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
        residuals_symmetric_cmap
            If true, the `pre_cti_residual_map` is plotted with a symmetric color map such that `abs(vmin) = abs(vmax)`.
        """
        super().__init__(
            dataset=dataset,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

        self.visuals_1d = visuals_1d
        self.include_1d = include_1d
        self.mat_plot_1d = mat_plot_1d

        self._imaging_meta_plotter = ImagingPlotterMeta(
            dataset=self.dataset,
            get_visuals_2d=self.get_visuals_2d,
            mat_plot_2d=self.mat_plot_2d,
            include_2d=self.include_2d,
            visuals_2d=self.visuals_2d,
        )

        self.residuals_symmetric_cmap = residuals_symmetric_cmap

    def get_visuals_1d(self):
        return self.visuals_1d

    def get_visuals_2d(self):
        return self.get_2d.via_mask_from(mask=self.dataset.mask)

    @property
    def extract_region_from(self) -> Callable:
        return self.dataset.layout.extract_region_from

    @property
    def extract_region_noise_map_from(self) -> Callable:
        return self.dataset.layout.extract_region_noise_map_from

    def figures_2d(
        self,
        data: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        pre_cti_data: bool = False,
        pre_cti_data_residual_map: bool = False,
        cosmic_ray_map: bool = False,
    ):
        """
        Plots the individual attributes of the plotter's `ImagingCI` object in 2D.

        The API is such that every plottable attribute of the `Imaging` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

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
        pre_cti_data_residual_map
            Whether to make a 2D plot (via `imshow`) of the pre-cti data residual-map.
        cosmic_ray_map
            Whether to make a 2D plot (via `imshow`) of the cosmic ray map.
        """

        title_str = self.title_str_2d_from()

        self._imaging_meta_plotter.figures_2d(
            data=data,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            title_str=title_str,
        )

        if pre_cti_data:
            self.mat_plot_2d.plot_array(
                array=self.dataset.pre_cti_data,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=AutoLabels(
                    title=title_str or f"Pre CTI Data", filename="pre_cti_data"
                ),
            )

        cmap_original = self.mat_plot_2d.cmap

        if self.residuals_symmetric_cmap:
            symmetric_value = conf.instance["visualize"]["general"]["general"][
                "symmetric_cmap_value"
            ]

            self.mat_plot_2d.cmap = self.mat_plot_2d.cmap.symmetric_cmap_from(
                symmetric_value=symmetric_value
            )

        if pre_cti_data_residual_map:
            self.mat_plot_2d.plot_array(
                array=self.dataset.pre_cti_data_residual_map,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=AutoLabels(
                    title=title_str or f"Pre CTI Data Residual Map",
                    filename="pre_cti_data_residual_map",
                ),
            )

        self.mat_plot_2d.cmap = cmap_original

        if cosmic_ray_map:
            self.mat_plot_2d.plot_array(
                array=self.dataset.cosmic_ray_map,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=AutoLabels(
                    title=title_str or f"Cosmic Ray Map", filename="cosmic_ray_map"
                ),
            )

    def figures_1d(
        self,
        region: str,
        data: bool = False,
        data_logy: bool = False,
        noise_map: bool = False,
        pre_cti_data: bool = False,
        signal_to_noise_map: bool = False,
    ):
        """
        Plots the individual attributes of the plotter's `ImagingCI` object in 1D.

        The API is such that every plottable attribute of the `Imaging` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        if a `region` string is input, the 1D plots correspond to a region in 2D on the charge injection image, which
        is binned up over the parallel or serial direction to produce a 1D plot. For example, for the
        input `region=parallel_fpr`, this function extracts the FPR over each charge injection region and bins such
        that the 1D plot shows the FPR in the parallel direction.

        The API is such that every plottable attribute of the `Imaging` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        region
            The region on the charge injection image where data is extracted and binned over the parallel or serial
            direction {"parallel_fpr", "parallel_eper", "serial_fpr", "serial_eper"}
        data
            Whether to make a 1D plot (via `plot`) of the image data extracted and binned over the region, with the
            noise-map values included as error bars.
        data_logy
            Whether to make a 1D plot (via `plot`) of the image data extracted and binned over the region, with the
            noise-map values included as error bars and the y-axis on a log10 scale.
        noise_map
            Whether to make a 1D plot (via `plot`) of the noise-map extracted and binned over the region.
        pre_cti_data
            Whether to make a 1D plot (via `plot`) of the pre-cti data extracted and binned over the region.
        signal_to_noise_map
            Whether to make a 1D plot (via `plot`) of the signal-to-noise map data extracted and binned over
            the region.
        """

        if region == "fpr_non_uniformity":
            ls_errorbar = "-"
        else:
            ls_errorbar = ""

        should_plot_zero = self.should_plot_zero_from(region=region)

        title_str = self.title_str_from(region=region)

        if data:
            y = self.extract_region_from(array=self.dataset.data, region=region)
            y_errors = self.extract_region_noise_map_from(
                array=self.dataset.noise_map, region=region
            )

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                plot_axis_type_override="errorbar",
                y_errors=y_errors,
                ls_errorbar=ls_errorbar,
                should_plot_zero=should_plot_zero,
                text_manual_dict=self.text_manual_dict_from(region=region),
                text_manual_dict_y=self.text_manual_dict_y_from(region=region),
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title=f"Data {title_str}",
                    yunit="e-",
                    filename=f"data_{region}",
                ),
            )

        if data_logy:
            y = self.extract_region_from(array=self.dataset.data, region=region)
            y_errors = self.extract_region_noise_map_from(
                array=self.dataset.noise_map, region=region
            )

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                plot_axis_type_override="errorbar_logy",
                y_errors=y_errors,
                text_manual_dict=self.text_manual_dict_from(region=region),
                text_manual_dict_y=self.text_manual_dict_y_from(region=region),
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title=f"Data {title_str} [log10 y]",
                    yunit="e-",
                    filename=f"data_logy_{region}",
                ),
            )

        if noise_map:
            y = self.extract_region_from(array=self.dataset.noise_map, region=region)

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title=f"Noise Map {title_str}",
                    ylabel="Noise (e-)",
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
                    title=f"CI Pre CTI {title_str}",
                    yunit="e-",
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
                    title=f"Signal To Noise Map {title_str}",
                    ylabel="Signal To Noise (e-)",
                    filename=f"signal_to_noise_map_{region}",
                ),
            )

    def figures_1d_data_binned(
        self,
        rows_fpr: bool = False,
        rows_no_fpr: bool = False,
        columns_fpr: bool = False,
        columns_no_fpr: bool = False,
    ):
        """
        Plots the charge injection data binned over the parallel and serial directions, with and without the
        FPR regions included.

        Plots binned over rows show the FPR of each injection, so that the FPR can be compared between injections.
        When the FPR is masked it allows comparison of the parallel EPER of each injection.

        Plots binned over columns shown the charge injection non-uniformity.

        Inaccurate bias subtraction / stray light subtraction and other systematics can produce a gradient over
        a full image, which these plots often show.

        Parameters
        ----------
        columns_fpr
            Whether to plot the data binned over columns with the FPR regions included.
        """

        fpr_size = self.dataset.layout.parallel_rows_within_regions[0]

        if any(
            [
                fpr_size != fpr_size_of_row
                for fpr_size_of_row in self.dataset.layout.parallel_rows_within_regions
            ]
        ):
            raise exc.PlottingException(
                "The FPR in this dataset have a variable number of rows. This means that masknig the FPR in the"
                "figures_1d_data_binned method is not supported."
            )

        fpr_mask = self.dataset.layout.extract.parallel_fpr.mask_from(
            settings=SettingsExtract(pixels=(0, fpr_size)),
            pixel_scales=self.dataset.pixel_scales,
        )

        if rows_fpr:
            y = self.dataset.data.apply_mask(
                mask=np.invert(fpr_mask)
            ).binned_across_rows

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                text_manual_dict=self.text_manual_dict_from(),
                text_manual_dict_y=self.text_manual_dict_y_from(),
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title=f"Data No FPR Binned Over Rows",
                    yunit="e-",
                    filename=f"data_binned_rows_fpr",
                ),
            )

        if rows_no_fpr:
            y = self.dataset.data.apply_mask(mask=fpr_mask).binned_across_rows

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                text_manual_dict=self.text_manual_dict_from(),
                text_manual_dict_y=self.text_manual_dict_y_from(),
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title=f"Data No FPR Binned Over Rows",
                    yunit="e-",
                    filename=f"data_binned_rows_no_fpr",
                ),
            )

        if columns_fpr:
            y = self.dataset.data.apply_mask(
                mask=np.invert(fpr_mask)
            ).binned_across_columns

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                text_manual_dict=self.text_manual_dict_from(),
                text_manual_dict_y=self.text_manual_dict_y_from(),
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title=f"Data No FPR Binned Over Columns",
                    yunit="e-",
                    filename=f"data_binned_columns_fpr",
                ),
            )

        if columns_no_fpr:
            y = self.dataset.data.apply_mask(mask=fpr_mask).binned_across_columns

            self.mat_plot_1d.plot_yx(
                y=y,
                x=range(len(y)),
                text_manual_dict=self.text_manual_dict_from(),
                text_manual_dict_y=self.text_manual_dict_y_from(),
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title=f"Data No FPR Binned Over Columns",
                    yunit="e-",
                    filename=f"data_binned_columns_no_fpr",
                ),
            )

    def subplot(
        self,
        data: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        pre_cti_data: bool = False,
        cosmic_ray_map: bool = False,
        auto_filename="subplot_dataset",
    ):
        """
        Plots the individual attributes of the plotter's `ImagingCI` object in 2D on a subplot.

        The API is such that every plottable attribute of the `Imaging` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is included on the subplot.

        Parameters
        ----------
        data
            Whether to include a 2D plot (via `imshow`) of the image data.
        noise_map
            Whether to include a 2D plot (via `imshow`) of the noise map.
        signal_to_noise_map
            Whether to include a 2D plot (via `imshow`) of the signal-to-noise map.
        pre_cti_data
            Whether to include a 2D plot (via `imshow`) of the pre-cti data.
        cosmic_ray_map
            Whether to include a 2D plot (via `imshow`) of the cosmic ray map.
        auto_filename
            The default filename of the output subplot if written to hard-disk.
        """
        self._subplot_custom_plot(
            data=data,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            pre_cti_data=pre_cti_data,
            cosmic_ray_map=cosmic_ray_map,
            auto_labels=AutoLabels(filename=auto_filename),
        )

    def subplot_dataset(self):
        """
        Standard subplot of the attributes of the plotter's `ImagingCI` object.
        """
        self.subplot(
            data=True,
            noise_map=True,
            signal_to_noise_map=True,
            pre_cti_data=True,
            cosmic_ray_map=True,
        )

    def subplot_1d(self, region: str):
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

        self.figures_1d(data=True, region=region)
        self.figures_1d(noise_map=True, region=region)
        self.figures_1d(pre_cti_data=True, region=region)
        self.figures_1d(signal_to_noise_map=True, region=region)

        self.mat_plot_1d.output.subplot_to_figure(
            auto_filename=f"subplot_1d_ci_{region}"
        )
        self.close_subplot_figure()
