import numpy as np

import autoarray.plot as aplt

from autoarray.plot.mat_wrap.mat_plot import AutoLabels

from autocti.plot.abstract_plotters import Plotter
from autocti.line.fit import FitDataset1D


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
            The fit to the dataset of lines the plotter plots.
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

    def figures_1d(
        self,
        data=False,
        noise_map=False,
        signal_to_noise_map=False,
        pre_cti_data=False,
        post_cti_data=False,
        residual_map=False,
        normalized_residual_map=False,
        chi_squared_map=False,
    ):
        """
        Plots the individual attributes of the plotter's `FitDataset1D` object in 1D.

        The API is such that every plottable attribute of the `FitDataset1D` object is an input parameter of type bool
        of the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        image
            Whether or not to make a 1D plot (via `plot`) of the image data.
        noise_map
            Whether or not to make a 1D plot (via `plot`) of the noise map.
        signal_to_noise_map
            Whether or not to make a 1D plot (via `plot`) of the signal-to-noise map.
        pre_cti_data
            Whether or not to make a 1D plot (via `plot`) of the pre-cti data.
        post_cti_data
            Whether or not to make a 1D plot (via `plot`) of the post-cti data.
        residual_map
            Whether or not to make a 1D plot (via `plot`) of the residual map.
        normalized_residual_map
            Whether or not to make a 1D plot (via `plot`) of the normalized residual map.
        chi_squared_map
            Whether or not to make a 1D plot (via `plot`) of the chi-squared map.
        """
        if data:

            self.mat_plot_1d.plot_yx(
                y=self.fit.data,
                x=self.fit.data.grid_radial,
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(title="Image", filename="data"),
            )

        if noise_map:

            self.mat_plot_1d.plot_yx(
                y=self.fit.noise_map,
                x=self.fit.noise_map.grid_radial,
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(title="Noise-Map", filename="noise_map"),
            )

        if signal_to_noise_map:

            self.mat_plot_1d.plot_yx(
                y=self.fit.signal_to_noise_map,
                x=self.fit.signal_to_noise_map.grid_radial,
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title="Signal-To-Noise Map", filename="signal_to_noise_map"
                ),
            )

        if residual_map:

            self.mat_plot_1d.plot_yx(
                y=self.fit.residual_map,
                x=self.fit.residual_map.grid_radial,
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(title="Residual Map", filename="residual_map"),
            )

        if normalized_residual_map:

            self.mat_plot_1d.plot_yx(
                y=self.fit.normalized_residual_map,
                x=self.fit.normalized_residual_map.grid_radial,
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title="Normalized Residual Map", filename="normalized_residual_map"
                ),
            )

        if chi_squared_map:

            self.mat_plot_1d.plot_yx(
                y=self.fit.chi_squared_map,
                x=self.fit.chi_squared_map.grid_radial,
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title="Chi-Squared Map", filename="chi_squared_map"
                ),
            )

        if pre_cti_data:

            self.mat_plot_1d.plot_yx(
                y=self.fit.pre_cti_data,
                x=self.fit.pre_cti_data.grid_radial,
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title="CI Pre CTI Image", filename="pre_cti_data"
                ),
            )

        if post_cti_data:

            self.mat_plot_1d.plot_yx(
                y=self.fit.post_cti_data,
                x=self.fit.post_cti_data.grid_radial,
                visuals_1d=self.get_visuals_1d(),
                auto_labels=AutoLabels(
                    title="CI Post CTI Image", filename="post_cti_data"
                ),
            )

    def subplot(
        self,
        data=False,
        noise_map=False,
        signal_to_noise_map=False,
        pre_cti_data=False,
        post_cti_data=False,
        residual_map=False,
        normalized_residual_map=False,
        chi_squared_map=False,
        auto_filename="subplot_fit_dataset_line",
    ):
        """
        Plots the individual attributes of the plotter's `FitDataset1D` object in 1D on a subplot.

        The API is such that every plottable attribute of the `FitDataset1D` object is an input parameter of type bool
        of the function, which if switched to `True` means that it is included on the subplot.

        Parameters
        ----------
        image
            Whether or not to include a 1D plot (via `plot`) of the image data.
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

    def subplot_fit_dataset_line(self):
        """
        Standard subplot of the attributes of the plotter's `FitDataset1D` object.
        """
        return self.subplot(
            data=True,
            signal_to_noise_map=True,
            pre_cti_data=True,
            post_cti_data=True,
            normalized_residual_map=True,
            chi_squared_map=True,
        )
