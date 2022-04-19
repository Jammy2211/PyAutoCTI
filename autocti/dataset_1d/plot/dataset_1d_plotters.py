import numpy as np

import autoarray.plot as aplt

from autoarray.plot.mat_wrap.mat_plot import AutoLabels

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

        self.dataset_1d = dataset

    def get_visuals_1d(self) -> aplt.Visuals1D:
        return self.visuals_1d

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
