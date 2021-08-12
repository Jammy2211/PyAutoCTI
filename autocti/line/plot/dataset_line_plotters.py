import numpy as np

from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import mat_plot as mp
from autoarray.plot import abstract_plotters
from autocti.line.dataset import DatasetLine


class DatasetLinePlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        dataset_line: DatasetLine,
        mat_plot_1d: mp.MatPlot1D = mp.MatPlot1D(),
        visuals_1d: vis.Visuals1D = vis.Visuals1D(),
        include_1d: inc.Include1D = inc.Include1D(),
    ):

        super().__init__(
            mat_plot_1d=mat_plot_1d, include_1d=include_1d, visuals_1d=visuals_1d
        )

        self.dataset_line = dataset_line

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

    def figures_1d(
        self, data=False, noise_map=False, signal_to_noise_map=False, pre_cti_data=False
    ):
        """
        Plot each attribute of the line data_type as individual figures one by one (e.g. the dataset, noise_map, PSF, \
         Signal-to_noise-map, etc).

        Set *autocti.data_type.array.plotter.plotter* for a description of all innput parameters not described below.

        Parameters
        -----------
        imaging_ci : data_type.ImagingData
            The line data_type, which includes the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
        origin : True
            If true, the origin of the dataset's coordinate system is plotted as a 'x'.
        """

        if data:

            self.mat_plot_1d.plot_yx(
                y=self.dataset_line.data,
                x=np.arange(len(self.dataset_line.data)),
                visuals_1d=self.visuals_with_include_1d,
                auto_labels=mp.AutoLabels(title="Line Dataset Line", filename="data"),
            )

        if noise_map:

            self.mat_plot_1d.plot_yx(
                y=self.dataset_line.noise_map,
                x=np.arange(len(self.dataset_line.noise_map)),
                visuals_1d=self.visuals_with_include_1d,
                auto_labels=mp.AutoLabels(
                    title="Line Dataset Noise Map", filename="noise_map"
                ),
            )

        if signal_to_noise_map:

            self.mat_plot_1d.plot_yx(
                y=self.dataset_line.signal_to_noise_map,
                x=np.arange(len(self.dataset_line.signal_to_noise_map)),
                visuals_1d=self.visuals_with_include_1d,
                auto_labels=mp.AutoLabels(
                    title="Line Dataset Signal-To-Noise Map",
                    filename="signal_to_noise_map",
                ),
            )

        if pre_cti_data:

            self.mat_plot_1d.plot_yx(
                y=self.dataset_line.pre_cti_data,
                x=np.arange(len(self.dataset_line.pre_cti_data)),
                visuals_1d=self.visuals_with_include_1d,
                auto_labels=mp.AutoLabels(
                    title="Line Dataset Pre CTI Line", filename="pre_cti_data"
                ),
            )

    def subplot(
        self,
        data=False,
        noise_map=False,
        signal_to_noise_map=False,
        pre_cti_data=False,
        auto_filename="subplot_dataset_line",
    ):

        self._subplot_custom_plot(
            data=data,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            pre_cti_data=pre_cti_data,
            auto_labels=mp.AutoLabels(filename=auto_filename),
        )

    def subplot_dataset_line(self):
        """Plot the line data_type as a sub-plotter of all its quantites (e.g. the dataset, noise_map, PSF, Signal-to_noise-map, \
         etc).

        Set *autocti.data_type.array.plotter.plotter* for a description of all innput parameters not described below.

        Parameters
        -----------
        imaging_ci : data_type.ImagingData
            The line data_type, which includes the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
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
            data=True, noise_map=True, signal_to_noise_map=True, pre_cti_data=True
        )
