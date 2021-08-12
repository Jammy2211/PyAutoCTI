import numpy as np
from autoarray.plot.mat_wrap.visuals import Visuals1D
from autoarray.plot.mat_wrap.include import Include1D
from autoarray.plot.mat_wrap.mat_plot import MatPlot1D
from autoarray.plot.mat_wrap.mat_plot import AutoLabels
from autoarray.plot.abstract_plotters import AbstractPlotter
from autocti.line.fit import FitDatasetLine


class FitDatasetLinePlotter(AbstractPlotter):
    def __init__(
        self,
        fit: FitDatasetLine,
        mat_plot_1d: MatPlot1D = MatPlot1D(),
        visuals_1d: Visuals1D = Visuals1D(),
        include_1d: Include1D = Include1D(),
    ):

        self.fit = fit

        super().__init__(
            mat_plot_1d=mat_plot_1d, include_1d=include_1d, visuals_1d=visuals_1d
        )

    @property
    def visuals_with_include_1d(self):
        return self.visuals_1d + self.visuals_1d.__class__()

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

        if data:

            self.mat_plot_1d.plot_yx(
                y=self.fit.data,
                x=np.arange(len(self.fit.data)),
                visuals_1d=self.visuals_with_include_1d,
                auto_labels=AutoLabels(title="Image", filename="data"),
            )

        if noise_map:

            self.mat_plot_1d.plot_yx(
                y=self.fit.noise_map,
                x=np.arange(len(self.fit.noise_map)),
                visuals_1d=self.visuals_with_include_1d,
                auto_labels=AutoLabels(title="Noise-Map", filename="noise_map"),
            )

        if signal_to_noise_map:

            self.mat_plot_1d.plot_yx(
                y=self.fit.signal_to_noise_map,
                x=np.arange(len(self.fit.signal_to_noise_map)),
                visuals_1d=self.visuals_with_include_1d,
                auto_labels=AutoLabels(
                    title="Signal-To-Noise Map", filename="signal_to_noise_map"
                ),
            )

        if residual_map:

            self.mat_plot_1d.plot_yx(
                y=self.fit.residual_map,
                x=np.arange(len(self.fit.residual_map)),
                visuals_1d=self.visuals_with_include_1d,
                auto_labels=AutoLabels(title="Residual Map", filename="residual_map"),
            )

        if normalized_residual_map:

            self.mat_plot_1d.plot_yx(
                y=self.fit.normalized_residual_map,
                x=np.arange(len(self.fit.normalized_residual_map)),
                visuals_1d=self.visuals_with_include_1d,
                auto_labels=AutoLabels(
                    title="Normalized Residual Map", filename="normalized_residual_map"
                ),
            )

        if chi_squared_map:

            self.mat_plot_1d.plot_yx(
                y=self.fit.chi_squared_map,
                x=np.arange(len(self.fit.chi_squared_map)),
                visuals_1d=self.visuals_with_include_1d,
                auto_labels=AutoLabels(
                    title="Chi-Squared Map", filename="chi_squared_map"
                ),
            )

        if pre_cti_data:

            self.mat_plot_1d.plot_yx(
                y=self.fit.pre_cti_data,
                x=np.arange(len(self.fit.pre_cti_data)),
                visuals_1d=self.visuals_with_include_1d,
                auto_labels=AutoLabels(
                    title="CI Pre CTI Image", filename="pre_cti_data"
                ),
            )

        if post_cti_data:

            self.mat_plot_1d.plot_yx(
                y=self.fit.post_cti_data,
                x=np.arange(len(self.fit.post_cti_data)),
                visuals_1d=self.visuals_with_include_1d,
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
        return self.subplot(
            data=True,
            signal_to_noise_map=True,
            pre_cti_data=True,
            post_cti_data=True,
            normalized_residual_map=True,
            chi_squared_map=True,
        )
