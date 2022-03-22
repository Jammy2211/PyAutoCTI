import autocti.plot as aplt

from autocti.model.visualizer import Visualizer
from autocti.model.visualizer import plot_setting


class VisualizerDataset1D(Visualizer):
    def visualize_dataset_line(self, dataset_line):
        def should_plot(name):
            return plot_setting(section="dataset", name=name)

        mat_plot_1d = self.mat_plot_1d_from(subfolders=f"dataset_line")

        imaging_ci_plotter = aplt.Dataset1DPlotter(
            dataset=dataset_line, mat_plot_1d=mat_plot_1d, include_1d=self.include_1d
        )

        imaging_ci_plotter.figures_1d(
            data=should_plot("data"),
            noise_map=should_plot("noise_map"),
            signal_to_noise_map=should_plot("signal_to_noise_map"),
            pre_cti_data=should_plot("pre_cti_data"),
        )

        if should_plot("subplot_dataset"):

            imaging_ci_plotter.subplot_dataset_line()

    def visualize_fit_line(self, fit, during_analysis):
        def should_plot(name):
            return plot_setting(section="fit", name=name)

        mat_plot_1d = self.mat_plot_1d_from(subfolders=f"fit_dataset_line")

        fit_line_plotter = aplt.FitDataset1DPlotter(
            fit=fit, mat_plot_1d=mat_plot_1d, include_1d=self.include_1d
        )

        fit_line_plotter.figures_1d(
            data=should_plot("data"),
            noise_map=should_plot("noise_map"),
            signal_to_noise_map=should_plot("signal_to_noise_map"),
            pre_cti_data=should_plot("pre_cti_data"),
            post_cti_data=should_plot("post_cti_data"),
            residual_map=should_plot("residual_map"),
            normalized_residual_map=should_plot("normalized_residual_map"),
            chi_squared_map=should_plot("chi_squared_map"),
        )

        if not during_analysis:

            if should_plot("all_at_end_png"):

                fit_line_plotter.figures_1d(
                    data=True,
                    noise_map=True,
                    signal_to_noise_map=True,
                    pre_cti_data=True,
                    post_cti_data=True,
                    residual_map=True,
                    normalized_residual_map=True,
                    chi_squared_map=True,
                )

        if should_plot("subplot_fit"):
            fit_line_plotter.subplot_fit_dataset_line()
