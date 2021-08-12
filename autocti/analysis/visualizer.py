from os import path
from autoconf import conf
from autoarray.plot.mat_wrap.wrap import wrap_base
from autoarray.plot.mat_wrap import mat_plot
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot import multi_plotters
from autocti.plot import dataset_line_plotters
from autocti.plot import fit_line_plotters
from autocti.plot import imaging_ci_plotters
from autocti.plot import fit_ci_plotters


def setting(section, name):
    return conf.instance["visualize"]["plots"][section][name]


def plot_setting(section, name):
    return setting(section, name)


class Visualizer:
    def __init__(self, visualize_path):

        self.visualize_path = visualize_path

        self.include_1d = inc.Include1D()
        self.include_2d = inc.Include2D()

    def mat_plot_1d_from(self, subfolders, format="png"):
        return mat_plot.MatPlot1D(
            output=wrap_base.Output(
                path=path.join(self.visualize_path, subfolders), format=format
            )
        )

    def mat_plot_2d_from(self, subfolders, format="png"):
        return mat_plot.MatPlot2D(
            output=wrap_base.Output(
                path=path.join(self.visualize_path, subfolders), format=format
            )
        )

    def visualize_dataset_line(self, dataset_line):
        def should_plot(name):
            return plot_setting(section="dataset", name=name)

        mat_plot_1d = self.mat_plot_1d_from(subfolders=f"dataset_line")

        imaging_ci_plotter = dataset_line_plotters.DatasetLinePlotter(
            dataset_line=dataset_line,
            mat_plot_1d=mat_plot_1d,
            include_1d=self.include_1d,
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

        fit_line_plotter = fit_line_plotters.FitDatasetLinePlotter(
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

            if should_plot("all_at_end_fits"):

                self.visualize_fit_in_fit(fit=fit)

        if should_plot("subplot_fit"):
            fit_line_plotter.subplot_fit_dataset_line()

    def visualize_imaging_ci(self, imaging_ci):
        def should_plot(name):
            return plot_setting(section="dataset", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders=f"imaging_ci")

        imaging_ci_plotter = imaging_ci_plotters.ImagingCIPlotter(
            imaging=imaging_ci, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        imaging_ci_plotter.figures_2d(
            image=should_plot("data"),
            noise_map=should_plot("noise_map"),
            inverse_noise_map=should_plot("inverse_noise_map"),
            signal_to_noise_map=should_plot("signal_to_noise_map"),
            absolute_signal_to_noise_map=should_plot("absolute_signal_to_noise_map"),
            potential_chi_squared_map=should_plot("potential_chi_squared_map"),
            pre_cti_data=should_plot("pre_cti_data"),
            cosmic_ray_map=should_plot("cosmic_ray_map"),
        )

        if should_plot("subplot_dataset"):

            imaging_ci_plotter.subplot_imaging_ci()

    def visualize_imaging_ci_lines(self, imaging_ci, line_region):
        def should_plot(name):
            return plot_setting(section="dataset", name=name)

        mat_plot_1d = self.mat_plot_1d_from(subfolders=f"imaging_ci")

        imaging_ci_plotter = imaging_ci_plotters.ImagingCIPlotter(
            imaging=imaging_ci, mat_plot_1d=mat_plot_1d, include_2d=self.include_2d
        )

        if should_plot("subplot_dataset"):

            imaging_ci_plotter.subplot_1d_ci_line_region(line_region=line_region)

        imaging_ci_plotter.figures_1d_ci_line_region(
            line_region=line_region,
            image=should_plot("data"),
            noise_map=should_plot("noise_map"),
            signal_to_noise_map=should_plot("signal_to_noise_map"),
            pre_cti_data=should_plot("pre_cti_data"),
        )

    def visualize_fit_ci(self, fit, during_analysis):
        def should_plot(name):
            return plot_setting(section="fit", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders=f"fit_imaging_ci")

        fit_ci_plotter = fit_ci_plotters.FitImagingCIPlotter(
            fit=fit, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        fit_ci_plotter.figures_2d(
            image=should_plot("data"),
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

                fit_ci_plotter.figures_2d(
                    image=True,
                    noise_map=True,
                    signal_to_noise_map=True,
                    pre_cti_data=True,
                    post_cti_data=True,
                    residual_map=True,
                    normalized_residual_map=True,
                    chi_squared_map=True,
                )

            if should_plot("all_at_end_fits"):

                self.visualize_fit_in_fit(fit=fit)

        if should_plot("subplot_fit"):
            fit_ci_plotter.subplot_fit_ci()

    def visualize_fit_ci_1d_lines(self, fit, line_region, during_analysis):
        def should_plot(name):
            return plot_setting(section="fit", name=name)

        mat_plot_1d = self.mat_plot_1d_from(subfolders=f"fit_imaging_ci")

        fit_ci_plotter = fit_ci_plotters.FitImagingCIPlotter(
            fit=fit, mat_plot_1d=mat_plot_1d, include_1d=self.include_1d
        )

        if should_plot("subplot_fit"):

            fit_ci_plotter.subplot_1d_ci_line_region(line_region=line_region)

        fit_ci_plotter.figures_1d_ci_line_region(
            line_region=line_region,
            image=should_plot("data"),
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

                fit_ci_plotter.figures_1d_ci_line_region(
                    line_region=line_region,
                    image=True,
                    noise_map=True,
                    signal_to_noise_map=True,
                    pre_cti_data=True,
                    post_cti_data=True,
                    residual_map=True,
                    normalized_residual_map=True,
                    chi_squared_map=True,
                )

    def visualize_multiple_fit_cis_subplots(self, fit):
        def should_plot(name):
            return plot_setting(section="fit", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="multiple_fit_cis")

        fit_ci_plotter_list = [
            fit_ci_plotters.FitImagingCIPlotter(fit=fit_ci, mat_plot_2d=mat_plot_2d)
            for fit_ci in fit
        ]
        multi_plotter = multi_plotters.MultiFigurePlotter(
            plotter_list=fit_ci_plotter_list
        )

        if should_plot("subplot_residual_maps"):
            multi_plotter.subplot_of_figure(
                func_name="figures_2d", figure_name="residual_map"
            )

        if should_plot("subplot_normalized_residual_maps"):
            multi_plotter.subplot_of_figure(
                func_name="figures_2d", figure_name="normalized_residual_map"
            )

        if should_plot("subplot_chi_squared_maps"):
            multi_plotter.subplot_of_figure(
                func_name="figures_2d", figure_name="chi_squared_map"
            )

    def visualize_multiple_fit_cis_subplots_1d_lines(self, fit, line_region):
        def should_plot(name):
            return plot_setting(section="fit", name=name)

        mat_plot_1d = self.mat_plot_1d_from(
            subfolders=f"multiple_fit_cis_1d_line_{line_region}"
        )

        fit_ci_plotter_list = [
            fit_ci_plotters.FitImagingCIPlotter(fit=fit_ci, mat_plot_1d=mat_plot_1d)
            for fit_ci in fit
        ]
        multi_plotter = multi_plotters.MultiFigurePlotter(
            plotter_list=fit_ci_plotter_list
        )

        if should_plot("subplot_residual_maps"):
            multi_plotter.subplot_of_figure(
                func_name="figures_1d_ci_line_region",
                figure_name="residual_map",
                line_region=line_region,
            )

        if should_plot("subplot_normalized_residual_maps"):
            multi_plotter.subplot_of_figure(
                func_name="figures_1d_ci_line_region",
                figure_name="normalized_residual_map",
                line_region=line_region,
            )

        if should_plot("subplot_chi_squared_maps"):
            multi_plotter.subplot_of_figure(
                func_name="figures_1d_ci_line_region",
                figure_name="chi_squared_map",
                line_region=line_region,
            )

    def visualize_fit_in_fit(self, fit):

        mat_plot_2d = self.mat_plot_2d_from(subfolders="fit_imaging/fit", format="fit")

        fit_ci_plotter = fit_ci_plotters.FitImagingCIPlotter(
            fit=fit, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        fit_ci_plotter.figures_2d(
            image=True,
            noise_map=True,
            signal_to_noise_map=True,
            pre_cti_data=True,
            post_cti_data=True,
            residual_map=True,
            normalized_residual_map=True,
            chi_squared_map=True,
        )
