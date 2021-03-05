from os import path
from autoconf import conf
from autoarray.plot.mat_wrap.wrap import wrap_base
from autoarray.plot.mat_wrap import mat_plot
from autoarray.plot.mat_wrap import include as inc
from autocti.plot import MultiPlotter
from autocti.plot import ci_imaging_plotters, ci_fit_plotters


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

    def visualize_ci_imaging(self, ci_imaging, index=0):
        def should_plot(name):
            return plot_setting(section="dataset", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders=f"ci_imaging_{index}")

        ci_imaging_plotter = ci_imaging_plotters.CIImagingPlotter(
            imaging=ci_imaging, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        ci_imaging_plotter.figures(
            image=should_plot("data"),
            noise_map=should_plot("noise_map"),
            inverse_noise_map=should_plot("inverse_noise_map"),
            signal_to_noise_map=should_plot("signal_to_noise_map"),
            absolute_signal_to_noise_map=should_plot("absolute_signal_to_noise_map"),
            potential_chi_squared_map=should_plot("potential_chi_squared_map"),
            ci_pre_cti=should_plot("ci_pre_cti"),
            cosmic_ray_map=should_plot("cosmic_ray_map"),
        )

        if should_plot("subplot_dataset"):

            ci_imaging_plotter.subplot_ci_imaging()

    def visualize_ci_imaging_lines(self, ci_imaging, line_region, index=0):
        def should_plot(name):
            return plot_setting(section="dataset", name=name)

        mat_plot_1d = self.mat_plot_1d_from(subfolders=f"ci_imaging_{index}")

        ci_imaging_plotter = ci_imaging_plotters.CIImagingPlotter(
            imaging=ci_imaging, mat_plot_1d=mat_plot_1d, include_2d=self.include_2d
        )

        if should_plot("subplot_dataset"):

            ci_imaging_plotter.subplot_1d_ci_line_region(line_region=line_region)

        ci_imaging_plotter.figures_1d_ci_line_region(
            line_region=line_region,
            image=should_plot("data"),
            noise_map=should_plot("noise_map"),
            signal_to_noise_map=should_plot("signal_to_noise_map"),
            ci_pre_cti=should_plot("ci_pre_cti"),
        )

    def visualize_ci_fit(self, fit, during_analysis, index=0):
        def should_plot(name):
            return plot_setting(section="fit", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders=f"fit_ci_imaging_{index}")

        ci_fit_plotter = ci_fit_plotters.CIFitPlotter(
            fit=fit, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        ci_fit_plotter.figures(
            image=should_plot("data"),
            noise_map=should_plot("noise_map"),
            signal_to_noise_map=should_plot("signal_to_noise_map"),
            ci_pre_cti=should_plot("ci_pre_cti"),
            ci_post_cti=should_plot("ci_post_cti"),
            residual_map=should_plot("residual_map"),
            normalized_residual_map=should_plot("normalized_residual_map"),
            chi_squared_map=should_plot("chi_squared_map"),
        )

        if not during_analysis:

            if should_plot("all_at_end_png"):

                ci_fit_plotter.figures(
                    image=True,
                    noise_map=True,
                    signal_to_noise_map=True,
                    ci_pre_cti=True,
                    ci_post_cti=True,
                    residual_map=True,
                    normalized_residual_map=True,
                    chi_squared_map=True,
                )

            if should_plot("all_at_end_fits"):

                self.visualize_fit_in_fits(fit=fit)

        if should_plot("subplot_fit"):
            ci_fit_plotter.subplot_ci_fit()

    def visualize_ci_fit_1d_lines(self, fit, line_region, during_analysis, index=0):
        def should_plot(name):
            return plot_setting(section="fit", name=name)

        mat_plot_1d = self.mat_plot_1d_from(subfolders=f"fit_ci_imaging_{index}")

        ci_fit_plotter = ci_fit_plotters.CIFitPlotter(
            fit=fit, mat_plot_1d=mat_plot_1d, include_1d=self.include_1d
        )

        if should_plot("subplot_fit"):

            ci_fit_plotter.subplot_1d_ci_line_region(line_region=line_region)

        ci_fit_plotter.figures_1d_ci_line_region(
            line_region=line_region,
            image=should_plot("data"),
            noise_map=should_plot("noise_map"),
            signal_to_noise_map=should_plot("signal_to_noise_map"),
            ci_pre_cti=should_plot("ci_pre_cti"),
            ci_post_cti=should_plot("ci_post_cti"),
            residual_map=should_plot("residual_map"),
            normalized_residual_map=should_plot("normalized_residual_map"),
            chi_squared_map=should_plot("chi_squared_map"),
        )

        if not during_analysis:

            if should_plot("all_at_end_png"):

                ci_fit_plotter.figures_1d_ci_line_region(
                    line_region=line_region,
                    image=True,
                    noise_map=True,
                    signal_to_noise_map=True,
                    ci_pre_cti=True,
                    ci_post_cti=True,
                    residual_map=True,
                    normalized_residual_map=True,
                    chi_squared_map=True,
                )

    def visualize_multiple_ci_fits_subplots(self, fits):
        def should_plot(name):
            return plot_setting(section="fit", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="multiple_ci_fits")

        ci_fit_plotter_list = [
            ci_fit_plotters.CIFitPlotter(fit=ci_fit, mat_plot_2d=mat_plot_2d)
            for ci_fit in fits
        ]
        multi_plotter = MultiPlotter(plotter_list=ci_fit_plotter_list)

        if should_plot("subplot_residual_maps"):
            multi_plotter.subplot_of_figure(
                func_name="figures", figure_name="residual_map"
            )

        if should_plot("subplot_normalized_residual_maps"):
            multi_plotter.subplot_of_figure(
                func_name="figures", figure_name="normalized_residual_map"
            )

        if should_plot("subplot_chi_squared_maps"):
            multi_plotter.subplot_of_figure(
                func_name="figures", figure_name="chi_squared_map"
            )

    def visualize_multiple_ci_fits_subplots_1d_lines(self, fits, line_region):
        def should_plot(name):
            return plot_setting(section="fit", name=name)

        mat_plot_1d = self.mat_plot_1d_from(
            subfolders=f"multiple_ci_fits_1d_line_{line_region}"
        )

        ci_fit_plotter_list = [
            ci_fit_plotters.CIFitPlotter(fit=ci_fit, mat_plot_1d=mat_plot_1d)
            for ci_fit in fits
        ]
        multi_plotter = MultiPlotter(plotter_list=ci_fit_plotter_list)

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

    def visualize_fit_in_fits(self, fit):

        mat_plot_2d = self.mat_plot_2d_from(
            subfolders="fit_imaging/fits", format="fits"
        )

        ci_fit_plotter = ci_fit_plotters.CIFitPlotter(
            fit=fit, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        ci_fit_plotter.figures(
            image=True,
            noise_map=True,
            signal_to_noise_map=True,
            ci_pre_cti=True,
            ci_post_cti=True,
            residual_map=True,
            normalized_residual_map=True,
            chi_squared_map=True,
        )
