import logging

from autoarray.plot.multi_plotters import MultiFigurePlotter

import autocti.plot as aplt

from autocti.model.visualizer import Visualizer
from autocti.model.visualizer import plot_setting

from autocti import exc

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class VisualizerImagingCI(Visualizer):
    def visualize_imaging_ci(self, imaging_ci, folder_suffix: str = ""):
        def should_plot(name):
            return plot_setting(section="dataset", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders=f"imaging_ci{folder_suffix}")

        imaging_ci_plotter = aplt.ImagingCIPlotter(
            dataset=imaging_ci, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
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

    def visualize_imaging_ci_regions(
        self, imaging_ci, region_list, folder_suffix: str = ""
    ):
        def should_plot(name):
            return plot_setting(section="dataset", name=name)

        mat_plot_1d = self.mat_plot_1d_from(subfolders=f"imaging_ci{folder_suffix}")

        imaging_ci_plotter = aplt.ImagingCIPlotter(
            dataset=imaging_ci, mat_plot_1d=mat_plot_1d, include_2d=self.include_2d
        )

        for region in region_list:

            try:

                if should_plot("subplot_dataset"):

                    imaging_ci_plotter.subplot_1d_of_region(region=region)

                imaging_ci_plotter.figures_1d_of_region(
                    region=region,
                    image=should_plot("data"),
                    noise_map=should_plot("noise_map"),
                    signal_to_noise_map=should_plot("signal_to_noise_map"),
                    pre_cti_data=should_plot("pre_cti_data"),
                    data_with_noise_map=should_plot("data_with_noise_map"),
                    data_with_noise_map_logy=should_plot("data_with_noise_map_logy"),
                )

            except (exc.RegionException, TypeError, ValueError):

                logger.info(
                    f"VISUALIZATION - Could not visualize the ImagingCI 1D {region}"
                )

    def visualize_fit_ci(self, fit, during_analysis, folder_suffix: str = ""):
        def should_plot(name):
            return plot_setting(section="fit", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders=f"fit_imaging_ci{folder_suffix}")

        fit_ci_plotter = aplt.FitImagingCIPlotter(
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

                self.visualize_fit_in_fits(fit=fit)

        if should_plot("subplot_fit"):
            fit_ci_plotter.subplot_fit_ci()

    def visualize_fit_ci_1d_regions(
        self, fit, region_list, during_analysis, folder_suffix: str = ""
    ):
        def should_plot(name):
            return plot_setting(section="fit", name=name)

        mat_plot_1d = self.mat_plot_1d_from(subfolders=f"fit_imaging_ci{folder_suffix}")

        fit_ci_plotter = aplt.FitImagingCIPlotter(
            fit=fit, mat_plot_1d=mat_plot_1d, include_1d=self.include_1d
        )

        for region in region_list:

            try:

                if should_plot("subplot_fit"):

                    fit_ci_plotter.subplot_1d_of_region(region=region)

                fit_ci_plotter.figures_1d_of_region(
                    region=region,
                    image=should_plot("data"),
                    noise_map=should_plot("noise_map"),
                    signal_to_noise_map=should_plot("signal_to_noise_map"),
                    pre_cti_data=should_plot("pre_cti_data"),
                    post_cti_data=should_plot("post_cti_data"),
                    residual_map=should_plot("residual_map"),
                    normalized_residual_map=should_plot("normalized_residual_map"),
                    chi_squared_map=should_plot("chi_squared_map"),
                    data_with_noise_map_model=should_plot("data_with_noise_map_model"),
                    data_with_noise_map_model_logy=should_plot(
                        "data_with_noise_map_model_logy"
                    ),
                )

                if not during_analysis:

                    if should_plot("all_at_end_png"):

                        fit_ci_plotter.figures_1d_of_region(
                            region=region,
                            image=True,
                            noise_map=True,
                            signal_to_noise_map=True,
                            pre_cti_data=True,
                            post_cti_data=True,
                            residual_map=True,
                            normalized_residual_map=True,
                            chi_squared_map=True,
                        )

            except (exc.RegionException, TypeError, ValueError):

                logger.info(
                    f"VISUALIZATION - Could not visualize the ImagingCI 1D {region}"
                )

    def visualize_multiple_fit_cis_subplots(self, fit):
        def should_plot(name):
            return plot_setting(section="fit", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="multiple_fit_cis")

        fit_ci_plotter_list = [
            aplt.FitImagingCIPlotter(fit=fit, mat_plot_2d=mat_plot_2d) for fit in fit
        ]
        multi_plotter = MultiFigurePlotter(plotter_list=fit_ci_plotter_list)

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

    def visualize_multiple_fit_cis_subplots_1d_lines(self, fit, region):
        def should_plot(name):
            return plot_setting(section="fit", name=name)

        mat_plot_1d = self.mat_plot_1d_from(
            subfolders=f"multiple_fit_cis_1d_line_{region}"
        )

        fit_ci_plotter_list = [
            aplt.FitImagingCIPlotter(fit=fit, mat_plot_1d=mat_plot_1d) for fit in fit
        ]
        multi_plotter = MultiFigurePlotter(plotter_list=fit_ci_plotter_list)

        if should_plot("subplot_residual_maps"):
            multi_plotter.subplot_of_figure(
                func_name="figures_1d_of_region",
                figure_name="residual_map",
                region=region,
            )

        if should_plot("subplot_normalized_residual_maps"):
            multi_plotter.subplot_of_figure(
                func_name="figures_1d_of_region",
                figure_name="normalized_residual_map",
                region=region,
            )

        if should_plot("subplot_chi_squared_maps"):
            multi_plotter.subplot_of_figure(
                func_name="figures_1d_of_region",
                figure_name="chi_squared_map",
                region=region,
            )

    def visualize_fit_in_fits(self, fit):

        mat_plot_2d = self.mat_plot_2d_from(subfolders="fit_imaging/fit", format="fit")

        fit_ci_plotter = aplt.FitImagingCIPlotter(
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
