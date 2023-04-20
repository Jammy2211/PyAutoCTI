import logging

import autocti.plot as aplt

from autocti.model.visualizer import Visualizer
from autocti.model.visualizer import plot_setting
from autocti import exc

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class VisualizerDataset1D(Visualizer):
    def visualize_dataset_1d(self, dataset_1d, folder_suffix=""):
        def should_plot(name):
            return plot_setting(section="dataset", name=name)

        mat_plot_1d = self.mat_plot_1d_from(subfolders=f"dataset_1d{folder_suffix}")

        imaging_ci_plotter = aplt.Dataset1DPlotter(
            dataset=dataset_1d, mat_plot_1d=mat_plot_1d, include_1d=self.include_1d
        )

        if should_plot("subplot_dataset"):

            imaging_ci_plotter.subplot_dataset_1d()

        imaging_ci_plotter.figures_1d(
            data=should_plot("data"),
            data_logy=should_plot("data_logy"),
            noise_map=should_plot("noise_map"),
            signal_to_noise_map=should_plot("signal_to_noise_map"),
            pre_cti_data=should_plot("pre_cti_data"),
        )

    def visualize_dataset_1d_regions(self, dataset_1d, region_list, folder_suffix=""):
        def should_plot(name):
            return plot_setting(section="dataset", name=name)

        mat_plot_1d = self.mat_plot_1d_from(subfolders=f"dataset_1d{folder_suffix}")

        imaging_ci_plotter = aplt.Dataset1DPlotter(
            dataset=dataset_1d, mat_plot_1d=mat_plot_1d, include_1d=self.include_1d
        )

        for region in region_list:

            try:

                if should_plot("subplot_dataset"):
                    imaging_ci_plotter.subplot_dataset_1d(region=region)

                imaging_ci_plotter.figures_1d(
                    region=region,
                    data=should_plot("data"),
                    data_logy=should_plot("data_logy"),
                    noise_map=should_plot("noise_map"),
                    signal_to_noise_map=should_plot("signal_to_noise_map"),
                    pre_cti_data=should_plot("pre_cti_data"),
                )

            except (exc.RegionException, TypeError, ValueError):

                logger.info(
                    f"VISUALIZATION - Could not visualize the Dataset1D {region}"
                )

    def visualize_fit_1d(self, fit, during_analysis, folder_suffix=""):
        def should_plot(name):
            return plot_setting(section="fit", name=name)

        mat_plot_1d = self.mat_plot_1d_from(subfolders=f"fit_dataset_1d{folder_suffix}")

        fit_1d_plotter = aplt.FitDataset1DPlotter(
            fit=fit, mat_plot_1d=mat_plot_1d, include_1d=self.include_1d
        )

        fit_1d_plotter.figures_1d(
            data=should_plot("data"),
            data_logy=should_plot("data_logy"),
            noise_map=should_plot("noise_map"),
            signal_to_noise_map=should_plot("signal_to_noise_map"),
            pre_cti_data=should_plot("pre_cti_data"),
            post_cti_data=should_plot("post_cti_data"),
            residual_map=should_plot("residual_map"),
            residual_map_logy=should_plot("residual_map_logy"),
            normalized_residual_map=should_plot("normalized_residual_map"),
            chi_squared_map=should_plot("chi_squared_map"),
        )

        if not during_analysis:

            if should_plot("all_at_end_png"):

                fit_1d_plotter.figures_1d(
                    data=True,
                    noise_map=True,
                    signal_to_noise_map=True,
                    pre_cti_data=True,
                    post_cti_data=True,
                    residual_map=True,
                    residual_map_logy=True,
                    normalized_residual_map=True,
                    chi_squared_map=True,
                )

        if should_plot("subplot_fit"):
            fit_1d_plotter.subplot_fit_dataset_1d()

    def visualize_fit_1d_regions(
        self, fit, region_list, during_analysis, folder_suffix=""
    ):
        def should_plot(name):
            return plot_setting(section="fit", name=name)

        mat_plot_1d = self.mat_plot_1d_from(subfolders=f"fit_dataset_1d{folder_suffix}")

        fit_1d_plotter = aplt.FitDataset1DPlotter(
            fit=fit, mat_plot_1d=mat_plot_1d, include_1d=self.include_1d
        )

        for region in region_list:

            try:

                if should_plot("subplot_fit"):
                    fit_1d_plotter.subplot_fit_dataset_1d(region=region)

                fit_1d_plotter.figures_1d(
                    region=region,
                    data=should_plot("data"),
                    data_logy=should_plot("data_logy"),
                    noise_map=should_plot("noise_map"),
                    signal_to_noise_map=should_plot("signal_to_noise_map"),
                    pre_cti_data=should_plot("pre_cti_data"),
                    post_cti_data=should_plot("post_cti_data"),
                    residual_map=should_plot("residual_map"),
                    residual_map_logy=should_plot("residual_map_logy"),
                    normalized_residual_map=should_plot("normalized_residual_map"),
                    chi_squared_map=should_plot("chi_squared_map"),
                )

            except (exc.RegionException, TypeError, ValueError):

                logger.info(
                    f"VISUALIZATION - Could not visualize the Dataset1D {region}"
                )

    def visualize_fit_1d_combined(self, fit_list, during_analysis, folder_suffix=""):
        def should_plot(name):
            return plot_setting(section="fit", name=name)

        mat_plot_1d = self.mat_plot_1d_from(
            subfolders=f"fit_dataset_1d_combined{folder_suffix}"
        )

        fit_1d_plotter_list = [
            aplt.FitDataset1DPlotter(
                fit=fit, mat_plot_1d=mat_plot_1d, include_1d=self.include_1d
            )
            for fit in fit_list
        ]
        multi_plotter = aplt.MultiFigurePlotter(plotter_list=fit_1d_plotter_list)

        if should_plot("data"):
            multi_plotter.subplot_of_figure(func_name="figures_1d", figure_name="data")

        if should_plot("data_logy"):
            multi_plotter.subplot_of_figure(
                func_name="figures_1d", figure_name="data_logy"
            )

        if should_plot("residual_map"):
            multi_plotter.subplot_of_figure(
                func_name="figures_1d", figure_name="residual_map"
            )

        if should_plot("residual_map_logy"):
            multi_plotter.subplot_of_figure(
                func_name="figures_1d", figure_name="residual_map_logy"
            )

        if should_plot("normalized_residual_map"):
            multi_plotter.subplot_of_figure(
                func_name="figures_1d", figure_name="normalized_residual_map"
            )

        if should_plot("chi_squared_map"):
            multi_plotter.subplot_of_figure(
                func_name="figures_1d", figure_name="chi_squared_map"
            )

    def visualize_fit_1d_region_combined(
        self, fit_list, region_list, during_analysis, folder_suffix=""
    ):
        def should_plot(name):
            return plot_setting(section="fit", name=name)

        mat_plot_1d = self.mat_plot_1d_from(
            subfolders=f"fit_dataset_1d_combined{folder_suffix}"
        )

        fit_1d_plotter_list = [
            aplt.FitDataset1DPlotter(
                fit=fit, mat_plot_1d=mat_plot_1d, include_1d=self.include_1d
            )
            for fit in fit_list
        ]
        multi_plotter = aplt.MultiFigurePlotter(plotter_list=fit_1d_plotter_list)

        for region in region_list:

            try:

                if should_plot("data"):
                    multi_plotter.subplot_of_figure(
                        func_name="figures_1d",
                        figure_name="data",
                        region=region,
                        filename_suffix=f"_{region}",
                    )

                if should_plot("data_logy"):
                    multi_plotter.subplot_of_figure(
                        func_name="figures_1d",
                        figure_name="data_logy",
                        region=region,
                        filename_suffix=f"_{region}",
                    )

                if should_plot("residual_map"):
                    multi_plotter.subplot_of_figure(
                        func_name="figures_1d",
                        figure_name="residual_map",
                        region=region,
                        filename_suffix=f"_{region}",
                    )

                if should_plot("residual_map_logy"):
                    multi_plotter.subplot_of_figure(
                        func_name="figures_1d",
                        figure_name="residual_map_logy",
                        region=region,
                        filename_suffix=f"_{region}",
                    )

                if should_plot("normalized_residual_map"):
                    multi_plotter.subplot_of_figure(
                        func_name="figures_1d",
                        figure_name="normalized_residual_map",
                        region=region,
                        filename_suffix=f"_{region}",
                    )

                if should_plot("chi_squared_map"):
                    multi_plotter.subplot_of_figure(
                        func_name="figures_1d",
                        figure_name="chi_squared_map",
                        region=region,
                        filename_suffix=f"_{region}",
                    )

            except (exc.RegionException, TypeError, ValueError):

                logger.info(
                    f"VISUALIZATION - Could not visualize the Dataset1D {region}"
                )
