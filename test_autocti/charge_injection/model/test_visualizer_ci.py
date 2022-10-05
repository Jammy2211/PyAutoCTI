import os
import shutil
from os import path

import pytest
from autocti.charge_injection.model.visualizer import VisualizerImagingCI
from autoconf import conf

directory = path.dirname(path.abspath(__file__))


@pytest.fixture(name="plot_path")
def make_visualizer_plotter_setup():
    return path.join("{}".format(directory), "files")


class TestVisualizerImagingCI:
    def test__visualizes_imaging_ci_using_configs(
        self, imaging_ci_7x7, plot_path, plot_patch
    ):

        if path.exists(plot_path):
            shutil.rmtree(plot_path)

        imaging_ci_7x7.cosmic_ray_map[0, 0] = 1

        visualizer = VisualizerImagingCI(visualize_path=plot_path)

        visualizer.visualize_imaging_ci(imaging_ci=imaging_ci_7x7)

        plot_path = path.join(plot_path, "imaging_ci")

        assert path.join(plot_path, "subplot_imaging_ci.png") in plot_patch.paths
        assert path.join(plot_path, "image_2d.png") in plot_patch.paths
        assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
        assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths
        assert path.join(plot_path, "pre_cti_data.png") in plot_patch.paths
        assert path.join(plot_path, "cosmic_ray_map.png") in plot_patch.paths

    def test__visualizes_imaging_ci_lines_using_configs(
        self, imaging_ci_7x7, plot_path, plot_patch
    ):

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = VisualizerImagingCI(visualize_path=plot_path)

        visualizer.visualize_imaging_ci_lines(
            imaging_ci=imaging_ci_7x7, region="parallel_fpr"
        )

        plot_path = path.join(plot_path, "imaging_ci")

        assert (
            path.join(plot_path, "subplot_1d_ci_parallel_fpr.png") in plot_patch.paths
        )
        assert path.join(plot_path, "image_parallel_fpr.png") in plot_patch.paths
        assert (
            path.join(plot_path, "noise_map_parallel_fpr.png") not in plot_patch.paths
        )
        assert (
            path.join(plot_path, "signal_to_noise_map_parallel_fpr.png")
            not in plot_patch.paths
        )
        assert path.join(plot_path, "pre_cti_data_parallel_fpr.png") in plot_patch.paths

    def test___visualizes_fit_ci_using_configs(self, fit_ci_7x7, plot_path, plot_patch):

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = VisualizerImagingCI(visualize_path=plot_path)

        visualizer.visualize_fit_ci(fit=fit_ci_7x7, during_analysis=True)

        plot_path = path.join(plot_path, "fit_imaging_ci")

        assert path.join(plot_path, "subplot_fit_ci.png") in plot_patch.paths
        assert path.join(plot_path, "image_2d.png") in plot_patch.paths
        assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
        assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths
        assert path.join(plot_path, "pre_cti_data.png") in plot_patch.paths
        assert path.join(plot_path, "post_cti_data.png") in plot_patch.paths
        assert path.join(plot_path, "residual_map.png") not in plot_patch.paths
        assert path.join(plot_path, "normalized_residual_map.png") in plot_patch.paths
        assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths

        plot_patch.paths = []

        visualizer.visualize_fit_ci(fit=fit_ci_7x7, during_analysis=False)

        assert path.join(plot_path, "subplot_fit_ci.png") in plot_patch.paths
        assert path.join(plot_path, "image_2d.png") in plot_patch.paths
        assert path.join(plot_path, "noise_map.png") in plot_patch.paths
        assert path.join(plot_path, "signal_to_noise_map.png") in plot_patch.paths
        assert path.join(plot_path, "pre_cti_data.png") in plot_patch.paths
        assert path.join(plot_path, "post_cti_data.png") in plot_patch.paths
        assert path.join(plot_path, "residual_map.png") in plot_patch.paths
        assert path.join(plot_path, "normalized_residual_map.png") in plot_patch.paths
        assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths

    def test___visualizes_fit_ci_lines_using_configs(
        self, fit_ci_7x7, plot_path, plot_patch
    ):

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = VisualizerImagingCI(visualize_path=plot_path)

        visualizer.visualize_fit_ci_1d_lines(
            fit=fit_ci_7x7, region="parallel_fpr", during_analysis=True
        )

        plot_path = path.join(plot_path, "fit_imaging_ci")

        assert (
            path.join(plot_path, "subplot_1d_fit_ci_parallel_fpr.png")
            in plot_patch.paths
        )
        assert path.join(plot_path, "image_parallel_fpr.png") in plot_patch.paths
        assert path.join(plot_path, "noise_parallel_fpr.png") not in plot_patch.paths
        assert (
            path.join(plot_path, "signal_to_noise_map_parallel_fpr.png")
            not in plot_patch.paths
        )
        assert path.join(plot_path, "pre_cti_data_parallel_fpr.png") in plot_patch.paths
        assert (
            path.join(plot_path, "post_cti_data_parallel_fpr.png") in plot_patch.paths
        )
        assert (
            path.join(plot_path, "residual_map_parallel_fpr.png")
            not in plot_patch.paths
        )
        assert (
            path.join(plot_path, "normalized_residual_map_parallel_fpr.png")
            in plot_patch.paths
        )
        assert (
            path.join(plot_path, "chi_squared_map_parallel_fpr.png") in plot_patch.paths
        )

        plot_patch.paths = []

        visualizer.visualize_fit_ci_1d_lines(
            fit=fit_ci_7x7, region="parallel_fpr", during_analysis=False
        )

        assert (
            path.join(plot_path, "subplot_1d_fit_ci_parallel_fpr.png")
            in plot_patch.paths
        )
        assert path.join(plot_path, "image_parallel_fpr.png") in plot_patch.paths
        assert path.join(plot_path, "noise_map_parallel_fpr.png") in plot_patch.paths
        assert (
            path.join(plot_path, "signal_to_noise_map_parallel_fpr.png")
            in plot_patch.paths
        )
        assert path.join(plot_path, "pre_cti_data_parallel_fpr.png") in plot_patch.paths
        assert (
            path.join(plot_path, "post_cti_data_parallel_fpr.png") in plot_patch.paths
        )
        assert path.join(plot_path, "residual_map_parallel_fpr.png") in plot_patch.paths
        assert (
            path.join(plot_path, "normalized_residual_map_parallel_fpr.png")
            in plot_patch.paths
        )
        assert (
            path.join(plot_path, "chi_squared_map_parallel_fpr.png") in plot_patch.paths
        )

    # def test__visualize_multiple_fit_cis_subplots(
    #     self, imaging_ci_7x7, fit_ci_7x7, plot_path, plot_patch
    # ):
    #
    #     if os.path.exists(plot_path):
    #         shutil.rmtree(plot_path)
    #
    #     visualizer = VisualizerImagingCI(visualize_path=plot_path)
    #
    #     visualizer.visualize_multiple_fit_cis_subplots(fits=[fit_ci_7x7, fit_ci_7x7])
    #
    #     plot_path = path.join(plot_path, "multiple_fit_cis")
    #
    #     assert path.join(plot_path, "subplot_residual_map_list.png") in plot_patch.paths
    #     assert (
    #         path.join(plot_path, "subplot_normalized_residual_map_list.png")
    #         in plot_patch.paths
    #     )
    #     assert (
    #         path.join(plot_path, "subplot_chi_squared_map_list.png")
    #         not in plot_patch.paths
    #     )
    #
    # def test__visualize_multiple_fit_cis_1d_line_subplots(
    #     self, imaging_ci_7x7, fit_ci_7x7, plot_path, plot_patch
    # ):
    #
    #     if os.path.exists(plot_path):
    #         shutil.rmtree(plot_path)
    #
    #     visualizer = VisualizerImagingCI(visualize_path=plot_path)
    #
    #     visualizer.visualize_multiple_fit_cis_subplots_1d_lines(
    #         fits=[fit_ci_7x7, fit_ci_7x7], region="parallel_fpr"
    #     )
    #
    #     plot_path = path.join(plot_path, "multiple_fit_cis_1d_line_parallel_fpr")
    #
    #     assert path.join(plot_path, "subplot_residual_map_list.png") in plot_patch.paths
    #     assert (
    #         path.join(plot_path, "subplot_normalized_residual_map_list.png")
    #         in plot_patch.paths
    #     )
    #     assert (
    #         path.join(plot_path, "subplot_chi_squared_map_list.png")
    #         not in plot_patch.paths
    #     )
