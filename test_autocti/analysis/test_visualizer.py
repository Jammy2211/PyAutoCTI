import os
import shutil
from os import path

import pytest
from autocti.analysis import visualizer as vis
from autoconf import conf

directory = path.dirname(path.abspath(__file__))


@pytest.fixture(name="plot_path")
def make_visualizer_plotter_setup():
    return path.join("{}".format(directory), "files")


@pytest.fixture(autouse=True)
def push_config(plot_path):
    conf.instance.push(path.join(directory, "config"), output_path=plot_path)


class TestVisualizer:
    def test__visualizes_imaging_using_configs(
        self, imaging_ci_7x7, plot_path, plot_patch
    ):

        if path.exists(plot_path):
            shutil.rmtree(plot_path)

        imaging_ci_7x7.cosmic_ray_map[0, 0] = 1

        visualizer = vis.Visualizer(visualize_path=plot_path)

        visualizer.visualize_imaging_ci(imaging_ci=imaging_ci_7x7)

        plot_path = path.join(plot_path, "imaging_ci_0")

        assert path.join(plot_path, "subplot_imaging_ci.png") in plot_patch.paths
        assert path.join(plot_path, "image.png") in plot_patch.paths
        assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
        assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths
        assert path.join(plot_path, "pre_cti_ci.png") in plot_patch.paths
        assert path.join(plot_path, "cosmic_ray_map.png") in plot_patch.paths

    def test__visualizes_imaging_lines_using_configs(
        self, imaging_ci_7x7, plot_path, plot_patch
    ):

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = vis.Visualizer(visualize_path=plot_path)

        visualizer.visualize_imaging_ci_lines(
            imaging_ci=imaging_ci_7x7, line_region="parallel_front_edge"
        )

        plot_path = path.join(plot_path, "imaging_ci_0")

        assert (
            path.join(plot_path, "subplot_1d_ci_parallel_front_edge.png")
            in plot_patch.paths
        )
        assert path.join(plot_path, "image_parallel_front_edge.png") in plot_patch.paths
        assert (
            path.join(plot_path, "noise_map_parallel_front_edge.png")
            not in plot_patch.paths
        )
        assert (
            path.join(plot_path, "signal_to_noise_map_parallel_front_edge.png")
            not in plot_patch.paths
        )
        assert (
            path.join(plot_path, "pre_cti_ci_parallel_front_edge.png")
            in plot_patch.paths
        )

    def test___visualizes_fit_using_configs(
        self, imaging_ci_7x7, fit_ci_7x7, plot_path, plot_patch
    ):

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = vis.Visualizer(visualize_path=plot_path)

        visualizer.visualize_fit_ci(fit=fit_ci_7x7, during_analysis=True)

        plot_path = path.join(plot_path, "fit_imaging_ci_0")

        assert path.join(plot_path, "subplot_fit_ci.png") in plot_patch.paths
        assert path.join(plot_path, "image.png") in plot_patch.paths
        assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
        assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths
        assert path.join(plot_path, "pre_cti_ci.png") in plot_patch.paths
        assert path.join(plot_path, "post_cti_ci.png") in plot_patch.paths
        assert path.join(plot_path, "residual_map.png") not in plot_patch.paths
        assert path.join(plot_path, "normalized_residual_map.png") in plot_patch.paths
        assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths

        plot_patch.paths = []

        visualizer.visualize_fit_ci(fit=fit_ci_7x7, during_analysis=False)

        assert path.join(plot_path, "subplot_fit_ci.png") in plot_patch.paths
        assert path.join(plot_path, "image.png") in plot_patch.paths
        assert path.join(plot_path, "noise_map.png") in plot_patch.paths
        assert path.join(plot_path, "signal_to_noise_map.png") in plot_patch.paths
        assert path.join(plot_path, "pre_cti_ci.png") in plot_patch.paths
        assert path.join(plot_path, "post_cti_ci.png") in plot_patch.paths
        assert path.join(plot_path, "residual_map.png") in plot_patch.paths
        assert path.join(plot_path, "normalized_residual_map.png") in plot_patch.paths
        assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths

    def test___visualizes_fit_lines_using_configs(
        self, imaging_ci_7x7, fit_ci_7x7, plot_path, plot_patch
    ):

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = vis.Visualizer(visualize_path=plot_path)

        visualizer.visualize_fit_ci_1d_lines(
            fit=fit_ci_7x7, line_region="parallel_front_edge", during_analysis=True
        )

        plot_path = path.join(plot_path, "fit_imaging_ci_0")

        assert (
            path.join(plot_path, "subplot_1d_fit_ci_parallel_front_edge.png")
            in plot_patch.paths
        )
        assert path.join(plot_path, "image_parallel_front_edge.png") in plot_patch.paths
        assert (
            path.join(plot_path, "noise_parallel_front_edge.png")
            not in plot_patch.paths
        )
        assert (
            path.join(plot_path, "signal_to_noise_map_parallel_front_edge.png")
            not in plot_patch.paths
        )
        assert (
            path.join(plot_path, "pre_cti_ci_parallel_front_edge.png")
            in plot_patch.paths
        )
        assert (
            path.join(plot_path, "post_cti_ci_parallel_front_edge.png")
            in plot_patch.paths
        )
        assert (
            path.join(plot_path, "residual_map_parallel_front_edge.png")
            not in plot_patch.paths
        )
        assert (
            path.join(plot_path, "normalized_residual_map_parallel_front_edge.png")
            in plot_patch.paths
        )
        assert (
            path.join(plot_path, "chi_squared_map_parallel_front_edge.png")
            in plot_patch.paths
        )

        plot_patch.paths = []

        visualizer.visualize_fit_ci_1d_lines(
            fit=fit_ci_7x7, line_region="parallel_front_edge", during_analysis=False
        )

        assert (
            path.join(plot_path, "subplot_1d_fit_ci_parallel_front_edge.png")
            in plot_patch.paths
        )
        assert path.join(plot_path, "image_parallel_front_edge.png") in plot_patch.paths
        assert (
            path.join(plot_path, "noise_map_parallel_front_edge.png")
            in plot_patch.paths
        )
        assert (
            path.join(plot_path, "signal_to_noise_map_parallel_front_edge.png")
            in plot_patch.paths
        )
        assert (
            path.join(plot_path, "pre_cti_ci_parallel_front_edge.png")
            in plot_patch.paths
        )
        assert (
            path.join(plot_path, "post_cti_ci_parallel_front_edge.png")
            in plot_patch.paths
        )
        assert (
            path.join(plot_path, "residual_map_parallel_front_edge.png")
            in plot_patch.paths
        )
        assert (
            path.join(plot_path, "normalized_residual_map_parallel_front_edge.png")
            in plot_patch.paths
        )
        assert (
            path.join(plot_path, "chi_squared_map_parallel_front_edge.png")
            in plot_patch.paths
        )

    def test__visualize_multiple_fit_cis_subplots(
        self, imaging_ci_7x7, fit_ci_7x7, plot_path, plot_patch
    ):

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = vis.Visualizer(visualize_path=plot_path)

        visualizer.visualize_multiple_fit_cis_subplots(fits=[fit_ci_7x7, fit_ci_7x7])

        plot_path = path.join(plot_path, "multiple_fit_cis")

        assert path.join(plot_path, "subplot_residual_map_list.png") in plot_patch.paths
        assert (
            path.join(plot_path, "subplot_normalized_residual_map_list.png")
            in plot_patch.paths
        )
        assert (
            path.join(plot_path, "subplot_chi_squared_map_list.png")
            not in plot_patch.paths
        )

    def test__visualize_multiple_fit_cis_1d_line_subplots(
        self, imaging_ci_7x7, fit_ci_7x7, plot_path, plot_patch
    ):

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = vis.Visualizer(visualize_path=plot_path)

        visualizer.visualize_multiple_fit_cis_subplots_1d_lines(
            fits=[fit_ci_7x7, fit_ci_7x7], line_region="parallel_front_edge"
        )

        plot_path = path.join(plot_path, "multiple_fit_cis_1d_line_parallel_front_edge")

        assert path.join(plot_path, "subplot_residual_map_list.png") in plot_patch.paths
        assert (
            path.join(plot_path, "subplot_normalized_residual_map_list.png")
            in plot_patch.paths
        )
        assert (
            path.join(plot_path, "subplot_chi_squared_map_list.png")
            not in plot_patch.paths
        )
