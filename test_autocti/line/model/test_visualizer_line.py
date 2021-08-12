import os
import shutil
from os import path

import pytest
from autocti.line.model.visualizer import VisualizerDatasetLine
from autoconf import conf

directory = path.dirname(path.abspath(__file__))


@pytest.fixture(name="plot_path")
def make_visualizer_plotter_setup():
    return path.join("{}".format(directory), "files")


@pytest.fixture(autouse=True)
def push_config(plot_path):
    conf.instance.push(path.join(directory, "config"), output_path=plot_path)


class TestVisualizerDatasetLine:
    def test__visualizes_dataset_line_using_configs(
        self, dataset_line_7, plot_path, plot_patch
    ):

        if path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = VisualizerDatasetLine(visualize_path=plot_path)

        visualizer.visualize_dataset_line(dataset_line=dataset_line_7)

        plot_path = path.join(plot_path, "dataset_line")

        assert path.join(plot_path, "subplot_dataset_line.png") in plot_patch.paths
        assert path.join(plot_path, "data.png") in plot_patch.paths
        assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
        assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths
        assert path.join(plot_path, "pre_cti_data.png") in plot_patch.paths

    def test___visualizes_fit_line_using_configs(
        self, fit_line_7, plot_path, plot_patch
    ):

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = VisualizerDatasetLine(visualize_path=plot_path)

        visualizer.visualize_fit_line(fit=fit_line_7, during_analysis=True)

        plot_path = path.join(plot_path, "fit_dataset_line")

        assert path.join(plot_path, "subplot_fit_dataset_line.png") in plot_patch.paths
        assert path.join(plot_path, "data.png") in plot_patch.paths
        assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
        assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths
        assert path.join(plot_path, "pre_cti_data.png") in plot_patch.paths
        assert path.join(plot_path, "post_cti_data.png") in plot_patch.paths
        assert path.join(plot_path, "residual_map.png") not in plot_patch.paths
        assert path.join(plot_path, "normalized_residual_map.png") in plot_patch.paths
        assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths

        plot_patch.paths = []

        visualizer.visualize_fit_line(fit=fit_line_7, during_analysis=False)

        assert path.join(plot_path, "subplot_fit_dataset_line.png") in plot_patch.paths
        assert path.join(plot_path, "data.png") in plot_patch.paths
        assert path.join(plot_path, "noise_map.png") in plot_patch.paths
        assert path.join(plot_path, "signal_to_noise_map.png") in plot_patch.paths
        assert path.join(plot_path, "pre_cti_data.png") in plot_patch.paths
        assert path.join(plot_path, "post_cti_data.png") in plot_patch.paths
        assert path.join(plot_path, "residual_map.png") in plot_patch.paths
        assert path.join(plot_path, "normalized_residual_map.png") in plot_patch.paths
        assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths
