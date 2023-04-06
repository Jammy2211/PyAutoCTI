import os
import shutil
from os import path

import pytest
from autocti.dataset_1d.model.visualizer import VisualizerDataset1D
from autoconf import conf

directory = path.dirname(path.abspath(__file__))


@pytest.fixture(name="plot_path")
def make_visualizer_plotter_setup():
    return path.join("{}".format(directory), "files")


def test__visualizes_dataset_1d_using_configs(dataset_1d_7, plot_path, plot_patch):

    if path.exists(plot_path):
        shutil.rmtree(plot_path)

    visualizer = VisualizerDataset1D(visualize_path=plot_path)

    visualizer.visualize_dataset_1d(dataset_1d=dataset_1d_7)

    plot_path = path.join(plot_path, "dataset_1d")

    assert path.join(plot_path, "subplot_dataset_1d.png") in plot_patch.paths
    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths


def test___visualizes_fit_line_using_configs(fit_line_7, plot_path, plot_patch):

    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)

    visualizer = VisualizerDataset1D(visualize_path=plot_path)

    visualizer.visualize_fit_line(fit=fit_line_7, during_analysis=True)

    plot_path = path.join(plot_path, "fit_dataset_1d")

    assert path.join(plot_path, "subplot_fit_dataset_1d.png") in plot_patch.paths
    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths

    plot_patch.paths = []

    visualizer.visualize_fit_line(fit=fit_line_7, during_analysis=False)

    assert path.join(plot_path, "subplot_fit_dataset_1d.png") in plot_patch.paths
    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") in plot_patch.paths
