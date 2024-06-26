import os
import shutil
from os import path

import pytest
from autocti.dataset_1d.model.plotter_interface import PlotterInterfaceDataset1D

directory = path.dirname(path.abspath(__file__))


@pytest.fixture(name="plot_path")
def make_visualizer_plotter_setup():
    return path.join("{}".format(directory), "files")


def test__dataset_1d(dataset_1d_7, plot_path, plot_patch):
    if path.exists(plot_path):
        shutil.rmtree(plot_path)

    visualizer = PlotterInterfaceDataset1D(image_path=plot_path)

    visualizer.dataset(dataset=dataset_1d_7)

    plot_path = path.join(plot_path, "dataset")

    assert path.join(plot_path, "subplot_dataset.png") in plot_patch.paths
    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths


def test__dataset_1d_region(dataset_1d_7, plot_path, plot_patch):
    if path.exists(plot_path):
        shutil.rmtree(plot_path)

    visualizer = PlotterInterfaceDataset1D(image_path=plot_path)

    visualizer.dataset_regions(dataset=dataset_1d_7, region_list=["fpr"])

    plot_path = path.join(plot_path, "dataset")

    assert path.join(plot_path, "subplot_dataset_fpr.png") in plot_patch.paths
    assert path.join(plot_path, "data_fpr.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map_fpr.png") not in plot_patch.paths


def test__fit_1d(fit_1d_7, plot_path, plot_patch):
    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)

    visualizer = PlotterInterfaceDataset1D(image_path=plot_path)

    visualizer.fit(fit=fit_1d_7, during_analysis=True)

    plot_path = path.join(plot_path, "fit_dataset")

    assert path.join(plot_path, "subplot_fit.png") in plot_patch.paths
    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths

    plot_patch.paths = []

    visualizer.fit(fit=fit_1d_7, during_analysis=False)

    assert path.join(plot_path, "subplot_fit.png") in plot_patch.paths
    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") in plot_patch.paths


def test__fit_1d_region(fit_1d_7, plot_path, plot_patch):
    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)

    visualizer = PlotterInterfaceDataset1D(image_path=plot_path)

    visualizer.fit_regions(fit=fit_1d_7, region_list=["fpr"], during_analysis=True)

    plot_path = path.join(plot_path, "fit_dataset")

    assert path.join(plot_path, "subplot_fit_fpr.png") in plot_patch.paths
    assert path.join(plot_path, "data_fpr.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map_fpr.png") not in plot_patch.paths


def test__fit_combined(fit_1d_7, plot_path, plot_patch):
    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)

    visualizer = PlotterInterfaceDataset1D(image_path=plot_path)

    visualizer.fit_combined(fit_list=[fit_1d_7, fit_1d_7], during_analysis=True)

    plot_path = path.join(plot_path, "fit_dataset_combined")

    assert path.join(plot_path, "subplot_residual_map.png") in plot_patch.paths
    assert path.join(plot_path, "subplot_chi_squared_map.png") not in plot_patch.paths


def test__fit_region(fit_1d_7, plot_path, plot_patch):
    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)

    visualizer = PlotterInterfaceDataset1D(image_path=plot_path)

    visualizer.fit_region_combined(
        fit_list=[fit_1d_7, fit_1d_7], region_list=["fpr"], during_analysis=True
    )

    plot_path = path.join(plot_path, "fit_dataset_combined")

    assert path.join(plot_path, "subplot_residual_map_fpr.png") in plot_patch.paths
    assert (
        path.join(plot_path, "subplot_chi_squared_map_fpr.png") not in plot_patch.paths
    )
