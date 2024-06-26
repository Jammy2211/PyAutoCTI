import os
import shutil
from os import path

import pytest
from autocti.charge_injection.model.plotter_interface import PlotterInterfaceImagingCI

directory = path.dirname(path.abspath(__file__))


@pytest.fixture(name="plot_path")
def make_visualizer_plotter_setup():
    return path.join("{}".format(directory), "files")


def test__dataset(imaging_ci_7x7, plot_path, plot_patch):
    if path.exists(plot_path):
        shutil.rmtree(plot_path)

    imaging_ci_7x7.cosmic_ray_map[0, 0] = 1

    visualizer = PlotterInterfaceImagingCI(image_path=plot_path)

    visualizer.dataset(dataset=imaging_ci_7x7)

    plot_path = path.join(plot_path, "dataset")

    assert path.join(plot_path, "subplot_dataset.png") in plot_patch.paths
    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths


def test__dataset_regions(imaging_ci_7x7, plot_path, plot_patch):
    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)

    visualizer = PlotterInterfaceImagingCI(image_path=plot_path)

    visualizer.dataset_regions(dataset=imaging_ci_7x7, region_list=["parallel_fpr"])

    plot_path = path.join(plot_path, "dataset")

    assert path.join(plot_path, "subplot_1d_ci_parallel_fpr.png") in plot_patch.paths
    assert path.join(plot_path, "data_parallel_fpr.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map_parallel_fpr.png") not in plot_patch.paths


def test__fit_ci(fit_ci_7x7, plot_path, plot_patch):
    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)

    visualizer = PlotterInterfaceImagingCI(image_path=plot_path)

    visualizer.fit(fit=fit_ci_7x7, during_analysis=True)

    plot_path = path.join(plot_path, "fit_dataset")

    assert path.join(plot_path, "subplot_fit.png") in plot_patch.paths
    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths

    plot_patch.paths = []

    visualizer.fit(fit=fit_ci_7x7, during_analysis=False)

    assert path.join(plot_path, "subplot_fit.png") in plot_patch.paths
    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") in plot_patch.paths


def test__fit_ci_regions(fit_ci_7x7, plot_path, plot_patch):
    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)

    visualizer = PlotterInterfaceImagingCI(image_path=plot_path)

    visualizer.fit_1d_regions(
        fit=fit_ci_7x7, region_list=["parallel_fpr"], during_analysis=True
    )

    plot_path = path.join(plot_path, "fit_dataset")

    assert (
        path.join(plot_path, "subplot_1d_fit_ci_parallel_fpr.png") in plot_patch.paths
    )
    assert path.join(plot_path, "data_parallel_fpr.png") in plot_patch.paths
    assert path.join(plot_path, "noise_parallel_fpr.png") not in plot_patch.paths

    plot_patch.paths = []

    visualizer.fit_1d_regions(
        fit=fit_ci_7x7, region_list=["parallel_fpr"], during_analysis=False
    )

    assert (
        path.join(plot_path, "subplot_1d_fit_ci_parallel_fpr.png") in plot_patch.paths
    )
    assert path.join(plot_path, "data_parallel_fpr.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map_parallel_fpr.png") not in plot_patch.paths


def test__fit_combined(fit_ci_7x7, plot_path, plot_patch):
    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)

    visualizer = PlotterInterfaceImagingCI(image_path=plot_path)

    visualizer.fit_combined(fit_list=[fit_ci_7x7, fit_ci_7x7], during_analysis=True)

    plot_path = path.join(plot_path, "fit_dataset_combined")

    assert path.join(plot_path, "subplot_residual_map.png") in plot_patch.paths
    assert path.join(plot_path, "subplot_chi_squared_map.png") not in plot_patch.paths


def test__fit_regions_combined(fit_ci_7x7, plot_path, plot_patch):
    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)

    visualizer = PlotterInterfaceImagingCI(image_path=plot_path)

    visualizer.fit_1d_regions_combined(
        fit_list=[fit_ci_7x7, fit_ci_7x7],
        region_list=["parallel_fpr"],
        during_analysis=True,
    )

    plot_path = path.join(plot_path, "fit_dataset_combined")

    assert path.join(plot_path, "subplot_data_parallel_fpr.png") in plot_patch.paths
    assert (
        path.join(plot_path, "subplot_data_logy_parallel_fpr.png")
        not in plot_patch.paths
    )
