from os import path

import pytest

from autocti import plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_dataset_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "files",
        "plots",
        "imaging_ci",
    )


def test__figures_2d__individual_attributes_are_output(
    imaging_ci_7x7, plot_path, plot_patch
):
    dataset_plotter = aplt.ImagingCIPlotter(
        dataset=imaging_ci_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    imaging_ci_7x7.cosmic_ray_map[0, 0] = 1.0

    dataset_plotter.figures_2d(
        data=True,
        noise_map=True,
        pre_cti_data=True,
        signal_to_noise_map=True,
        cosmic_ray_map=True,
    )

    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") in plot_patch.paths
    assert path.join(plot_path, "pre_cti_data.png") in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") in plot_patch.paths
    assert path.join(plot_path, "cosmic_ray_map.png") in plot_patch.paths

    plot_patch.paths = []

    dataset_plotter.figures_2d(data=True, pre_cti_data=True)

    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "pre_cti_data.png") in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths


def test__figures_1d__individual_1d_of_region_are_output(
    imaging_ci_7x7, plot_path, plot_patch
):
    dataset_plotter = aplt.ImagingCIPlotter(
        dataset=imaging_ci_7x7,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    dataset_plotter.figures_1d(
        region="parallel_fpr",
        data=True,
        noise_map=True,
        pre_cti_data=True,
        signal_to_noise_map=True,
    )

    assert path.join(plot_path, "data_parallel_fpr.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map_parallel_fpr.png") in plot_patch.paths
    assert path.join(plot_path, "pre_cti_data_parallel_fpr.png") in plot_patch.paths
    assert (
        path.join(plot_path, "signal_to_noise_map_parallel_fpr.png") in plot_patch.paths
    )

    plot_patch.paths = []

    dataset_plotter.figures_1d(region="parallel_fpr", data=True, pre_cti_data=True)

    assert path.join(plot_path, "data_parallel_fpr.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map_parallel_fpr.png") not in plot_patch.paths
    assert path.join(plot_path, "pre_cti_data_parallel_fpr.png") in plot_patch.paths
    assert (
        path.join(plot_path, "signal_to_noise_map_parallel_fpr.png")
        not in plot_patch.paths
    )


def test__figures_1d_data_binned(
    imaging_ci_7x7, plot_path, plot_patch
):
    dataset_plotter = aplt.ImagingCIPlotter(
        dataset=imaging_ci_7x7,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    dataset_plotter.figures_1d_data_binned(
        columns_fpr=True
    )

    assert path.join(plot_path, "data_binned_columns_fpr.png") in plot_patch.paths




def test__subplots__output(imaging_ci_7x7, plot_path, plot_patch):
    dataset_plotter = aplt.ImagingCIPlotter(
        dataset=imaging_ci_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    dataset_plotter.subplot_dataset()
    assert path.join(plot_path, "subplot_dataset.png") in plot_patch.paths

    dataset_plotter.subplot_1d(region="parallel_fpr")
    assert path.join(plot_path, "subplot_1d_ci_parallel_fpr.png") in plot_patch.paths
