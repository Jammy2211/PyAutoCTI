from os import path

import pytest

from autocti import plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_dataset_1d_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "files",
        "plots",
        "dataset_1d",
    )


def test__individual_attributes_are_output(dataset_1d_7, plot_path, plot_patch):

    dataset_1d_plotter = aplt.Dataset1DPlotter(
        dataset=dataset_1d_7,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    dataset_1d_plotter.figures_1d(
        data=True, noise_map=True, pre_cti_data=True, signal_to_noise_map=True
    )

    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") in plot_patch.paths
    assert path.join(plot_path, "pre_cti_data.png") in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") in plot_patch.paths

    plot_patch.paths = []

    dataset_1d_plotter.figures_1d(data=True, pre_cti_data=True)

    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "pre_cti_data.png") in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths


def test__individual_1d_of_region_are_output(dataset_1d_7, plot_path, plot_patch):

    imaging_ci_plotter = aplt.Dataset1DPlotter(
        dataset=dataset_1d_7,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    imaging_ci_plotter.figures_1d(
        region="fpr",
        data=True,
        noise_map=True,
        pre_cti_data=True,
        signal_to_noise_map=True,
    )

    assert path.join(plot_path, "data_fpr.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map_fpr.png") in plot_patch.paths
    assert path.join(plot_path, "pre_cti_data_fpr.png") in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map_fpr.png") in plot_patch.paths

    plot_patch.paths = []

    imaging_ci_plotter.figures_1d(region="fpr", data=True, pre_cti_data=True)

    assert path.join(plot_path, "data_fpr.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map_fpr.png") not in plot_patch.paths
    assert path.join(plot_path, "pre_cti_data_fpr.png") in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map_fpr.png") not in plot_patch.paths


def test__subplots__output(dataset_1d_7, plot_path, plot_patch):

    dataset_1d_plotter = aplt.Dataset1DPlotter(
        dataset=dataset_1d_7,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    dataset_1d_plotter.subplot_dataset_1d()
    assert path.join(plot_path, "subplot_dataset_1d.png") in plot_patch.paths

    dataset_1d_plotter.subplot_dataset_1d(region="fpr")
    assert path.join(plot_path, "subplot_dataset_1d_fpr.png") in plot_patch.paths
