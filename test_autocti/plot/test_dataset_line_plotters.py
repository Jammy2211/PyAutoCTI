from os import path

import pytest

from autocti import plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_dataset_line_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "files",
        "plots",
        "dataset_line",
    )


def test__individual_attributes_are_output(dataset_line_7, plot_path, plot_patch):

    dataset_line_plotter = aplt.DatasetLinePlotter(
        dataset_line=dataset_line_7,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    dataset_line_plotter.figures_1d(
        data=True, noise_map=True, pre_cti_data=True, signal_to_noise_map=True
    )

    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") in plot_patch.paths
    assert path.join(plot_path, "pre_cti_data.png") in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") in plot_patch.paths

    plot_patch.paths = []

    dataset_line_plotter.figures_1d(data=True, pre_cti_data=True)

    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "pre_cti_data.png") in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths


def test__subplot_dataset_line__is_output(dataset_line_7, plot_path, plot_patch):

    dataset_line_plotter = aplt.DatasetLinePlotter(
        dataset_line=dataset_line_7,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    dataset_line_plotter.subplot_dataset_line()
    assert path.join(plot_path, "subplot_dataset_line.png") in plot_patch.paths
