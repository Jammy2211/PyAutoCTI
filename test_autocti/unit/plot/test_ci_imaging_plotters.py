from os import path

import pytest

from autocti import plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_ci_fit_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "fit"
    )


def test__individual_attributes_are_output(ci_imaging_7x7, plot_path, plot_patch):

    ci_imaging_plotter = aplt.CIImagingPlotter(
        imaging=ci_imaging_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    ci_imaging_plotter.figure_image()
    assert path.join(plot_path, "image.png") in plot_patch.paths

    ci_imaging_plotter.figure_noise_map()
    assert path.join(plot_path, "noise_map.png") in plot_patch.paths

    ci_imaging_plotter.figure_ci_pre_cti()
    assert path.join(plot_path, "ci_pre_cti.png") in plot_patch.paths

    ci_imaging_plotter.figure_signal_to_noise_map()
    assert path.join(plot_path, "signal_to_noise_map.png") in plot_patch.paths

    ci_imaging_7x7.cosmic_ray_map[0, 0] = 1.0

    ci_imaging_plotter.figure_cosmic_ray_map()
    assert path.join(plot_path, "cosmic_ray_map.png") in plot_patch.paths


def test__individual_lines_are_output(ci_imaging_7x7, plot_path, plot_patch):

    ci_imaging_plotter = aplt.CIImagingPlotter(
        imaging=ci_imaging_7x7,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    ci_imaging_plotter.figure_image_line(line_region="parallel_front_edge")
    assert path.join(plot_path, "image_line.png") in plot_patch.paths

    ci_imaging_plotter.figure_noise_map_line(line_region="parallel_front_edge")

    assert path.join(plot_path, "noise_map_line.png") in plot_patch.paths

    ci_imaging_plotter.figure_ci_pre_cti_line(line_region="parallel_front_edge")
    assert path.join(plot_path, "ci_pre_cti_line.png") in plot_patch.paths

    ci_imaging_plotter.figure_signal_to_noise_map_line(
        line_region="parallel_front_edge"
    )
    assert path.join(plot_path, "signal_to_noise_map_line.png") in plot_patch.paths


def test__subplot_ci_lines__is_output(ci_imaging_7x7, plot_path, plot_patch):

    ci_imaging_plotter = aplt.CIImagingPlotter(
        imaging=ci_imaging_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    ci_imaging_plotter.subplot_ci_lines(line_region="parallel_front_edge")
    assert path.join(plot_path, "subplot_ci_lines.png") in plot_patch.paths


def test__subplot_ci_imaging__is_output(ci_imaging_7x7, plot_path, plot_patch):

    ci_imaging_plotter = aplt.CIImagingPlotter(
        imaging=ci_imaging_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    ci_imaging_plotter.subplot_ci_imaging()
    assert path.join(plot_path, "subplot_ci_imaging.png") in plot_patch.paths


def test__ci_individuals__output_dependent_on_inputs(
    ci_imaging_7x7, plot_path, plot_patch
):

    ci_imaging_plotter = aplt.CIImagingPlotter(
        imaging=ci_imaging_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    ci_imaging_plotter.figure_individuals(plot_image=True, plot_ci_pre_cti=True)
    assert path.join(plot_path, "image.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "ci_pre_cti.png") in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths


def test__ci_line_individuals__output_dependent_on_inputs(
    ci_imaging_7x7, plot_path, plot_patch
):

    ci_imaging_plotter = aplt.CIImagingPlotter(
        imaging=ci_imaging_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    ci_imaging_plotter.figure_individual_ci_lines(
        line_region="parallel_front_edge", plot_image=True, plot_ci_pre_cti=True
    )
    assert path.join(plot_path, "image_line.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map_line.png") not in plot_patch.paths
    assert path.join(plot_path, "ci_pre_cti_line.png") in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map_line.png") not in plot_patch.paths
