import copy
from os import path
import pytest
from autocti import plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_fit_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "fit"
    )


def test__individual_attribute_plots__all_plot_correctly(
    fit_ci_7x7, plot_path, plot_patch
):
    fit_plotter = aplt.FitImagingCIPlotter(
        fit=fit_ci_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    fit_plotter.figures_2d(
        data=True,
        noise_map=True,
        signal_to_noise_map=True,
        pre_cti_data=True,
        post_cti_data=True,
        residual_map=True,
        normalized_residual_map=True,
        chi_squared_map=True,
    )

    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") in plot_patch.paths
    assert path.join(plot_path, "pre_cti_data.png") in plot_patch.paths
    assert path.join(plot_path, "post_cti_data.png") in plot_patch.paths
    assert path.join(plot_path, "residual_map.png") in plot_patch.paths
    assert path.join(plot_path, "normalized_residual_map.png") in plot_patch.paths
    assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths

    plot_patch.paths = []

    fit_plotter.figures_2d(
        data=True,
        noise_map=False,
        signal_to_noise_map=False,
        pre_cti_data=True,
        post_cti_data=True,
        chi_squared_map=True,
    )

    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "pre_cti_data.png") in plot_patch.paths
    assert path.join(plot_path, "post_cti_data.png") in plot_patch.paths
    assert path.join(plot_path, "residual_map.png") not in plot_patch.paths
    assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths


def test__individual_line_attriutes_plot__all_plot_correctly_output(
    fit_ci_7x7, plot_path, plot_patch
):
    fit_plotter = aplt.FitImagingCIPlotter(
        fit=fit_ci_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    fit_plotter.figures_1d(
        region="parallel_fpr",
        data=True,
        noise_map=True,
        signal_to_noise_map=True,
        pre_cti_data=True,
        post_cti_data=True,
        residual_map=True,
        normalized_residual_map=True,
        chi_squared_map=True,
    )

    assert path.join(plot_path, "data_parallel_fpr.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map_parallel_fpr.png") in plot_patch.paths
    assert (
        path.join(plot_path, "signal_to_noise_map_parallel_fpr.png") in plot_patch.paths
    )
    assert path.join(plot_path, "pre_cti_data_parallel_fpr.png") in plot_patch.paths
    assert path.join(plot_path, "post_cti_data_parallel_fpr.png") in plot_patch.paths
    assert path.join(plot_path, "residual_map_parallel_fpr.png") in plot_patch.paths
    assert (
        path.join(plot_path, "normalized_residual_map_parallel_fpr.png")
        in plot_patch.paths
    )
    assert path.join(plot_path, "chi_squared_map_parallel_fpr.png") in plot_patch.paths

    plot_patch.paths = []

    fit_plotter.figures_1d(
        region="parallel_fpr",
        data=True,
        noise_map=False,
        signal_to_noise_map=False,
        pre_cti_data=True,
        post_cti_data=True,
        chi_squared_map=True,
    )

    assert path.join(plot_path, "data_parallel_fpr.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map_parallel_fpr.png") not in plot_patch.paths
    assert (
        path.join(plot_path, "signal_to_noise_map_parallel_fpr.png")
        not in plot_patch.paths
    )
    assert path.join(plot_path, "pre_cti_data_parallel_fpr.png") in plot_patch.paths
    assert path.join(plot_path, "post_cti_data_parallel_fpr.png") in plot_patch.paths
    assert path.join(plot_path, "residual_map_parallel_fpr.png") not in plot_patch.paths
    assert path.join(plot_path, "chi_squared_map_parallel_fpr.png") in plot_patch.paths


def test__figures_1d_data_binned(fit_ci_7x7, plot_path, plot_patch):
    fit_plotter = aplt.FitImagingCIPlotter(
        fit=fit_ci_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    fit_plotter.figures_1d_data_binned(
        rows_fpr=True,
        rows_no_fpr=True,
        columns_fpr=True,
        columns_no_fpr=True,
    )

    assert path.join(plot_path, "fit_data_binned_rows_fpr.png") in plot_patch.paths
    assert path.join(plot_path, "fit_data_binned_rows_no_fpr.png") in plot_patch.paths
    assert path.join(plot_path, "fit_data_binned_columns_fpr.png") in plot_patch.paths
    assert (
        path.join(plot_path, "fit_data_binned_columns_no_fpr.png") in plot_patch.paths
    )


def test__fit_ci_subplots_are_output(fit_ci_7x7, plot_path, plot_patch):
    fit_plotter = aplt.FitImagingCIPlotter(
        fit=fit_ci_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    fit_plotter.subplot_fit()
    assert path.join(plot_path, "subplot_fit.png") in plot_patch.paths

    fit_plotter.subplot_1d(region="parallel_fpr")
    assert (
        path.join(plot_path, "subplot_1d_fit_ci_parallel_fpr.png") in plot_patch.paths
    )

    fit_plotter.subplot_noise_scaling_map_dict()
    assert (
        path.join(plot_path, "subplot_noise_scaling_map_dict.png") in plot_patch.paths
    )


# def test__fit_ci_subplots_lines_are_output(fit_ci_7x7, plot_path, plot_patch):
#
#     fit_plotter = aplt.FitImagingCIPlotter(fit=fit_ci_7x7,
#                                         mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
#                                         mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
#                                         )
#
#     fit_plotter.subplot_residual_map_lines(
#         region="parallel_fpr",
#     )
#
#     assert path.join(plot_path, "subplot_residual_map_lines.png") in plot_patch.paths
#
#     fit_plotter.subplot_normalized_residual_map_lines(
#         region="parallel_fpr",
#     )
#     assert (
#         path.join(plot_path, "subplot_normalized_residual_map_lines.png")
#         in plot_patch.paths
#     )
#
#     fit_plotter.subplot_chi_squared_map_lines(
#         region="parallel_fpr",
#     )
#     assert path.join(plot_path, "subplot_chi_squared_map_lines.png") in plot_patch.paths
