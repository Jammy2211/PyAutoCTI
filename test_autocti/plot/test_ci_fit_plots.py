from os import path
import pytest
from autocti import plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_ci_fit_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "fit"
    )


def test__individual_attribute_plots__all_plot_correctly(
    ci_fit_7x7, plot_path, plot_patch
):

    ci_fit_plotter = aplt.CIFitPlotter(
        fit=ci_fit_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    ci_fit_plotter.figures(
        image=True,
        noise_map=True,
        signal_to_noise_map=True,
        ci_pre_cti=True,
        ci_post_cti=True,
        residual_map=True,
        normalized_residual_map=True,
        chi_squared_map=True,
    )

    assert path.join(plot_path, "image.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") in plot_patch.paths
    assert path.join(plot_path, "ci_pre_cti.png") in plot_patch.paths
    assert path.join(plot_path, "ci_post_cti.png") in plot_patch.paths
    assert path.join(plot_path, "residual_map.png") in plot_patch.paths
    assert path.join(plot_path, "normalized_residual_map.png") in plot_patch.paths
    assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths

    plot_patch.paths = []

    ci_fit_plotter.figures(
        image=True,
        noise_map=False,
        signal_to_noise_map=False,
        ci_pre_cti=True,
        ci_post_cti=True,
        chi_squared_map=True,
    )

    assert path.join(plot_path, "image.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "ci_pre_cti.png") in plot_patch.paths
    assert path.join(plot_path, "ci_post_cti.png") in plot_patch.paths
    assert path.join(plot_path, "residual_map.png") not in plot_patch.paths
    assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths


def test__individual_line_attriutes_plot__all_plot_correctly_output(
    ci_fit_7x7, plot_path, plot_patch
):

    ci_fit_plotter = aplt.CIFitPlotter(
        fit=ci_fit_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    ci_fit_plotter.figures_1d_ci_line_region(
        line_region="parallel_front_edge",
        image=True,
        noise_map=True,
        signal_to_noise_map=True,
        ci_pre_cti=True,
        ci_post_cti=True,
        residual_map=True,
        normalized_residual_map=True,
        chi_squared_map=True,
    )

    assert path.join(plot_path, "image_parallel_front_edge.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map_parallel_front_edge.png") in plot_patch.paths
    assert (
        path.join(plot_path, "signal_to_noise_map_parallel_front_edge.png")
        in plot_patch.paths
    )
    assert (
        path.join(plot_path, "ci_pre_cti_parallel_front_edge.png") in plot_patch.paths
    )
    assert (
        path.join(plot_path, "ci_post_cti_parallel_front_edge.png") in plot_patch.paths
    )
    assert (
        path.join(plot_path, "residual_map_parallel_front_edge.png") in plot_patch.paths
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

    ci_fit_plotter.figures_1d_ci_line_region(
        line_region="parallel_front_edge",
        image=True,
        noise_map=False,
        signal_to_noise_map=False,
        ci_pre_cti=True,
        ci_post_cti=True,
        chi_squared_map=True,
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
        path.join(plot_path, "ci_pre_cti_parallel_front_edge.png") in plot_patch.paths
    )
    assert (
        path.join(plot_path, "ci_post_cti_parallel_front_edge.png") in plot_patch.paths
    )
    assert (
        path.join(plot_path, "residual_map_parallel_front_edge.png")
        not in plot_patch.paths
    )
    assert (
        path.join(plot_path, "chi_squared_map_parallel_front_edge.png")
        in plot_patch.paths
    )


def test__ci_fit_subplots_are_output(ci_fit_7x7, plot_path, plot_patch):

    ci_fit_plotter = aplt.CIFitPlotter(
        fit=ci_fit_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    ci_fit_plotter.subplot_ci_fit()
    assert path.join(plot_path, "subplot_ci_fit.png") in plot_patch.paths

    ci_fit_plotter.subplot_1d_ci_line_region(line_region="parallel_front_edge")
    assert (
        path.join(plot_path, "subplot_1d_ci_fit_parallel_front_edge.png")
        in plot_patch.paths
    )

    ci_fit_plotter.subplot_noise_scaling_maps()
    assert path.join(plot_path, "subplot_noise_scaling_maps.png") in plot_patch.paths

    # ci_fit_plotter.subplot_residual_maps(
    #     fits=[ci_fit_7x7],
    #     plotter=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    # )
    #
    # assert path.join(plot_path, "subplot_residual_maps.png") in plot_patch.paths
    #
    # ci_fit_plotter.subplot_normalized_residual_maps(
    #     fits=[ci_fit_7x7],
    #     plotter=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    # )
    #
    # assert (
    #     path.join(plot_path, "subplot_normalized_residual_maps.png") in plot_patch.paths
    # )
    #
    # ci_fit_plotter.subplot_chi_squared_maps(
    #     fits=[ci_fit_7x7],
    #     plotter=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    # )
    #
    # assert path.join(plot_path, "subplot_chi_squared_maps.png") in plot_patch.paths


# def test__ci_fit_subplots_lines_are_output(ci_fit_7x7, plot_path, plot_patch):
#
#     ci_fit_plotter = aplt.CIFitPlotter(fit=ci_fit_7x7,
#                                         mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
#                                         mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
#                                         )
#
#     ci_fit_plotter.subplot_residual_map_lines(
#         line_region="parallel_front_edge",
#     )
#
#     assert path.join(plot_path, "subplot_residual_map_lines.png") in plot_patch.paths
#
#     ci_fit_plotter.subplot_normalized_residual_map_lines(
#         line_region="parallel_front_edge",
#     )
#     assert (
#         path.join(plot_path, "subplot_normalized_residual_map_lines.png")
#         in plot_patch.paths
#     )
#
#     ci_fit_plotter.subplot_chi_squared_map_lines(
#         line_region="parallel_front_edge",
#     )
#     assert path.join(plot_path, "subplot_chi_squared_map_lines.png") in plot_patch.paths
