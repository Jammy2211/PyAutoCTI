import numpy as np

from autoconf import conf
import autocti.plot as aplt

import os
import pytest

directory = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_ci_fit_plotter_setup():
    return "{}/files/plots/fit/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        os.path.join(directory, "files/plotter"), os.path.join(directory, "output")
    )


def test__individual_attriute_plots__all_plot_correctly(
    ci_fit_7x7, plot_path, plot_patch
):

    aplt.CIFit.image(
        fit=ci_fit_7x7,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )
    assert plot_path + "image.png" in plot_patch.paths

    aplt.CIFit.noise_map(
        fit=ci_fit_7x7,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )
    assert plot_path + "noise_map.png" in plot_patch.paths

    aplt.CIFit.signal_to_noise_map(
        fit=ci_fit_7x7,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )
    assert plot_path + "signal_to_noise_map.png" in plot_patch.paths

    aplt.CIFit.ci_pre_cti(
        fit=ci_fit_7x7,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )
    assert plot_path + "ci_pre_cti.png" in plot_patch.paths

    aplt.CIFit.ci_post_cti(
        fit=ci_fit_7x7,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )
    assert plot_path + "ci_post_cti.png" in plot_patch.paths

    aplt.CIFit.residual_map(
        fit=ci_fit_7x7,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )
    assert plot_path + "residual_map.png" in plot_patch.paths

    aplt.CIFit.chi_squared_map(
        fit=ci_fit_7x7,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )
    assert plot_path + "chi_squared_map.png" in plot_patch.paths

    aplt.CIFit.noise_scaling_maps(
        fit=ci_fit_7x7,
        sub_plotter=aplt.SubPlotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "noise_scaling_maps.png" in plot_patch.paths


def test__individual_line_attriutes_plot__all_plot_correctly_output(
    ci_fit_7x7, plot_path, plot_patch
):

    aplt.CIFit.image_line(
        fit=ci_fit_7x7,
        line_region="parallel_front_edge",
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "image_line.png" in plot_patch.paths

    aplt.CIFit.noise_map_line(
        fit=ci_fit_7x7,
        line_region="parallel_front_edge",
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "noise_map_line.png" in plot_patch.paths

    aplt.CIFit.signal_to_noise_map_line(
        fit=ci_fit_7x7,
        line_region="parallel_front_edge",
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "signal_to_noise_map_line.png" in plot_patch.paths

    aplt.CIFit.ci_pre_cti_line(
        fit=ci_fit_7x7,
        line_region="parallel_front_edge",
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "ci_pre_cti_line.png" in plot_patch.paths

    aplt.CIFit.ci_post_cti_line(
        fit=ci_fit_7x7,
        line_region="parallel_front_edge",
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "ci_post_cti_line.png" in plot_patch.paths

    aplt.CIFit.residual_map_line(
        fit=ci_fit_7x7,
        line_region="parallel_front_edge",
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "residual_map_line.png" in plot_patch.paths

    aplt.CIFit.chi_squared_map_line(
        fit=ci_fit_7x7,
        line_region="parallel_front_edge",
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "chi_squared_map_line.png" in plot_patch.paths


def test__ci_fit_subplots_are_output(ci_fit_7x7, plot_path, plot_patch):

    aplt.CIFit.subplot_ci_fit(
        fit=ci_fit_7x7,
        sub_plotter=aplt.SubPlotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "subplot_ci_fit.png" in plot_patch.paths

    aplt.CIFit.subplot_residual_maps(
        fits=[ci_fit_7x7],
        sub_plotter=aplt.SubPlotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "subplot_residual_maps.png" in plot_patch.paths

    aplt.CIFit.subplot_chi_squared_maps(
        fits=[ci_fit_7x7],
        sub_plotter=aplt.SubPlotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "subplot_chi_squared_maps.png" in plot_patch.paths


def test__ci_fit_subplots_lines_are_output(ci_fit_7x7, plot_path, plot_patch):

    aplt.CIFit.subplot_residual_map_lines(
        fits=[ci_fit_7x7],
        line_region="parallel_front_edge",
        sub_plotter=aplt.SubPlotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "subplot_residual_map_lines.png" in plot_patch.paths

    aplt.CIFit.subplot_chi_squared_map_lines(
        fits=[ci_fit_7x7],
        line_region="parallel_front_edge",
        sub_plotter=aplt.SubPlotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "subplot_chi_squared_map_lines.png" in plot_patch.paths


def test__fit_individuals__dependent_on_input(ci_fit_7x7, plot_path, plot_patch):

    aplt.CIFit.individuals(
        fit=ci_fit_7x7,
        plot_image=True,
        plot_noise_map=False,
        plot_signal_to_noise_map=False,
        plot_ci_pre_cti=True,
        plot_ci_post_cti=True,
        plot_chi_squared_map=True,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "image.png" in plot_patch.paths

    assert plot_path + "noise_map.png" not in plot_patch.paths

    assert plot_path + "signal_to_noise_map.png" not in plot_patch.paths

    assert plot_path + "ci_pre_cti.png" in plot_patch.paths

    assert plot_path + "ci_post_cti.png" in plot_patch.paths

    assert plot_path + "residual_map.png" not in plot_patch.paths

    assert plot_path + "chi_squared_map.png" in plot_patch.paths


def test__fit_individuals_line__dependent_on_input(ci_fit_7x7, plot_path, plot_patch):

    aplt.CIFit.individuals_lines(
        fit=ci_fit_7x7,
        line_region="parallel_front_edge",
        plot_image=True,
        plot_noise_map=False,
        plot_signal_to_noise_map=False,
        plot_ci_pre_cti=True,
        plot_ci_post_cti=True,
        plot_chi_squared_map=True,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "image_line.png" in plot_patch.paths

    assert plot_path + "noise_map_line.png" not in plot_patch.paths

    assert plot_path + "signal_to_noise_map_line.png" not in plot_patch.paths

    assert plot_path + "ci_pre_cti_line.png" in plot_patch.paths

    assert plot_path + "ci_post_cti_line.png" in plot_patch.paths

    assert plot_path + "residual_map_line.png" not in plot_patch.paths

    assert plot_path + "chi_squared_map_line.png" in plot_patch.paths


def test__plot_ci_fit_for_phase(ci_fit_7x7, plot_path, plot_patch):

    aplt.CIFit.plot_ci_fit_for_phase(
        fits=[ci_fit_7x7],
        during_analysis=False,
        extract_array_from_mask=True,
        plot_all_at_end_png=False,
        plot_all_at_end_fits=False,
        plot_as_subplot=True,
        plot_residual_maps_subplot=True,
        plot_chi_squared_maps_subplot=False,
        plot_image=True,
        plot_noise_map=False,
        plot_ci_pre_cti=True,
        plot_signal_to_noise_map=False,
        plot_ci_post_cti=False,
        plot_residual_map=True,
        plot_chi_squared_map=False,
        plot_noise_scaling_maps_list=False,
        plot_parallel_front_edge_line=True,
        plot_parallel_trails_line=False,
        plot_serial_front_edge_line=True,
        plot_serial_trails_line=False,
        visualize_path=plot_path,
    )

    assert plot_path + "/ci_image_10/structures/ci_fit.png" in plot_patch.paths
    assert plot_path + "/ci_image_10/structures/fit_image.png" in plot_patch.paths
    assert (
        plot_path + "/ci_image_10/structures/fit_noise_map.png" not in plot_patch.paths
    )
    assert plot_path + "/ci_image_10/structures/fit_ci_pre_cti.png" in plot_patch.paths
    assert (
        plot_path + "/ci_image_10/structures/fit_signal_to_noise_map.png"
        not in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/structures/fit_residual_map.png" in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/structures/fit_chi_squared_map.png"
        not in plot_patch.paths
    )

    assert (
        plot_path + "/ci_image_10/parallel_front_edge/ci_fit_line.png"
        in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/parallel_front_edge/fit_image_line.png"
        in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/parallel_front_edge/fit_noise_map_line.png"
        not in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/parallel_front_edge/fit_ci_pre_cti_line.png"
        in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/parallel_front_edge/fit_signal_to_noise_map_line.png"
        not in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/parallel_front_edge/fit_residual_map_line.png"
        in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/parallel_front_edge/fit_chi_squared_map_line.png"
        not in plot_patch.paths
    )

    assert (
        plot_path + "/ci_image_10/serial_front_edge/ci_fit_line.png" in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/serial_front_edge/fit_image_line.png"
        in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/serial_front_edge/fit_noise_map_line.png"
        not in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/serial_front_edge/fit_ci_pre_cti_line.png"
        in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/serial_front_edge/fit_signal_to_noise_map_line.png"
        not in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/serial_front_edge/fit_residual_map_line.png"
        in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/serial_front_edge/fit_chi_squared_map_line.png"
        not in plot_patch.paths
    )

    assert plot_path + "/ci_fits_residual_maps.png" in plot_patch.paths
    assert plot_path + "/ci_fits_chi_sqaured_maps.png" not in plot_patch.paths
