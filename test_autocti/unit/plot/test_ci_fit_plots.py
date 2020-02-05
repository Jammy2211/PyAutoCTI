import numpy as np

import autoarray as aa
import autocti as ac
import autocti.plot as aplt
from test_autocti.mock.mock import MockPattern, MockCIFrame

import os
import pytest

directory = os.path.dirname(os.path.realpath(__file__))

@pytest.fixture(name="plot_path")
def make_ci_fit_plotter_setup():
    return "{}/../../test_files/plotting/fit/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


@pytest.fixture(autouse=True)
def set_config_path():
    aa.conf.instance = aa.conf.Config(
        os.path.join(directory, "../test_files/plot"), os.path.join(directory, "output")
    )


def test__individual_attriute_plots__all_plot_correctly(fit_ci_imaging_7x7, plot_path, plot_patch):

    aplt.ci_fit.image(
        fit=fit_ci_imaging_7x7,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )
    assert plot_path + "image.png" in plot_patch.paths

    aplt.ci_fit.noise_map(
        fit=fit_ci_imaging_7x7,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )
    assert plot_path + "noise_map.png" in plot_patch.paths


    aplt.ci_fit.signal_to_noise_map(
        fit=fit_ci_imaging_7x7,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )
    assert plot_path + "signal_to_noise_map.png" in plot_patch.paths

    aplt.ci_fit.ci_pre_cti(
        fit=fit_ci_imaging_7x7,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )
    assert plot_path + "ci_pre_cti.png" in plot_patch.paths

    aplt.ci_fit.ci_post_cti(
        fit=fit_ci_imaging_7x7,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )
    assert plot_path + "ci_post_cti.png" in plot_patch.paths

    aplt.ci_fit.residual_map(
        fit=fit_ci_imaging_7x7,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )
    assert plot_path + "residual_map.png" in plot_patch.paths

    aplt.ci_fit.chi_squared_map(
        fit=fit_ci_imaging_7x7,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )
    assert plot_path + "chi_squared_map.png" in plot_patch.paths

    aplt.ci_fit.noise_scaling_maps(
        fit=fit_ci_imaging_7x7,
        sub_plotter=aplt.SubPlotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "noise_scaling_maps.png" in plot_patch.paths


def test__individual_line_attriutes_plot__all_plot_correctly_output(fit_ci_imaging_7x7, plot_path, plot_patch):

    aplt.ci_fit.image_line(
        fit=fit_ci_imaging_7x7,
        line_region="parallel_front_edge",
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "image_line.png" in plot_patch.paths

    aplt.ci_fit.noise_map_line(
        fit=fit_ci_imaging_7x7,
        line_region="parallel_front_edge",
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "noise_map_line.png" in plot_patch.paths

    aplt.ci_fit.signal_to_noise_map_line(
        fit=fit_ci_imaging_7x7,
        line_region="parallel_front_edge",
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "signal_to_noise_map_line.png" in plot_patch.paths

    aplt.ci_fit.ci_pre_cti_line(
        fit=fit_ci_imaging_7x7,
        line_region="parallel_front_edge",
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "ci_pre_cti_line.png" in plot_patch.paths

    aplt.ci_fit.ci_post_cti_line(
        fit=fit_ci_imaging_7x7,
        line_region="parallel_front_edge",
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "ci_post_cti_line.png" in plot_patch.paths

    aplt.ci_fit.residual_map_line(
        fit=fit_ci_imaging_7x7,
        line_region="parallel_front_edge",
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "residual_map_line.png" in plot_patch.paths

    aplt.ci_fit.chi_squared_map_line(
        fit=fit_ci_imaging_7x7,
        line_region="parallel_front_edge",
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "chi_squared_map_line.png" in plot_patch.paths


def test__ci_fit_subplots_are_output(fit_ci_imaging_7x7, plot_path, plot_patch):

    aplt.ci_fit.subplot_ci_fit(
        fit=fit_ci_imaging_7x7,
        sub_plotter=aplt.SubPlotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "subplot_ci_fit.png" in plot_patch.paths

    aplt.ci_fit.subplot_residual_maps(
        fits=[fit_ci_imaging_7x7],
        sub_plotter=aplt.SubPlotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "subplot_residual_maps.png" in plot_patch.paths

    aplt.ci_fit.subplot_chi_squared_maps(
        fits=[fit_ci_imaging_7x7],
        sub_plotter=aplt.SubPlotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "subplot_chi_squared_maps.png" in plot_patch.paths

def test__ci_fit_subplots_lines_are_output(fit_ci_imaging_7x7, plot_path, plot_patch):

    aplt.ci_fit.subplot_residual_map_lines(
        fits=[fit_ci_imaging_7x7],
        line_region="parallel_front_edge",
        sub_plotter=aplt.SubPlotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "subplot_residual_map_lines.png" in plot_patch.paths

    aplt.ci_fit.subplot_chi_squared_map_lines(
        fits=[fit_ci_imaging_7x7],
        line_region="parallel_front_edge",
        sub_plotter=aplt.SubPlotter(output=aplt.Output(plot_path, format="png")),
    )

    assert (
        plot_path + "subplot_chi_squared_map_lines.png" in plot_patch.paths
    )


def test__fit_individuals__depedent_on_input(fit_ci_imaging_7x7, plot_path, plot_patch):

    aplt.ci_fit.individuals(
        fit=fit_ci_imaging_7x7,
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


def test__fit_individuals_line__dependent_on_input(
    fit_ci_imaging_7x7, plot_path, plot_patch
):

    aplt.ci_fit.individuals_lines(
        fit=fit_ci_imaging_7x7,
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

    assert (
        plot_path + "signal_to_noise_map_line.png" not in plot_patch.paths
    )

    assert plot_path + "ci_pre_cti_line.png" in plot_patch.paths

    assert plot_path + "ci_post_cti_line.png" in plot_patch.paths

    assert plot_path + "residual_map_line.png" not in plot_patch.paths

    assert plot_path + "chi_squared_map_line.png" in plot_patch.paths


def test__plot_ci_fit_for_phase(fit, plot_path, plot_patch):

    aplt.ci_fit.plot_ci_fit_for_phase(
        fits=[fit],
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
        plot_noise_scaling_maps=False,
        plot_parallel_front_edge_line=True,
        plot_parallel_trails_line=False,
        plot_serial_front_edge_line=True,
        plot_serial_trails_line=False,
        visualize_path=plot_path,
    )

    assert (
        plot_path + "/ci_image_10/structures/ci_fit.png" in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/structures/fit_image.png"
        in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/structures/fit_noise_map.png"
        not in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/structures/fit_ci_pre_cti.png"
        in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/structures/fit_signal_to_noise_map.png"
        not in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/structures/fit_residual_map.png"
        in plot_patch.paths
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
        plot_path
        + "/ci_image_10/parallel_front_edge/fit_signal_to_noise_map_line.png"
        not in plot_patch.paths
    )
    assert (
        plot_path
        + "/ci_image_10/parallel_front_edge/fit_residual_map_line.png"
        in plot_patch.paths
    )
    assert (
        plot_path
        + "/ci_image_10/parallel_front_edge/fit_chi_squared_map_line.png"
        not in plot_patch.paths
    )

    assert (
        plot_path + "/ci_image_10/serial_front_edge/ci_fit_line.png"
        in plot_patch.paths
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
        plot_path
        + "/ci_image_10/serial_front_edge/fit_signal_to_noise_map_line.png"
        not in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/serial_front_edge/fit_residual_map_line.png"
        in plot_patch.paths
    )
    assert (
        plot_path
        + "/ci_image_10/serial_front_edge/fit_chi_squared_map_line.png"
        not in plot_patch.paths
    )

    assert plot_path + "/ci_fits_residual_maps.png" in plot_patch.paths
    assert plot_path + "/ci_fits_chi_sqaured_maps.png" not in plot_patch.paths