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

    aplt.CIImaging.image(
        ci_imaging=ci_imaging_7x7,
        include=aplt.Include(),
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )
    assert path.join(plot_path, "image.png") in plot_patch.paths

    aplt.CIImaging.noise_map(
        ci_imaging=ci_imaging_7x7,
        include=aplt.Include(),
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )
    assert path.join(plot_path, "noise_map.png") in plot_patch.paths

    aplt.CIImaging.ci_pre_cti(
        ci_imaging=ci_imaging_7x7,
        include=aplt.Include(),
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )
    assert path.join(plot_path, "ci_pre_cti.png") in plot_patch.paths

    aplt.CIImaging.signal_to_noise_map(
        ci_imaging=ci_imaging_7x7,
        include=aplt.Include(),
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )
    assert path.join(plot_path, "signal_to_noise_map.png") in plot_patch.paths

    ci_imaging_7x7.cosmic_ray_map[0, 0] = 1.0

    aplt.CIImaging.cosmic_ray_map(
        ci_imaging=ci_imaging_7x7,
        include=aplt.Include(),
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )
    assert path.join(plot_path, "cosmic_ray_map.png") in plot_patch.paths


def test__individual_lines_are_output(ci_imaging_7x7, plot_path, plot_patch):

    aplt.CIImaging.image_line(
        ci_imaging=ci_imaging_7x7,
        line_region="parallel_front_edge",
        include=aplt.Include(),
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "image_line.png") in plot_patch.paths

    aplt.CIImaging.noise_map_line(
        ci_imaging=ci_imaging_7x7,
        line_region="parallel_front_edge",
        include=aplt.Include(),
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "noise_map_line.png") in plot_patch.paths

    aplt.CIImaging.ci_pre_cti_line(
        ci_imaging=ci_imaging_7x7,
        line_region="parallel_front_edge",
        include=aplt.Include(),
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "ci_pre_cti_line.png") in plot_patch.paths

    aplt.CIImaging.signal_to_noise_map_line(
        ci_imaging=ci_imaging_7x7,
        line_region="parallel_front_edge",
        include=aplt.Include(),
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "signal_to_noise_map_line.png") in plot_patch.paths


def test__subplot_ci_lines__is_output(ci_imaging_7x7, plot_path, plot_patch):

    aplt.CIImaging.subplot_ci_lines(
        ci_imaging=ci_imaging_7x7,
        line_region="parallel_front_edge",
        sub_plotter=aplt.SubPlotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "subplot_ci_lines.png") in plot_patch.paths


def test__subplot_ci_imaging__is_output(ci_imaging_7x7, plot_path, plot_patch):

    aplt.CIImaging.subplot_ci_imaging(
        ci_imaging=ci_imaging_7x7,
        sub_plotter=aplt.SubPlotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "subplot_ci_imaging.png") in plot_patch.paths


def test__ci_individuals__output_dependent_on_inputs(
    ci_imaging_7x7, plot_path, plot_patch
):
    aplt.CIImaging.individual(
        ci_imaging=ci_imaging_7x7,
        plot_image=True,
        plot_ci_pre_cti=True,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "image.png") in plot_patch.paths

    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths

    assert path.join(plot_path, "ci_pre_cti.png") in plot_patch.paths

    assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths


def test__ci_line_individuals__output_dependent_on_inputs(
    ci_imaging_7x7, plot_path, plot_patch
):

    aplt.CIImaging.individual_ci_lines(
        ci_imaging=ci_imaging_7x7,
        line_region="parallel_front_edge",
        plot_image=True,
        plot_ci_pre_cti=True,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "image_line.png") in plot_patch.paths

    assert path.join(plot_path, "noise_map_line.png") not in plot_patch.paths

    assert path.join(plot_path, "ci_pre_cti_line.png") in plot_patch.paths

    assert path.join(plot_path, "signal_to_noise_map_line.png") not in plot_patch.paths
