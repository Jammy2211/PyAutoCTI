import autoarray as aa
import autocti as ac
import autocti.plot as aplt

import os
import pytest

directory = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_ci_imaging_plotter_setup():
    return "{}/../../test_files/plotting/ci_imaging/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


@pytest.fixture(autouse=True)
def set_config_path():
    aa.conf.instance = aa.conf.Config(
        os.path.join(directory, "../test_files/plot"), os.path.join(directory, "output")
    )


def test__individual_attributes_are_output(ci_imaging_7x7, plot_path, plot_patch):

    aplt.ci_imaging.image(
        ci_imaging=ci_imaging_7x7,
        include=aplt.Include(),
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )
    assert plot_path + "image.png" in plot_patch.paths

    aplt.ci_imaging.noise_map(
        ci_imaging=ci_imaging_7x7,
        include=aplt.Include(),
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )
    assert plot_path + "noise_map.png" in plot_patch.paths

    aplt.ci_imaging.ci_pre_cti(
        ci_imaging=ci_imaging_7x7,
        include=aplt.Include(),
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )
    assert plot_path + "ci_pre_cti.png" in plot_patch.paths

    aplt.ci_imaging.signal_to_noise_map(
        ci_imaging=ci_imaging_7x7,
        include=aplt.Include(),
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )
    assert plot_path + "signal_to_noise_map.png" in plot_patch.paths

    aplt.ci_imaging.cosmic_ray_map(
        ci_imaging=ci_imaging_7x7,
        include=aplt.Include(),
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )
    assert plot_path + "cosmic_ray_map.png" in plot_patch.paths


def test__individual_lines_are_output(ci_imaging_7x7, plot_path, plot_patch):

    aplt.ci_imaging.image_line(
        ci_imaging=ci_imaging_7x7,
        line_region="parallel_front_edge",
        include=aplt.Include(),
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "image_line.png" in plot_patch.paths

    aplt.ci_imaging.noise_map_line(
        ci_imaging=ci_imaging_7x7,
        line_region="parallel_front_edge",
        include=aplt.Include(),
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "noise_map_line.png" in plot_patch.paths

    aplt.ci_imaging.ci_pre_cti_line(
        ci_imaging=ci_imaging_7x7,
        line_region="parallel_front_edge",
        include=aplt.Include(),
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "ci_pre_cti_line.png" in plot_patch.paths

    aplt.ci_imaging.signal_to_noise_map_line(
        ci_imaging=ci_imaging_7x7,
        line_region="parallel_front_edge",
        include=aplt.Include(),
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "signal_to_noise_map_line.png" in plot_patch.paths


def test__subplot_ci_lines__is_output(ci_imaging_7x7, plot_path, plot_patch):

    aplt.ci_imaging.subplot_ci_lines(
        ci_imaging=ci_imaging_7x7,
        line_region="parallel_front_edge",
        sub_plotter=aplt.SubPlotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "subplot_ci_lines.png" in plot_patch.paths


def test__subplot_ci_imaging__is_output(ci_imaging_7x7, plot_path, plot_patch):

    aplt.ci_imaging.subplot_ci_imaging(
        ci_imaging=ci_imaging_7x7,
        sub_plotter=aplt.SubPlotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "subplot_ci_imaging.png" in plot_patch.paths


def test__ci_individuals__output_dependent_on_inputs(
    ci_imaging_7x7, plot_path, plot_patch
):
    aplt.ci_imaging.individual(
        ci_imaging=ci_imaging_7x7,
        plot_image=True,
        plot_ci_pre_cti=True,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "image.png" in plot_patch.paths

    assert plot_path + "noise_map.png" not in plot_patch.paths

    assert plot_path + "ci_pre_cti.png" in plot_patch.paths

    assert plot_path + "signal_to_noise_map.png" not in plot_patch.paths


def test__ci_line_individuals__output_dependent_on_inputs(
    ci_imaging_7x7, plot_path, plot_patch
):

    aplt.ci_imaging.individual_ci_lines(
        ci_imaging=ci_imaging_7x7,
        line_region="parallel_front_edge",
        plot_image=True,
        plot_ci_pre_cti=True,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert plot_path + "image_line.png" in plot_patch.paths

    assert plot_path + "noise_map_line.png" not in plot_patch.paths

    assert plot_path + "ci_pre_cti_line.png" in plot_patch.paths

    assert plot_path + "signal_to_noise_map_line.png" not in plot_patch.paths


@pytest.fixture(name="data_extracted")
def make_ci_data_extracted(ci_imaging_7x7, mask):
    return ac.MaskedCIImaging(
        image=ci_imaging_7x7.image,
        noise_map=ci_imaging_7x7.noise_map,
        ci_pre_cti=ci_imaging_7x7.ci_pre_cti,
        mask=mask,
        ci_pattern=ci_imaging_7x7.ci_pattern,
        ci_frame=ci_imaging_7x7.ci_frame,
    )


def test__plot_ci_data_for_phase(data_extracted, plot_path, plot_patch):

    aplt.ci_imaging.ci_data_for_phase(
        ci_datas_extracted=[data_extracted],
        extract_array_from_mask=True,
        plot_as_subplot=True,
        plot_image=True,
        plot_noise_map=False,
        plot_ci_pre_cti=True,
        plot_signal_to_noise_map=False,
        plot_parallel_front_edge_line=True,
        plot_parallel_trails_line=False,
        plot_serial_front_edge_line=True,
        plot_serial_trails_line=False,
        visualize_path=plot_path,
    )

    assert plot_path + "/ci_image_10/structures/ci_data.png" in plot_patch.paths
    assert plot_path + "/ci_image_10/structures/ci_image.png" in plot_patch.paths
    assert (
        plot_path + "/ci_image_10/structures/ci_noise_map.png" not in plot_patch.paths
    )
    assert plot_path + "/ci_image_10/structures/ci_pre_cti.png" in plot_patch.paths
    assert (
        plot_path + "/ci_image_10/structures/ci_signal_to_noise_map.png"
        not in plot_patch.paths
    )

    assert (
        plot_path + "/ci_image_10/parallel_front_edge/ci_data_line.png"
        in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/parallel_front_edge/ci_image_line.png"
        in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/parallel_front_edge/ci_noise_map_line.png"
        not in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/parallel_front_edge/ci_pre_cti_line.png"
        in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/parallel_front_edge/ci_signal_to_noise_map_line.png"
        not in plot_patch.paths
    )

    assert (
        plot_path + "/ci_image_10/serial_front_edge/ci_data_line.png"
        in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/serial_front_edge/ci_image_line.png"
        in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/serial_front_edge/ci_noise_map_line.png"
        not in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/serial_front_edge/ci_pre_cti_line.png"
        in plot_patch.paths
    )
    assert (
        plot_path + "/ci_image_10/serial_front_edge/ci_signal_to_noise_map_line.png"
        not in plot_patch.paths
    )
