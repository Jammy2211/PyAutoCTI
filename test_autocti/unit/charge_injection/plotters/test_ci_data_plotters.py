import numpy as np

import autocti as ac
from test import MockPattern, MockCIFrame

import os
import pytest


@pytest.fixture(name="data_plotter_path")
def make_ci_data_plotter_setup():
    return "{}/../../test_files/plotting/ci_data/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


@pytest.fixture(name="mask")
def make_mask():
    return ac.Mask.empty_for_shape(shape=(6, 6))


@pytest.fixture(name="image")
def make_image():
    return np.ones((6, 6))


@pytest.fixture(name="noise_map")
def make_noise_map():
    return 2.0 * np.ones((6, 6))


@pytest.fixture(name="ci_pre_cti")
def make_ci_pre_cti():
    return 3.0 * np.ones((6, 6))


@pytest.fixture(name="data")
def make_ci_data(image, noise_map, ci_pre_cti):
    return ac.CIData(
        image=image,
        noise_map=noise_map,
        ci_pre_cti=ci_pre_cti,
        ci_pattern=MockPattern(),
        ci_frame=MockCIFrame(value=3.0),
    )


def test__image_is_output(data, mask, data_plotter_path, plot_patch):
    ac.ci_data_plotters.plot_image(
        ci_data=data,
        mask=mask,
        extract_array_from_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=data_plotter_path,
        output_format="png",
    )
    assert data_plotter_path + "ci_image.png" in plot_patch.paths


def test__noise_map_is_output(data, mask, data_plotter_path, plot_patch):
    ac.ci_data_plotters.plot_noise_map(
        ci_data=data,
        mask=mask,
        extract_array_from_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=data_plotter_path,
        output_format="png",
    )
    assert data_plotter_path + "ci_noise_map.png" in plot_patch.paths


def test__ci_pre_cti_is_output(data, mask, data_plotter_path, plot_patch):
    ac.ci_data_plotters.plot_ci_pre_cti(
        ci_data=data,
        mask=mask,
        extract_array_from_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=data_plotter_path,
        output_format="png",
    )
    assert data_plotter_path + "ci_pre_cti.png" in plot_patch.paths


def test__signal_to_noise_map_is_output(data, mask, data_plotter_path, plot_patch):
    ac.ci_data_plotters.plot_signal_to_noise_map(
        ci_data=data,
        mask=mask,
        extract_array_from_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=data_plotter_path,
        output_format="png",
    )
    assert data_plotter_path + "ci_signal_to_noise_map.png" in plot_patch.paths


def test__ci_line_sub_plot_output(data, data_plotter_path, plot_patch):

    ac.ci_data_plotters.plot_ci_line_subplot(
        ci_data=data,
        line_region="parallel_front_edge",
        output_path=data_plotter_path,
        output_format="png",
    )

    assert data_plotter_path + "ci_data_line.png" in plot_patch.paths


def test__ci_line_individuals__output_dependent_on_inputs(
    data, data_plotter_path, plot_patch
):

    ac.ci_data_plotters.plot_ci_data_line_individual(
        ci_data=data,
        line_region="parallel_front_edge",
        should_plot_image=True,
        should_plot_ci_pre_cti=True,
        output_path=data_plotter_path,
        output_format="png",
    )

    assert data_plotter_path + "ci_image_line.png" in plot_patch.paths

    assert data_plotter_path + "ci_noise_map_line.png" not in plot_patch.paths

    assert data_plotter_path + "ci_pre_cti_line.png" in plot_patch.paths

    assert data_plotter_path + "ci_signal_to_noise_map_line.png" not in plot_patch.paths


def test__image_line_is_output(data, mask, data_plotter_path, plot_patch):

    ac.ci_data_plotters.plot_image_line(
        ci_data=data,
        line_region="parallel_front_edge",
        mask=mask,
        output_path=data_plotter_path,
        output_format="png",
    )

    assert data_plotter_path + "ci_image_line.png" in plot_patch.paths


def test__noise_map_line_is_output(data, mask, data_plotter_path, plot_patch):

    ac.ci_data_plotters.plot_noise_map_line(
        ci_data=data,
        line_region="parallel_front_edge",
        mask=mask,
        output_path=data_plotter_path,
        output_format="png",
    )

    assert data_plotter_path + "ci_noise_map_line.png" in plot_patch.paths


def test__ci_pre_cti_line_is_output(data, mask, data_plotter_path, plot_patch):

    ac.ci_data_plotters.plot_ci_pre_cti_line(
        ci_data=data,
        line_region="parallel_front_edge",
        mask=mask,
        output_path=data_plotter_path,
        output_format="png",
    )

    assert data_plotter_path + "ci_pre_cti_line.png" in plot_patch.paths


def test__signal_to_noise_map_line_is_output(data, mask, data_plotter_path, plot_patch):

    ac.ci_data_plotters.plot_signal_to_noise_map_line(
        ci_data=data,
        line_region="parallel_front_edge",
        mask=mask,
        output_path=data_plotter_path,
        output_format="png",
    )

    assert data_plotter_path + "ci_signal_to_noise_map_line.png" in plot_patch.paths


def test__ci_sub_plot_output(data, data_plotter_path, plot_patch):

    ac.ci_data_plotters.plot_ci_subplot(
        ci_data=data,
        extract_array_from_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=data_plotter_path,
        output_format="png",
    )

    assert data_plotter_path + "ci_data.png" in plot_patch.paths


def test__ci_individuals__output_dependent_on_inputs(
    data, data_plotter_path, plot_patch
):
    ac.ci_data_plotters.plot_ci_data_individual(
        ci_data=data,
        extract_array_from_mask=True,
        should_plot_image=True,
        should_plot_ci_pre_cti=True,
        output_path=data_plotter_path,
        output_format="png",
    )

    assert data_plotter_path + "ci_image.png" in plot_patch.paths

    assert data_plotter_path + "ci_noise_map.png" not in plot_patch.paths

    assert data_plotter_path + "ci_pre_cti.png" in plot_patch.paths

    assert data_plotter_path + "ci_signal_to_noise_map.png" not in plot_patch.paths


@pytest.fixture(name="data_extracted")
def make_ci_data_extracted(data, mask):
    return ac.CIDataMasked(
        image=data.image,
        noise_map=data.noise_map,
        ci_pre_cti=data.ci_pre_cti,
        mask=mask,
        ci_pattern=data.ci_pattern,
        ci_frame=data.ci_frame,
    )


def test__plot_ci_data_for_phase(data_extracted, data_plotter_path, plot_patch):

    ac.ci_data_plotters.plot_ci_data_for_phase(
        ci_datas_extracted=[data_extracted],
        extract_array_from_mask=True,
        should_plot_as_subplot=True,
        should_plot_image=True,
        should_plot_noise_map=False,
        should_plot_ci_pre_cti=True,
        should_plot_signal_to_noise_map=False,
        should_plot_parallel_front_edge_line=True,
        should_plot_parallel_trails_line=False,
        should_plot_serial_front_edge_line=True,
        should_plot_serial_trails_line=False,
        visualize_path=data_plotter_path,
    )

    assert data_plotter_path + "/ci_image_10/structures/ci_data.png" in plot_patch.paths
    assert (
        data_plotter_path + "/ci_image_10/structures/ci_image.png" in plot_patch.paths
    )
    assert (
        data_plotter_path + "/ci_image_10/structures/ci_noise_map.png"
        not in plot_patch.paths
    )
    assert (
        data_plotter_path + "/ci_image_10/structures/ci_pre_cti.png" in plot_patch.paths
    )
    assert (
        data_plotter_path + "/ci_image_10/structures/ci_signal_to_noise_map.png"
        not in plot_patch.paths
    )

    assert (
        data_plotter_path + "/ci_image_10/parallel_front_edge/ci_data_line.png"
        in plot_patch.paths
    )
    assert (
        data_plotter_path + "/ci_image_10/parallel_front_edge/ci_image_line.png"
        in plot_patch.paths
    )
    assert (
        data_plotter_path + "/ci_image_10/parallel_front_edge/ci_noise_map_line.png"
        not in plot_patch.paths
    )
    assert (
        data_plotter_path + "/ci_image_10/parallel_front_edge/ci_pre_cti_line.png"
        in plot_patch.paths
    )
    assert (
        data_plotter_path
        + "/ci_image_10/parallel_front_edge/ci_signal_to_noise_map_line.png"
        not in plot_patch.paths
    )

    assert (
        data_plotter_path + "/ci_image_10/serial_front_edge/ci_data_line.png"
        in plot_patch.paths
    )
    assert (
        data_plotter_path + "/ci_image_10/serial_front_edge/ci_image_line.png"
        in plot_patch.paths
    )
    assert (
        data_plotter_path + "/ci_image_10/serial_front_edge/ci_noise_map_line.png"
        not in plot_patch.paths
    )
    assert (
        data_plotter_path + "/ci_image_10/serial_front_edge/ci_pre_cti_line.png"
        in plot_patch.paths
    )
    assert (
        data_plotter_path
        + "/ci_image_10/serial_front_edge/ci_signal_to_noise_map_line.png"
        not in plot_patch.paths
    )
