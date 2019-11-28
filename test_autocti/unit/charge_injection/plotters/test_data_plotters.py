import numpy as np

import autocti as ac
from test import MockCIFrame

import os
import pytest


@pytest.fixture(name="data_plotter_path")
def make_data_plotter_setup():
    return "{}/../../test_files/plotting/dataset/".format(
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


@pytest.fixture(name="ci_frame")
def make_ci_frame():
    return MockCIFrame(value=3.0)


def test__image_is_output(image, mask, data_plotter_path, plot_patch):

    ac.data_plotters.plot_image(
        image=image,
        mask=mask,
        extract_array_from_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=data_plotter_path,
        output_format="png",
    )

    assert data_plotter_path + "image.png" in plot_patch.paths


def test__noise_map_is_output(noise_map, mask, data_plotter_path, plot_patch):
    ac.data_plotters.plot_noise_map(
        noise_map=noise_map,
        mask=mask,
        extract_array_from_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=data_plotter_path,
        output_format="png",
    )
    assert data_plotter_path + "noise_map.png" in plot_patch.paths


def test__ci_pre_cti_is_output(ci_pre_cti, mask, data_plotter_path, plot_patch):
    ac.data_plotters.plot_ci_pre_cti(
        ci_pre_cti=ci_pre_cti,
        mask=mask,
        extract_array_from_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=data_plotter_path,
        output_format="png",
    )
    assert data_plotter_path + "ci_pre_cti.png" in plot_patch.paths


def test__signal_to_noise_map_is_output(
    image, noise_map, mask, data_plotter_path, plot_patch
):
    ac.data_plotters.plot_signal_to_noise_map(
        signal_to_noise_map=image / noise_map,
        mask=mask,
        extract_array_from_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=data_plotter_path,
        output_format="png",
    )
    assert data_plotter_path + "signal_to_noise_map.png" in plot_patch.paths


def test__image_line_is_output(image, mask, ci_frame, data_plotter_path, plot_patch):

    ac.data_plotters.plot_image_line(
        image=image,
        line_region="parallel_front_edge",
        mask=mask,
        ci_frame=ci_frame,
        output_path=data_plotter_path,
        output_format="png",
    )

    assert data_plotter_path + "image_line.png" in plot_patch.paths


def test__noise_map_line_is_output(
    noise_map, mask, ci_frame, data_plotter_path, plot_patch
):

    ac.data_plotters.plot_noise_map_line(
        noise_map=noise_map,
        line_region="parallel_front_edge",
        mask=mask,
        ci_frame=ci_frame,
        output_path=data_plotter_path,
        output_format="png",
    )

    assert data_plotter_path + "noise_map_line.png" in plot_patch.paths


def test__ci_pre_cti_line_is_output(
    ci_pre_cti, mask, ci_frame, data_plotter_path, plot_patch
):

    ac.data_plotters.plot_ci_pre_cti_line(
        ci_pre_cti=ci_pre_cti,
        line_region="parallel_front_edge",
        ci_frame=ci_frame,
        mask=mask,
        output_path=data_plotter_path,
        output_format="png",
    )

    assert data_plotter_path + "ci_pre_cti_line.png" in plot_patch.paths


def test__signal_to_noise_map_line_is_output(
    image, noise_map, mask, ci_frame, data_plotter_path, plot_patch
):

    ac.data_plotters.plot_signal_to_noise_map_line(
        signal_to_noise_map=image / noise_map,
        ci_frame=ci_frame,
        line_region="parallel_front_edge",
        mask=mask,
        output_path=data_plotter_path,
        output_format="png",
    )

    assert data_plotter_path + "signal_to_noise_map_line.png" in plot_patch.paths
