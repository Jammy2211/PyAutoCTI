import numpy as np

import autocti as ac
from test_autocti.mock.mock import MockPattern, MockCIFrame

import os
import pytest


@pytest.fixture(name="ci_fit_plotter_path")
def make_ci_data_plotter_setup():
    return "{}/../../test_files/plotting/ci_fit/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


@pytest.fixture(name="mask")
def make_mask():
    return ac.Mask.unmasked(shape_2d=(2, 2))


@pytest.fixture(name="image")
def make_image():
    return np.ones((2, 2))


@pytest.fixture(name="noise_map")
def make_noise_map():
    return 2.0 * np.ones((2, 2))


@pytest.fixture(name="ci_pre_cti")
def make_ci_pre_cti():
    return 3.0 * np.ones((2, 2))


@pytest.fixture(name="ci_frame")
def make_ci_frame():
    return MockCIFrame(value=3.0)


@pytest.fixture(name="ci_data_masked")
def make_ci_data_fit(image, noise_map, mask, ci_pre_cti):
    return ac.CIMaskedImaging(
        image=image,
        noise_map=noise_map,
        ci_pre_cti=ci_pre_cti,
        mask=mask,
        ci_pattern=MockPattern(),
        ci_frame=MockCIFrame(value=3.0),
    )


@pytest.fixture(name="cti_settings")
def make_cti_settings():
    parallel_settings = ac.Settings(well_depth=0, niter=1, express=1, n_levels=2000)
    return ac.ArcticSettings(neomode="NEO", parallel=parallel_settings)


@pytest.fixture(name="cti_params")
def make_cti_params():
    parallel_1_species = ac.Species(trap_density=0.1, trap_lifetime=1.0)
    return ac.ArcticParams(parallel_species=parallel_1_species)


@pytest.fixture(name="fit")
def make_fit(ci_data_masked, cti_params, cti_settings):
    return ac.CIImagingFit(
        ci_masked_imaging=ci_data_masked,
        cti_params=cti_params,
        cti_settings=cti_settings,
    )


def test__ci_fit_subplot_is_output(fit, ci_fit_plotter_path, plot_patch):

    autocti.plotters.plotters.ci_fit_plotters.plot_fit_subplot(
        fit=fit,
        extract_array_from_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=ci_fit_plotter_path,
        output_format="png",
    )

    assert ci_fit_plotter_path + "ci_fit.png" in plot_patch.paths


def test__ci_fit_residual_maps_subplot_is_output(fit, ci_fit_plotter_path, plot_patch):

    autocti.plotters.plotters.ci_fit_plotters.plot_fit_residual_maps_subplot(
        fits=[fit],
        extract_array_from_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=ci_fit_plotter_path,
        output_format="png",
    )

    assert ci_fit_plotter_path + "ci_fits_residual_maps.png" in plot_patch.paths

    autocti.plotters.plotters.ci_fit_plotters.plot_fit_residual_maps_lines_subplot(
        fits=[fit],
        line_region="parallel_front_edge",
        output_path=ci_fit_plotter_path,
        output_format="png",
    )

    assert ci_fit_plotter_path + "ci_fits_residual_maps_lines.png" in plot_patch.paths


def test__ci_fit_chi_squareds_subplot_is_output(fit, ci_fit_plotter_path, plot_patch):

    autocti.plotters.plotters.ci_fit_plotters.plot_fit_chi_squared_maps_subplot(
        fits=[fit],
        extract_array_from_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=ci_fit_plotter_path,
        output_format="png",
    )

    assert ci_fit_plotter_path + "ci_fits_chi_squared_maps.png" in plot_patch.paths

    autocti.plotters.plotters.ci_fit_plotters.plot_fit_chi_squared_maps_lines_subplot(
        fits=[fit],
        line_region="parallel_front_edge",
        output_path=ci_fit_plotter_path,
        output_format="png",
    )

    assert (
        ci_fit_plotter_path + "ci_fits_chi_squared_maps_lines.png" in plot_patch.paths
    )


def test__fit_individuals__depedent_on_input(fit, ci_fit_plotter_path, plot_patch):

    autocti.plotters.plotters.ci_fit_plotters.plot_fit_individuals(
        fit=fit,
        should_plot_image=True,
        should_plot_noise_map=False,
        should_plot_signal_to_noise_map=False,
        should_plot_ci_pre_cti=True,
        should_plot_ci_post_cti=True,
        should_plot_chi_squared_map=True,
        output_path=ci_fit_plotter_path,
        output_format="png",
    )

    assert ci_fit_plotter_path + "fit_image.png" in plot_patch.paths

    assert ci_fit_plotter_path + "fit_noise_map.png" not in plot_patch.paths

    assert ci_fit_plotter_path + "fit_signal_to_noise_map.png" not in plot_patch.paths

    assert ci_fit_plotter_path + "fit_ci_pre_cti.png" in plot_patch.paths

    assert ci_fit_plotter_path + "fit_ci_post_cti.png" in plot_patch.paths

    assert ci_fit_plotter_path + "fit_residual_map.png" not in plot_patch.paths

    assert ci_fit_plotter_path + "fit_chi_squared_map.png" in plot_patch.paths


def test__fit_individuals_line__depedent_on_input(
    fit, ci_frame, ci_fit_plotter_path, plot_patch
):

    autocti.plotters.plotters.ci_fit_plotters.plot_fit_line_individuals(
        fit=fit,
        line_region="parallel_front_edge",
        should_plot_image=True,
        should_plot_noise_map=False,
        should_plot_signal_to_noise_map=False,
        should_plot_ci_pre_cti=True,
        should_plot_ci_post_cti=True,
        should_plot_chi_squared_map=True,
        output_path=ci_fit_plotter_path,
        output_format="png",
    )

    assert ci_fit_plotter_path + "fit_image_line.png" in plot_patch.paths

    assert ci_fit_plotter_path + "fit_noise_map_line.png" not in plot_patch.paths

    assert (
        ci_fit_plotter_path + "fit_signal_to_noise_map_line.png" not in plot_patch.paths
    )

    assert ci_fit_plotter_path + "fit_ci_pre_cti_line.png" in plot_patch.paths

    assert ci_fit_plotter_path + "fit_ci_post_cti_line.png" in plot_patch.paths

    assert ci_fit_plotter_path + "fit_residual_map_line.png" not in plot_patch.paths

    assert ci_fit_plotter_path + "fit_chi_squared_map_line.png" in plot_patch.paths


def test__plot_ci_fit_for_phase(fit, ci_fit_plotter_path, plot_patch):

    autocti.plotters.plotters.ci_fit_plotters.plot_ci_fit_for_phase(
        fits=[fit],
        during_analysis=False,
        extract_array_from_mask=True,
        should_plot_all_at_end_png=False,
        should_plot_all_at_end_fits=False,
        should_plot_as_subplot=True,
        should_plot_residual_maps_subplot=True,
        should_plot_chi_squared_maps_subplot=False,
        should_plot_image=True,
        should_plot_noise_map=False,
        should_plot_ci_pre_cti=True,
        should_plot_signal_to_noise_map=False,
        should_plot_ci_post_cti=False,
        should_plot_residual_map=True,
        should_plot_chi_squared_map=False,
        should_plot_noise_scaling_maps=False,
        should_plot_parallel_front_edge_line=True,
        should_plot_parallel_trails_line=False,
        should_plot_serial_front_edge_line=True,
        should_plot_serial_trails_line=False,
        visualize_path=ci_fit_plotter_path,
    )

    assert (
        ci_fit_plotter_path + "/ci_image_10/structures/ci_fit.png" in plot_patch.paths
    )
    assert (
        ci_fit_plotter_path + "/ci_image_10/structures/fit_image.png"
        in plot_patch.paths
    )
    assert (
        ci_fit_plotter_path + "/ci_image_10/structures/fit_noise_map.png"
        not in plot_patch.paths
    )
    assert (
        ci_fit_plotter_path + "/ci_image_10/structures/fit_ci_pre_cti.png"
        in plot_patch.paths
    )
    assert (
        ci_fit_plotter_path + "/ci_image_10/structures/fit_signal_to_noise_map.png"
        not in plot_patch.paths
    )
    assert (
        ci_fit_plotter_path + "/ci_image_10/structures/fit_residual_map.png"
        in plot_patch.paths
    )
    assert (
        ci_fit_plotter_path + "/ci_image_10/structures/fit_chi_squared_map.png"
        not in plot_patch.paths
    )

    assert (
        ci_fit_plotter_path + "/ci_image_10/parallel_front_edge/ci_fit_line.png"
        in plot_patch.paths
    )
    assert (
        ci_fit_plotter_path + "/ci_image_10/parallel_front_edge/fit_image_line.png"
        in plot_patch.paths
    )
    assert (
        ci_fit_plotter_path + "/ci_image_10/parallel_front_edge/fit_noise_map_line.png"
        not in plot_patch.paths
    )
    assert (
        ci_fit_plotter_path + "/ci_image_10/parallel_front_edge/fit_ci_pre_cti_line.png"
        in plot_patch.paths
    )
    assert (
        ci_fit_plotter_path
        + "/ci_image_10/parallel_front_edge/fit_signal_to_noise_map_line.png"
        not in plot_patch.paths
    )
    assert (
        ci_fit_plotter_path
        + "/ci_image_10/parallel_front_edge/fit_residual_map_line.png"
        in plot_patch.paths
    )
    assert (
        ci_fit_plotter_path
        + "/ci_image_10/parallel_front_edge/fit_chi_squared_map_line.png"
        not in plot_patch.paths
    )

    assert (
        ci_fit_plotter_path + "/ci_image_10/serial_front_edge/ci_fit_line.png"
        in plot_patch.paths
    )
    assert (
        ci_fit_plotter_path + "/ci_image_10/serial_front_edge/fit_image_line.png"
        in plot_patch.paths
    )
    assert (
        ci_fit_plotter_path + "/ci_image_10/serial_front_edge/fit_noise_map_line.png"
        not in plot_patch.paths
    )
    assert (
        ci_fit_plotter_path + "/ci_image_10/serial_front_edge/fit_ci_pre_cti_line.png"
        in plot_patch.paths
    )
    assert (
        ci_fit_plotter_path
        + "/ci_image_10/serial_front_edge/fit_signal_to_noise_map_line.png"
        not in plot_patch.paths
    )
    assert (
        ci_fit_plotter_path + "/ci_image_10/serial_front_edge/fit_residual_map_line.png"
        in plot_patch.paths
    )
    assert (
        ci_fit_plotter_path
        + "/ci_image_10/serial_front_edge/fit_chi_squared_map_line.png"
        not in plot_patch.paths
    )

    assert ci_fit_plotter_path + "/ci_fits_residual_maps.png" in plot_patch.paths
    assert ci_fit_plotter_path + "/ci_fits_chi_sqaured_maps.png" not in plot_patch.paths


@pytest.fixture(name="ci_data_fit_hyper")
def make_ci_data_fit_hyper(image, noise_map, mask, ci_pre_cti):
    return ac.CIMaskedImaging(
        image=image,
        noise_map=noise_map,
        ci_pre_cti=ci_pre_cti,
        mask=mask,
        ci_pattern=MockPattern(),
        ci_frame=MockCIFrame(value=3.0),
        noise_scaling_maps=[np.ones((2, 2)), 2.0 * np.ones((2, 2))],
    )


@pytest.fixture(name="hyper_noise_scalars")
def make_hyper_noise_scalars():
    return [
        ac.CIHyperNoiseScalar(scale_factor=1.0),
        ac.CIHyperNoiseScalar(scale_factor=2.0),
    ]


@pytest.fixture(name="fit_hyper")
def make_fit_hyper(ci_data_fit_hyper, cti_params, cti_settings, hyper_noise_scalars):
    return ac.CIImagingFit(
        ci_masked_imaging=ci_data_fit_hyper,
        cti_params=cti_params,
        cti_settings=cti_settings,
        hyper_noise_scalars=hyper_noise_scalars,
    )


def test__fit_individuals__fit_hyper_plots_noise_scaling_maps(
    fit, fit_hyper, ci_fit_plotter_path, plot_patch
):

    autocti.plotters.plotters.ci_fit_plotters.plot_fit_individuals(
        fit=fit,
        should_plot_noise_scaling_maps=True,
        output_path=ci_fit_plotter_path,
        output_format="png",
    )

    assert ci_fit_plotter_path + "fit_noise_scaling_maps.png" not in plot_patch.paths

    autocti.plotters.plotters.ci_fit_plotters.plot_fit_individuals(
        fit=fit_hyper,
        should_plot_noise_scaling_maps=True,
        output_path=ci_fit_plotter_path,
        output_format="png",
    )

    assert ci_fit_plotter_path + "fit_noise_scaling_maps.png" in plot_patch.paths
