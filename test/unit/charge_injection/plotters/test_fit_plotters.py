import numpy as np

import autocti as ac
from test.unit.mock.mock import MockPattern, MockCIFrame

from test.fixtures import make_plot_patch
import os
import pytest


@pytest.fixture(name="fit_plotter_path")
def make_ci_data_plotter_setup():
    return "{}/../../test_files/plotting/fit/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


@pytest.fixture(name="mask")
def make_mask():
    return ac.Mask.empty_for_shape(shape=(2, 2))


@pytest.fixture(name="image")
def make_image():
    return np.ones((2, 2))


@pytest.fixture(name="noise_map")
def make_noise_map():
    return 2.0 * np.ones((2, 2))


@pytest.fixture(name="ci_pre_cti")
def make_ci_pre_cti():
    return 3.0 * np.ones((2, 2))


@pytest.fixture(name="ci_data_fit")
def make_ci_data_fit(image, noise_map, mask, ci_pre_cti):
    return ac.MaskedCIData(
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
def make_fit(ci_data_fit, cti_params, cti_settings):
    return ac.CIFit(
        masked_ci_data=ci_data_fit, cti_params=cti_params, cti_settings=cti_settings
    )


def test__image_is_output(fit, mask, fit_plotter_path, plot_patch):
    ac.fit_plotters.plot_image(
        fit=fit,
        mask=mask,
        extract_array_from_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_path,
        output_format="png",
    )
    assert fit_plotter_path + "fit_image.png" in plot_patch.paths


def test__noise_map_is_output(fit, mask, fit_plotter_path, plot_patch):
    ac.fit_plotters.plot_noise_map(
        fit=fit,
        mask=mask,
        extract_array_from_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_path,
        output_format="png",
    )
    assert fit_plotter_path + "fit_noise_map.png" in plot_patch.paths


def test__signal_to_noise_map_is_output(fit, mask, fit_plotter_path, plot_patch):
    ac.fit_plotters.plot_signal_to_noise_map(
        fit=fit,
        mask=mask,
        extract_array_from_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_path,
        output_format="png",
    )
    assert fit_plotter_path + "fit_signal_to_noise_map.png" in plot_patch.paths


def test__ci_pre_cti_is_output(fit, mask, fit_plotter_path, plot_patch):
    ac.fit_plotters.plot_ci_pre_cti(
        fit=fit,
        mask=mask,
        extract_array_from_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_path,
        output_format="png",
    )
    assert fit_plotter_path + "fit_ci_pre_cti.png" in plot_patch.paths


def test__ci_post_cti_is_output(fit, mask, fit_plotter_path, plot_patch):
    ac.fit_plotters.plot_ci_post_cti(
        fit=fit,
        mask=mask,
        extract_array_from_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_path,
        output_format="png",
    )
    assert fit_plotter_path + "fit_ci_post_cti.png" in plot_patch.paths


def test__residual_map_is_output(fit, mask, fit_plotter_path, plot_patch):
    ac.fit_plotters.plot_residual_map(
        fit=fit,
        mask=mask,
        extract_array_from_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_path,
        output_format="png",
    )
    assert fit_plotter_path + "fit_residual_map.png" in plot_patch.paths


def test__chi_squared_map_is_output(fit, mask, fit_plotter_path, plot_patch):
    ac.fit_plotters.plot_chi_squared_map(
        fit=fit,
        mask=mask,
        extract_array_from_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_path,
        output_format="png",
    )
    assert fit_plotter_path + "fit_chi_squared_map.png" in plot_patch.paths


def test__image_line_is_output(fit, mask, fit_plotter_path, plot_patch):

    ac.fit_plotters.plot_image_line(
        fit=fit,
        line_region="parallel_front_edge",
        mask=mask,
        output_path=fit_plotter_path,
        output_format="png",
    )

    assert fit_plotter_path + "fit_image_line.png" in plot_patch.paths


def test__noise_map_line_is_output(fit, mask, fit_plotter_path, plot_patch):

    ac.fit_plotters.plot_noise_map_line(
        fit=fit,
        line_region="parallel_front_edge",
        mask=mask,
        output_path=fit_plotter_path,
        output_format="png",
    )

    assert fit_plotter_path + "fit_noise_map_line.png" in plot_patch.paths


def test__signal_to_noise_map_line_is_output(fit, mask, fit_plotter_path, plot_patch):

    ac.fit_plotters.plot_signal_to_noise_map_line(
        fit=fit,
        line_region="parallel_front_edge",
        mask=mask,
        output_path=fit_plotter_path,
        output_format="png",
    )

    assert fit_plotter_path + "fit_signal_to_noise_map_line.png" in plot_patch.paths


def test__ci_pre_cti_line_is_output(fit, mask, fit_plotter_path, plot_patch):

    ac.fit_plotters.plot_ci_pre_cti_line(
        fit=fit,
        line_region="parallel_front_edge",
        mask=mask,
        output_path=fit_plotter_path,
        output_format="png",
    )

    assert fit_plotter_path + "fit_ci_pre_cti_line.png" in plot_patch.paths


def test__ci_post_cti_line_is_output(fit, mask, fit_plotter_path, plot_patch):

    ac.fit_plotters.plot_ci_post_cti_line(
        fit=fit,
        line_region="parallel_front_edge",
        mask=mask,
        output_path=fit_plotter_path,
        output_format="png",
    )

    assert fit_plotter_path + "fit_ci_post_cti_line.png" in plot_patch.paths


def test__residual_map_line_is_output(fit, mask, fit_plotter_path, plot_patch):

    ac.fit_plotters.plot_residual_map_line(
        fit=fit,
        line_region="parallel_front_edge",
        mask=mask,
        output_path=fit_plotter_path,
        output_format="png",
    )

    print(plot_patch.paths)

    assert fit_plotter_path + "fit_residual_map_line.png" in plot_patch.paths


def test__chi_squared_map_line_is_output(fit, mask, fit_plotter_path, plot_patch):

    ac.fit_plotters.plot_chi_squared_map_line(
        fit=fit,
        line_region="parallel_front_edge",
        mask=mask,
        output_path=fit_plotter_path,
        output_format="png",
    )

    assert fit_plotter_path + "fit_chi_squared_map_line.png" in plot_patch.paths


@pytest.fixture(name="ci_data_fit_hyper")
def make_ci_data_fit_hyper(image, noise_map, mask, ci_pre_cti):
    return ac.MaskedCIHyperData(
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
    return ac.CIFit(
        masked_ci_data=ci_data_fit_hyper,
        cti_params=cti_params,
        cti_settings=cti_settings,
        hyper_noise_scalars=hyper_noise_scalars,
    )


def test__noise_scaling_map_is_output(fit_hyper, mask, fit_plotter_path, plot_patch):

    ac.fit_plotters.plot_noise_scaling_maps(
        fit_hyper=fit_hyper,
        mask=mask,
        extract_array_from_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_path,
        output_format="png",
    )

    assert fit_plotter_path + "fit_noise_scaling_maps.png" in plot_patch.paths
