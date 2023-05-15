from os import path
import pytest
from autocti import plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_fit_ci_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "fit"
    )


def test__individual_attribute_plots__all_plot_correctly(
    fit_1d_7, plot_path, plot_patch
):
    fit_ci_plotter = aplt.FitDataset1DPlotter(
        fit=fit_1d_7,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    fit_ci_plotter.figures_1d(
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

    fit_ci_plotter.figures_1d(
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


def test__fit_1d_subplots_are_output(fit_1d_7, plot_path, plot_patch):
    fit_ci_plotter = aplt.FitDataset1DPlotter(
        fit=fit_1d_7,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    fit_ci_plotter.subplot_fit()
    assert path.join(plot_path, "subplot_fit.png") in plot_patch.paths
