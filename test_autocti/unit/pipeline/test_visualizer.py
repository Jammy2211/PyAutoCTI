from autocti.pipeline import visualizer as vis

import os
import pytest
from os import path
import shutil
from autofit import conf

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_visualizer_plotter_setup():
    return "{}/files/plot/visualizer/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "files/plotter"), path.join(directory, "output")
    )


class TestPhaseCIImagingVisualizer:
    def test__visualizes_imaging_using_configs(
        self, masked_ci_imaging_7x7, plot_path, plot_patch
    ):

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = vis.PhaseCIImagingVisualizer(
            masked_dataset=masked_ci_imaging_7x7, image_path=plot_path
        )

        visualizer.visualize_ci_imaging()

        assert plot_path + "subplots/subplot_ci_imaging.png" in plot_patch.paths
        assert plot_path + "ci_imaging/image.png" in plot_patch.paths
        assert plot_path + "ci_imaging/noise_map.png" not in plot_patch.paths
        assert plot_path + "ci_imaging/cosmic_ray_map.png" in plot_patch.paths
        assert plot_path + "ci_imaging/signal_to_noise_map.png" not in plot_patch.paths

    def test___visualizes_fit_using_configs(
        self, masked_ci_imaging_7x7, ci_fit_7x7, plot_path, plot_patch
    ):

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = vis.PhaseCIImagingVisualizer(
            masked_dataset=masked_ci_imaging_7x7, image_path=plot_path
        )

        visualizer.visualize_fit(fit=ci_fit_7x7, during_analysis=True)

        assert plot_path + "subplots/subplot_ci_fit.png" in plot_patch.paths
        assert plot_path + "fit_ci_imaging/image.png" in plot_patch.paths
        assert plot_path + "fit_ci_imaging/noise_map.png" not in plot_patch.paths
        assert (
            plot_path + "fit_ci_imaging/signal_to_noise_map.png" not in plot_patch.paths
        )
        assert plot_path + "fit_ci_imaging/ci_pre_cti.png" in plot_patch.paths
        assert plot_path + "fit_ci_imaging/ci_post_cti.png" in plot_patch.paths
        assert plot_path + "fit_ci_imaging/residual_map.png" not in plot_patch.paths
        assert (
            plot_path + "fit_ci_imaging/normalized_residual_map.png" in plot_patch.paths
        )
        assert plot_path + "fit_ci_imaging/chi_squared_map.png" in plot_patch.paths

        visualizer.visualize_fit(fit=ci_fit_7x7, during_analysis=False)

        assert plot_path + "subplots/subplot_ci_fit.png" in plot_patch.paths
        assert plot_path + "fit_ci_imaging/image.png" in plot_patch.paths
        assert plot_path + "fit_ci_imaging/noise_map.png" in plot_patch.paths
        assert plot_path + "fit_ci_imaging/signal_to_noise_map.png" in plot_patch.paths
        assert plot_path + "fit_ci_imaging/ci_pre_cti.png" in plot_patch.paths
        assert plot_path + "fit_ci_imaging/ci_post_cti.png" in plot_patch.paths
        assert plot_path + "fit_ci_imaging/residual_map.png" in plot_patch.paths
        assert (
            plot_path + "fit_ci_imaging/normalized_residual_map.png" in plot_patch.paths
        )
        assert plot_path + "fit_ci_imaging/chi_squared_map.png" in plot_patch.paths
