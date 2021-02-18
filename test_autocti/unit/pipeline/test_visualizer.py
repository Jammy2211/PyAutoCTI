import os
import shutil
from os import path

import pytest
import autofit as af
from autocti.pipeline import visualizer as vis
from autoconf import conf

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_visualizer_plotter_setup():
    return path.join("{}".format(directory), "files", "plot", "visualizer")


@pytest.fixture()
def set_config_path(plot_path):

    conf.instance.push(
        new_path=path.join(directory, "unit", "pipeline", "config"),
        output_path=path.join(plot_path),
    )


class TestVisualizer:
    def test__visualizes_imaging_using_configs(
        self, masked_ci_imaging_7x7, plot_path, plot_patch
    ):

        if path.exists(plot_path):
            shutil.rmtree(plot_path)

        masked_ci_imaging_7x7.cosmic_ray_map[0, 0] = 1

        visualizer = vis.Visualizer(visualize_path=plot_path)

        visualizer.visualize_ci_imaging(ci_imaging=masked_ci_imaging_7x7)

        plot_path = path.join(plot_path, "ci_imaging_0")

        assert path.join(plot_path, "subplot_ci_imaging.png") in plot_patch.paths
        assert path.join(plot_path, "image.png") in plot_patch.paths
        assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
        assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths
        assert path.join(plot_path, "ci_pre_cti.png") in plot_patch.paths
        assert path.join(plot_path, "cosmic_ray_map.png") in plot_patch.paths

    def test__visualizes_imaging_lines_using_configs(
        self, masked_ci_imaging_7x7, plot_path, plot_patch
    ):

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = vis.Visualizer(visualize_path=plot_path)

        visualizer.visualize_ci_imaging_lines(
            ci_imaging=masked_ci_imaging_7x7, line_region="parallel_front_edge"
        )

        plot_path = path.join(plot_path, "ci_imaging_0")

        assert (
            path.join(plot_path, "subplot_1d_ci_parallel_front_edge.png")
            in plot_patch.paths
        )
        assert path.join(plot_path, "image_parallel_front_edge.png") in plot_patch.paths
        assert (
            path.join(plot_path, "noise_map_parallel_front_edge.png")
            not in plot_patch.paths
        )
        assert (
            path.join(plot_path, "signal_to_noise_map_parallel_front_edge.png")
            not in plot_patch.paths
        )
        assert (
            path.join(plot_path, "ci_pre_cti_parallel_front_edge.png")
            in plot_patch.paths
        )

    def test___visualizes_fit_using_configs(
        self, masked_ci_imaging_7x7, ci_fit_7x7, plot_path, plot_patch
    ):

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = vis.Visualizer(visualize_path=plot_path)

        visualizer.visualize_ci_fit(fit=ci_fit_7x7, during_analysis=True)

        plot_path = path.join(plot_path, "fit_ci_imaging_0")

        assert path.join(plot_path, "subplot_ci_fit.png") in plot_patch.paths
        assert path.join(plot_path, "image.png") in plot_patch.paths
        assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
        assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths
        assert path.join(plot_path, "ci_pre_cti.png") in plot_patch.paths
        assert path.join(plot_path, "ci_post_cti.png") in plot_patch.paths
        assert path.join(plot_path, "residual_map.png") not in plot_patch.paths
        assert path.join(plot_path, "normalized_residual_map.png") in plot_patch.paths
        assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths

        plot_patch.paths = []

        visualizer.visualize_ci_fit(fit=ci_fit_7x7, during_analysis=False)

        assert path.join(plot_path, "subplot_ci_fit.png") in plot_patch.paths
        assert path.join(plot_path, "image.png") in plot_patch.paths
        assert path.join(plot_path, "noise_map.png") in plot_patch.paths
        assert path.join(plot_path, "signal_to_noise_map.png") in plot_patch.paths
        assert path.join(plot_path, "ci_pre_cti.png") in plot_patch.paths
        assert path.join(plot_path, "ci_post_cti.png") in plot_patch.paths
        assert path.join(plot_path, "residual_map.png") in plot_patch.paths
        assert path.join(plot_path, "normalized_residual_map.png") in plot_patch.paths
        assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths

    def test___visualizes_fit_lines_using_configs(
        self, masked_ci_imaging_7x7, ci_fit_7x7, plot_path, plot_patch
    ):

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = vis.Visualizer(visualize_path=plot_path)

        visualizer.visualize_ci_fit_lines(
            fit=ci_fit_7x7, line_region="parallel_front_edge", during_analysis=True
        )

        plot_path = path.join(plot_path, "fit_ci_imaging_0")

        assert (
            path.join(plot_path, "subplot_1d_ci_fit_parallel_front_edge.png")
            in plot_patch.paths
        )
        assert path.join(plot_path, "image_parallel_front_edge.png") in plot_patch.paths
        assert (
            path.join(plot_path, "noise_parallel_front_edge.png")
            not in plot_patch.paths
        )
        assert (
            path.join(plot_path, "signal_to_noise_map_parallel_front_edge.png")
            not in plot_patch.paths
        )
        assert (
            path.join(plot_path, "ci_pre_cti_parallel_front_edge.png")
            in plot_patch.paths
        )
        assert (
            path.join(plot_path, "ci_post_cti_parallel_front_edge.png")
            in plot_patch.paths
        )
        assert (
            path.join(plot_path, "residual_map_parallel_front_edge.png")
            not in plot_patch.paths
        )
        assert (
            path.join(plot_path, "normalized_residual_map_parallel_front_edge.png")
            in plot_patch.paths
        )
        assert (
            path.join(plot_path, "chi_squared_map_parallel_front_edge.png")
            in plot_patch.paths
        )

        plot_patch.paths = []

        visualizer.visualize_ci_fit_lines(
            fit=ci_fit_7x7, line_region="parallel_front_edge", during_analysis=False
        )

        assert (
            path.join(plot_path, "subplot_1d_ci_fit_parallel_front_edge.png")
            in plot_patch.paths
        )
        assert path.join(plot_path, "image_parallel_front_edge.png") in plot_patch.paths
        assert (
            path.join(plot_path, "noise_map_parallel_front_edge.png")
            in plot_patch.paths
        )
        assert (
            path.join(plot_path, "signal_to_noise_map_parallel_front_edge.png")
            in plot_patch.paths
        )
        assert (
            path.join(plot_path, "ci_pre_cti_parallel_front_edge.png")
            in plot_patch.paths
        )
        assert (
            path.join(plot_path, "ci_post_cti_parallel_front_edge.png")
            in plot_patch.paths
        )
        assert (
            path.join(plot_path, "residual_map_parallel_front_edge.png")
            in plot_patch.paths
        )
        assert (
            path.join(plot_path, "normalized_residual_map_parallel_front_edge.png")
            in plot_patch.paths
        )
        assert (
            path.join(plot_path, "chi_squared_map_parallel_front_edge.png")
            in plot_patch.paths
        )

    # def test___visualizes_multiple_ci_fits_subplot__using_configs(
    #     self, masked_ci_imaging_7x7, ci_fit_7x7, plot_path, plot_patch
    # ):
    #
    #     if os.path.exists(plot_path):
    #         shutil.rmtree(plot_path)
    #
    #     visualizer = vis.Visualizer(visualize_path=plot_path)
    #
    #     visualizer.visualize_multiple_ci_fits_subplots(fits=[ci_fit_7x7])
    #
    #     assert (
    #         path.join(plot_path, "image", "subplots", "subplot_residual_maps.png")
    #         in plot_patch.paths
    #     )
    #     assert (
    #         path.join(
    #             plot_path, "image", "subplots", "subplot_normalized_residual_maps.png"
    #         )
    #         in plot_patch.paths
    #     )
    #     assert (
    #         path.join(plot_path, "image", "subplots", "subplot_chi_squared_maps.png")
    #         in plot_patch.paths
    #     )
    #
    # def test___visualizes_multiple_ci_fits_lines_subplot__using_configs(
    #     self, masked_ci_imaging_7x7, ci_fit_7x7, plot_path, plot_patch
    # ):
    #
    #     if os.path.exists(plot_path):
    #         shutil.rmtree(plot_path)
    #
    #     visualizer = vis.PhaseCIImagingVisualizer(masked_dataset=masked_ci_imaging_7x7)
    #
    #     visualizer.visualize_multiple_ci_fits_subplots_lines(
    #         paths=af.Paths(), fits=[ci_fit_7x7], line_region="parallel_front_edge"
    #     )
    #
    #     assert (
    #         path.join(
    #             plot_path,
    #             "image",
    #             "subplots",
    #             "subplot_residual_maps_lines_parallel_front_edge.png",
    #         )
    #         in plot_patch.paths
    #     )
    #     assert (
    #         path.join(
    #             plot_path,
    #             "image",
    #             "subplots",
    #             "subplot_normalized_residual_maps_lines_parallel_front_edge.png",
    #         )
    #         in plot_patch.paths
    #     )
    #     assert (
    #         path.join(
    #             plot_path,
    #             "image",
    #             "subplots",
    #             "subplot_chi_squared_maps_lines_parallel_front_edge.png",
    #         )
    #         in plot_patch.paths
    #     )
