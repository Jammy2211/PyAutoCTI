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
    return "{}/files/plot/visualizer".format(
        os.path.dirname(os.path.realpath(__file__))
    )


@pytest.fixture(autouse=True)
def set_config_path(plot_path):
    conf.instance.push(f"{directory}/config", output_path=plot_path)


class TestPhaseCIImagingVisualizer:
    def test__visualizes_imaging_using_configs(
        self, masked_ci_imaging_7x7, plot_path, plot_patch
    ):

        masked_ci_imaging_7x7.cosmic_ray_map[0, 0] = 1

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = vis.PhaseCIImagingVisualizer(masked_dataset=masked_ci_imaging_7x7)

        visualizer.visualize_ci_imaging(paths=af.Paths())

        assert f"{plot_path}/image//subplots/subplot_ci_imaging.png" in plot_patch.paths
        assert f"{plot_path}/image//ci_imaging/image.png" in plot_patch.paths
        assert f"{plot_path}/image//ci_imaging/noise_map.png" not in plot_patch.paths
        assert (
            f"{plot_path}/image//ci_imaging/signal_to_noise_map.png"
            not in plot_patch.paths
        )
        assert f"{plot_path}/image//ci_imaging/ci_pre_cti.png" in plot_patch.paths
        assert f"{plot_path}/image//ci_imaging/cosmic_ray_map.png" in plot_patch.paths

    def test__visualizes_imaging_lines_using_configs(
        self, masked_ci_imaging_7x7, plot_path, plot_patch
    ):

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = vis.PhaseCIImagingVisualizer(masked_dataset=masked_ci_imaging_7x7)

        visualizer.visualize_ci_imaging_lines(
            paths=af.Paths(), line_region="parallel_front_edge"
        )

        assert (
            f"{plot_path}/image//subplots/subplot_ci_lines_parallel_front_edge.png"
            in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//ci_imaging_parallel_front_edge/image_line.png"
            in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//ci_imaging_parallel_front_edge/noise_map_line.png"
            not in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//ci_imaging_parallel_front_edge/signal_to_noise_map_line.png"
            not in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//ci_imaging_parallel_front_edge/ci_pre_cti_line.png"
            in plot_patch.paths
        )

    def test___visualizes_fit_using_configs(
        self, masked_ci_imaging_7x7, ci_fit_7x7, plot_path, plot_patch
    ):

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = vis.PhaseCIImagingVisualizer(masked_dataset=masked_ci_imaging_7x7)

        visualizer.visualize_ci_fit(
            paths=af.Paths(), fit=ci_fit_7x7, during_analysis=True
        )

        assert f"{plot_path}/image//subplots/subplot_ci_fit.png" in plot_patch.paths
        assert f"{plot_path}/image//fit_ci_imaging/image.png" in plot_patch.paths
        assert (
            f"{plot_path}/image//fit_ci_imaging/noise_map.png" not in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//fit_ci_imaging/signal_to_noise_map.png"
            not in plot_patch.paths
        )
        assert f"{plot_path}/image//fit_ci_imaging/ci_pre_cti.png" in plot_patch.paths
        assert f"{plot_path}/image//fit_ci_imaging/ci_post_cti.png" in plot_patch.paths
        assert (
            f"{plot_path}/image//fit_ci_imaging/residual_map.png"
            not in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//fit_ci_imaging/normalized_residual_map.png"
            in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//fit_ci_imaging/chi_squared_map.png" in plot_patch.paths
        )

        visualizer.visualize_ci_fit(
            paths=af.Paths(), fit=ci_fit_7x7, during_analysis=False
        )

        assert f"{plot_path}/image//subplots/subplot_ci_fit.png" in plot_patch.paths
        assert f"{plot_path}/image//fit_ci_imaging/image.png" in plot_patch.paths
        assert f"{plot_path}/image//fit_ci_imaging/noise_map.png" in plot_patch.paths
        assert (
            f"{plot_path}/image//fit_ci_imaging/signal_to_noise_map.png"
            in plot_patch.paths
        )
        assert f"{plot_path}/image//fit_ci_imaging/ci_pre_cti.png" in plot_patch.paths
        assert f"{plot_path}/image//fit_ci_imaging/ci_post_cti.png" in plot_patch.paths
        assert f"{plot_path}/image//fit_ci_imaging/residual_map.png" in plot_patch.paths
        assert (
            f"{plot_path}/image//fit_ci_imaging/normalized_residual_map.png"
            in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//fit_ci_imaging/chi_squared_map.png" in plot_patch.paths
        )

    def test___visualizes_fit_lines_using_configs(
        self, masked_ci_imaging_7x7, ci_fit_7x7, plot_path, plot_patch
    ):

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = vis.PhaseCIImagingVisualizer(masked_dataset=masked_ci_imaging_7x7)

        visualizer.visualize_ci_fit_lines(
            paths=af.Paths(),
            fit=ci_fit_7x7,
            line_region="parallel_front_edge",
            during_analysis=True,
        )

        assert (
            f"{plot_path}/image//subplots/subplot_ci_fit_lines_parallel_front_edge.png"
            in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//fit_ci_imaging_parallel_front_edge/image_line.png"
            in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//fit_ci_imaging_parallel_front_edge/noise_map_line.png"
            not in plot_patch.paths
        )
        assert (
            plot_path
            + "fit_ci_imaging_parallel_front_edge/signal_to_noise_map_line.png"
            not in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//fit_ci_imaging_parallel_front_edge/ci_pre_cti_line.png"
            in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//fit_ci_imaging_parallel_front_edge/ci_post_cti_line.png"
            in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//fit_ci_imaging_parallel_front_edge/residual_map_line.png"
            not in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//fit_ci_imaging_parallel_front_edge/normalized_residual_map_line.png"
            in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//fit_ci_imaging_parallel_front_edge/chi_squared_map_line.png"
            in plot_patch.paths
        )

        visualizer.visualize_ci_fit_lines(
            paths=af.Paths(),
            fit=ci_fit_7x7,
            line_region="parallel_front_edge",
            during_analysis=False,
        )

        assert (
            f"{plot_path}/image//subplots/subplot_ci_fit_lines_parallel_front_edge.png"
            in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//fit_ci_imaging_parallel_front_edge/image_line.png"
            in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//fit_ci_imaging_parallel_front_edge/noise_map_line.png"
            in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//fit_ci_imaging_parallel_front_edge/signal_to_noise_map_line.png"
            in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//fit_ci_imaging_parallel_front_edge/ci_pre_cti_line.png"
            in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//fit_ci_imaging_parallel_front_edge/ci_post_cti_line.png"
            in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//fit_ci_imaging_parallel_front_edge/residual_map_line.png"
            in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//fit_ci_imaging_parallel_front_edge/normalized_residual_map_line.png"
            in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//fit_ci_imaging_parallel_front_edge/chi_squared_map_line.png"
            in plot_patch.paths
        )

    def test___visualizes_multiple_ci_fits_subplot__using_configs(
        self, masked_ci_imaging_7x7, ci_fit_7x7, plot_path, plot_patch
    ):

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = vis.PhaseCIImagingVisualizer(masked_dataset=masked_ci_imaging_7x7)

        visualizer.visualize_multiple_ci_fits_subplots(
            paths=af.Paths(), fits=[ci_fit_7x7]
        )

        assert (
            f"{plot_path}/image//subplots/subplot_residual_maps.png" in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//subplots/subplot_normalized_residual_maps.png"
            in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//subplots/subplot_chi_squared_maps.png"
            in plot_patch.paths
        )

    def test___visualizes_multiple_ci_fits_lines_subplot__using_configs(
        self, masked_ci_imaging_7x7, ci_fit_7x7, plot_path, plot_patch
    ):

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = vis.PhaseCIImagingVisualizer(masked_dataset=masked_ci_imaging_7x7)

        visualizer.visualize_multiple_ci_fits_subplots_lines(
            paths=af.Paths(), fits=[ci_fit_7x7], line_region="parallel_front_edge"
        )

        assert (
            f"{plot_path}/image//subplots/subplot_residual_maps_lines_parallel_front_edge.png"
            in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//subplots/subplot_normalized_residual_maps_lines_parallel_front_edge.png"
            in plot_patch.paths
        )
        assert (
            f"{plot_path}/image//subplots/subplot_chi_squared_maps_lines_parallel_front_edge.png"
            in plot_patch.paths
        )
