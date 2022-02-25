import os
from os import path
import shutil

import numpy as np
import pytest
import autocti as ac

test_data_path = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "arrays"
)


class TestSimulatorImagingCI:
    def test__pre_cti_data_from(self):

        simulator = ac.ci.SimulatorImagingCI(normalization=30.0, pixel_scales=1.0)

        layout = ac.ci.Layout2DCI(
            shape_2d=(4, 3), region_list=[(0, 3, 0, 2), (2, 3, 2, 3)]
        )

        pre_cti_data = simulator.pre_cti_data_uniform_from(layout=layout)

        assert (
            pre_cti_data.native
            == np.array(
                [
                    [30.0, 30.0, 0.0],
                    [30.0, 30.0, 0.0],
                    [30.0, 30.0, 30.0],
                    [0.0, 0.0, 0.0],
                ]
            )
        ).all()

    def test__region_ci_from(self):

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0,
            normalization=100.0,
            row_slope=0.0,
            column_sigma=0.0,
            ci_seed=1,
        )

        layout = ac.ci.Layout2DCI(shape_2d=(5, 3), region_list=[(0, 1, 0, 1)])

        region = simulator.region_ci_from(region_dimensions=(3, 3))

        assert (
            region
            == np.array(
                [[100.0, 100.0, 100.0], [100.0, 100.0, 100.0], [100.0, 100.0, 100.0]]
            )
        ).all()

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0,
            normalization=100.0,
            row_slope=0.0,
            column_sigma=1.0,
            ci_seed=1,
        )

        layout = ac.ci.Layout2DCI(shape_2d=(5, 3), region_list=[(0, 1, 0, 1)])

        region = simulator.region_ci_from(region_dimensions=(3, 3))

        region = np.round(region, 1)

        assert (
            region
            == np.array([[101.6, 99.4, 99.5], [101.6, 99.4, 99.5], [101.6, 99.4, 99.5]])
        ).all()

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0,
            normalization=100.0,
            row_slope=-0.01,
            column_sigma=0.0,
            ci_seed=1,
        )

        layout = ac.ci.Layout2DCI(shape_2d=(5, 3), region_list=[(0, 1, 0, 1)])

        region = simulator.region_ci_from(region_dimensions=(3, 3))

        region = np.round(region, 1)

        assert (
            region
            == np.array([[100.0, 100.0, 100.0], [99.3, 99.3, 99.3], [98.9, 98.9, 98.9]])
        ).all()

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0,
            normalization=100.0,
            row_slope=-0.01,
            column_sigma=1.0,
            ci_seed=1,
        )

        layout = ac.ci.Layout2DCI(shape_2d=(5, 3), region_list=[(0, 1, 0, 1)])

        region = simulator.region_ci_from(region_dimensions=(3, 3))

        region = np.round(region, 1)

        assert (
            region
            == np.array([[101.6, 99.4, 99.5], [100.9, 98.7, 98.8], [100.5, 98.3, 98.4]])
        ).all()

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0,
            normalization=100.0,
            row_slope=0.0,
            column_sigma=100.0,
            ci_seed=1,
        )

        layout = ac.ci.Layout2DCI(shape_2d=(5, 3), region_list=[(0, 1, 0, 1)])

        region = simulator.region_ci_from(region_dimensions=(10, 10))

        assert (region > 0).all()

    def test__pre_cti_data_from__compare_uniform_to_non_uniform(self):

        simulator = ac.ci.SimulatorImagingCI(pixel_scales=1.0, normalization=10.0)

        layout = ac.ci.Layout2DCI(shape_2d=(5, 5), region_list=[(2, 4, 0, 5)])

        pre_cti_data_0 = simulator.pre_cti_data_uniform_from(layout=layout)

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0, normalization=10.0, row_slope=0.0, column_sigma=0.0
        )

        pre_cti_data_1 = simulator.pre_cti_data_non_uniform_from(layout=layout)

        assert (pre_cti_data_0 == pre_cti_data_1).all()

    def test__pre_cti_data_from__non_uniformity_in_columns(self):

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0,
            normalization=100.0,
            row_slope=0.0,
            column_sigma=1.0,
            ci_seed=1,
        )

        layout = ac.ci.Layout2DCI(shape_2d=(5, 5), region_list=[(0, 3, 0, 3)])

        image = simulator.pre_cti_data_non_uniform_from(layout=layout)

        image[:] = np.round(image[:], 1)

        assert (
            image.native
            == np.array(
                [
                    [101.6, 99.4, 99.5, 0.0, 0.0],
                    [101.6, 99.4, 99.5, 0.0, 0.0],
                    [101.6, 99.4, 99.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0,
            normalization=100.0,
            row_slope=0.0,
            column_sigma=1.0,
            ci_seed=1,
        )

        layout = ac.ci.Layout2DCI(
            shape_2d=(5, 5), region_list=[(1, 4, 1, 3), (1, 4, 4, 5)]
        )

        image = simulator.pre_cti_data_non_uniform_from(layout=layout)

        image[:] = np.round(image[:], 1)

        assert (
            image.native
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 101.6, 99.4, 0.0, 101.6],
                    [0.0, 101.6, 99.4, 0.0, 101.6],
                    [0.0, 101.6, 99.4, 0.0, 101.6],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0,
            normalization=100.0,
            row_slope=0.0,
            column_sigma=100.0,
            max_normalization=100.0,
            ci_seed=1,
        )

        layout = ac.ci.Layout2DCI(shape_2d=(5, 5), region_list=[(0, 5, 0, 5)])

        image = simulator.pre_cti_data_non_uniform_from(layout=layout)

        image = np.round(image, 1)

        # Checked ci_seed to ensure the max is above 100.0 without a max_normalization
        assert np.max(image) < 100.0

    def test__pre_cti_data_from__non_uniformity_in_rows(self):

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0, normalization=100.0, row_slope=-0.01, column_sigma=0.0
        )

        layout = ac.ci.Layout2DCI(shape_2d=(5, 5), region_list=[(0, 3, 0, 3)])

        image = simulator.pre_cti_data_non_uniform_from(layout=layout)

        image[:] = np.round(image[:], 1)

        assert (
            image.native
            == np.array(
                [
                    [100.0, 100.0, 100.0, 0.0, 0.0],
                    [99.3, 99.3, 99.3, 0.0, 0.0],
                    [98.9, 98.9, 98.9, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0, normalization=100.0, row_slope=-0.01, column_sigma=0.0
        )

        layout = ac.ci.Layout2DCI(
            shape_2d=(5, 5), region_list=[(1, 5, 1, 4), (0, 5, 4, 5)]
        )

        image = simulator.pre_cti_data_non_uniform_from(layout=layout)

        image[:] = np.round(image[:], 1)

        assert (
            image.native
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 100.0],
                    [0.0, 100.0, 100.0, 100.0, 99.3],
                    [0.0, 99.3, 99.3, 99.3, 98.9],
                    [0.0, 98.9, 98.9, 98.9, 98.6],
                    [0.0, 98.6, 98.6, 98.6, 98.4],
                ]
            )
        ).all()

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0,
            normalization=100.0,
            row_slope=-0.01,
            column_sigma=1.0,
            ci_seed=1,
        )

        layout = ac.ci.Layout2DCI(
            shape_2d=(5, 5), region_list=[(1, 5, 1, 4), (0, 5, 4, 5)]
        )

        image = simulator.pre_cti_data_non_uniform_from(layout=layout)

        image[:] = np.round(image[:], 1)

        assert (
            image.native
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 101.6],
                    [0.0, 101.6, 99.4, 99.5, 100.9],
                    [0.0, 100.9, 98.7, 98.8, 100.5],
                    [0.0, 100.5, 98.3, 98.4, 100.2],
                    [0.0, 100.2, 98.0, 98.1, 100.0],
                ]
            )
        ).all()

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0,
            normalization=100.0,
            row_slope=-0.01,
            column_sigma=1.0,
            read_noise=0.0,
            ci_seed=1,
        )

        layout = ac.ci.Layout2DCI(
            shape_2d=(5, 5), region_list=[(0, 2, 0, 5), (3, 5, 0, 5)]
        )

        image = simulator.pre_cti_data_non_uniform_from(layout=layout)

        image[:] = np.round(image[:], 1)

        print(image.native)

        assert (image.native[0:2, 0:5] == image.native[3:5, 0:5]).all()

    def test__no_instrumental_effects_input__only_cti_simulated(
        self, parallel_clocker_2d, traps_x2, ccd
    ):

        layout = ac.ci.Layout2DCI(
            shape_2d=(5, 5),
            region_list=[(0, 1, 0, 5)],
            serial_overscan=ac.Region2D((1, 2, 1, 2)),
        )

        simulator = ac.ci.SimulatorImagingCI(normalization=10.0, pixel_scales=1.0)

        imaging = simulator.from_layout(
            layout=layout,
            clocker=parallel_clocker_2d,
            parallel_trap_list=traps_x2,
            parallel_ccd=ccd,
        )

        assert imaging.image[0, 0:5] == pytest.approx(
            np.array([9.43, 9.43, 9.43, 9.43, 9.43]), 1e-1
        )
        assert imaging.layout == layout

    def test__include_read_noise__is_added_after_cti(
        self, parallel_clocker_2d, traps_x2, ccd
    ):

        layout = ac.ci.Layout2DCI(
            shape_2d=(3, 3),
            region_list=[(0, 1, 0, 3)],
            serial_overscan=ac.Region2D((1, 2, 1, 2)),
        )

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0, normalization=10.0, read_noise=1.0, noise_seed=1
        )

        imaging = simulator.from_layout(
            layout=layout,
            clocker=parallel_clocker_2d,
            parallel_trap_list=traps_x2,
            parallel_ccd=ccd,
        )

        image_no_noise = simulator.pre_cti_data_uniform_from(layout=layout)

        # Use seed to give us a known read noises map we'll test_autocti for

        assert imaging.image - image_no_noise.native == pytest.approx(
            np.array(
                [
                    [1.055, -1.180, -1.097],
                    [-0.780, 1.1574, -2.009],
                    [1.863, -0.642, 0.437],
                ]
            ),
            1e-2,
        )
        assert imaging.layout == layout

    def test__include_cosmics__is_added_to_image_and_trailed(
        self, parallel_clocker_2d, traps_x2, ccd
    ):

        layout = ac.ci.Layout2DCI(
            shape_2d=(5, 5),
            region_list=[(0, 1, 0, 5)],
            serial_overscan=ac.Region2D((1, 2, 1, 2)),
        )

        simulator = ac.ci.SimulatorImagingCI(pixel_scales=1.0, normalization=10.0)

        cosmic_ray_map = ac.Array2D.zeros(shape_native=(5, 5), pixel_scales=0.1).native
        cosmic_ray_map[2, 2] = 100.0

        imaging = simulator.from_layout(
            layout=layout,
            clocker=parallel_clocker_2d,
            parallel_trap_list=traps_x2,
            parallel_ccd=ccd,
            cosmic_ray_map=cosmic_ray_map,
        )

        assert imaging.image[0, 0:5] == pytest.approx(
            np.array([9.43, 9.43, 9.43, 9.43, 9.43]), 1e-1
        )
        assert 0.0 < imaging.image[1, 1] < 100.0
        assert imaging.image[2, 2] > 94.0
        assert (imaging.image[1, 1:4] > 0.0).all()
        assert (
            imaging.cosmic_ray_map
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 100.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()
        assert imaging.layout == layout

    def test__from_pre_cti_data(self, parallel_clocker_2d, traps_x2, ccd):

        layout = ac.ci.Layout2DCI(
            shape_2d=(5, 5),
            region_list=[(0, 1, 0, 5)],
            serial_overscan=ac.Region2D((1, 2, 1, 2)),
        )

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0, normalization=1.0, read_noise=4.0, noise_seed=1
        )

        cosmic_ray_map = ac.Array2D.zeros(shape_native=(5, 5), pixel_scales=0.1).native
        cosmic_ray_map[2, 2] = 100.0

        imaging = simulator.from_layout(
            layout=layout,
            clocker=parallel_clocker_2d,
            parallel_trap_list=traps_x2,
            parallel_ccd=ccd,
            cosmic_ray_map=cosmic_ray_map,
        )

        pre_cti_data = simulator.pre_cti_data_uniform_from(layout=layout)

        imaging_via_pre_cti_data = simulator.from_pre_cti_data(
            pre_cti_data=pre_cti_data.native,
            layout=layout,
            clocker=parallel_clocker_2d,
            parallel_trap_list=traps_x2,
            parallel_ccd=ccd,
            cosmic_ray_map=cosmic_ray_map,
        )

        assert (imaging.image == imaging_via_pre_cti_data.image).all()
        assert (imaging.noise_map == imaging_via_pre_cti_data.noise_map).all()
        assert (imaging.pre_cti_data == imaging_via_pre_cti_data.pre_cti_data).all()
        assert (imaging.cosmic_ray_map == imaging_via_pre_cti_data.cosmic_ray_map).all()

    def test__from_post_cti_data(self, parallel_clocker_2d, traps_x2, ccd):

        layout = ac.ci.Layout2DCI(
            shape_2d=(5, 5),
            region_list=[(0, 1, 0, 5)],
            serial_overscan=ac.Region2D((1, 2, 1, 2)),
        )

        simulator = ac.ci.SimulatorImagingCI(
            pixel_scales=1.0, normalization=1.0, read_noise=4.0, noise_seed=1
        )

        cosmic_ray_map = ac.Array2D.zeros(shape_native=(5, 5), pixel_scales=0.1).native
        cosmic_ray_map[2, 2] = 100.0

        imaging = simulator.from_layout(
            layout=layout,
            clocker=parallel_clocker_2d,
            parallel_trap_list=traps_x2,
            parallel_ccd=ccd,
            cosmic_ray_map=cosmic_ray_map,
        )

        pre_cti_data = simulator.pre_cti_data_uniform_from(layout=layout).native
        pre_cti_data += cosmic_ray_map

        post_cti_data = parallel_clocker_2d.add_cti(
            data=pre_cti_data, parallel_trap_list=traps_x2, parallel_ccd=ccd
        )

        pre_cti_data -= cosmic_ray_map

        imaging_via_post_cti_data = simulator.from_post_cti_data(
            post_cti_data=post_cti_data,
            pre_cti_data=pre_cti_data.native,
            layout=layout,
            cosmic_ray_map=cosmic_ray_map,
        )

        assert (imaging.image == imaging_via_post_cti_data.image).all()
        assert (imaging.noise_map == imaging_via_post_cti_data.noise_map).all()
        assert (imaging.pre_cti_data == imaging_via_post_cti_data.pre_cti_data).all()
        assert (
            imaging.cosmic_ray_map == imaging_via_post_cti_data.cosmic_ray_map
        ).all()
