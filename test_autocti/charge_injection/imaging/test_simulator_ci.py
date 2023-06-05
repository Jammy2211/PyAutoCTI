import numpy as np
import pytest
import autocti as ac


def test__pre_cti_data_from():
    simulator = ac.SimulatorImagingCI(norm=30.0, pixel_scales=1.0)

    layout = ac.Layout2DCI(shape_2d=(4, 3), region_list=[(0, 3, 0, 2), (2, 3, 2, 3)])

    pre_cti_data = simulator.pre_cti_data_uniform_from(layout=layout)

    assert (
        pre_cti_data.native
        == np.array(
            [[30.0, 30.0, 0.0], [30.0, 30.0, 0.0], [30.0, 30.0, 30.0], [0.0, 0.0, 0.0]]
        )
    ).all()


def test__pre_cti_data_from__compare_uniform_to_non_uniform():
    simulator = ac.SimulatorImagingCI(pixel_scales=1.0, norm=10.0)

    layout = ac.Layout2DCI(shape_2d=(5, 5), region_list=[(2, 4, 0, 5)])

    pre_cti_data_0 = simulator.pre_cti_data_uniform_from(layout=layout)

    simulator = ac.SimulatorImagingCI(
        pixel_scales=1.0, norm=10.0, row_slope=0.0, column_sigma=0.0
    )

    pre_cti_data_1 = simulator.pre_cti_data_non_uniform_from(layout=layout)

    assert (pre_cti_data_0 == pre_cti_data_1).all()


def test__pre_cti_data_from__non_uniformity_in_columns():
    simulator = ac.SimulatorImagingCI(
        pixel_scales=1.0, norm=100.0, row_slope=0.0, column_sigma=1.0, ci_seed=1
    )

    layout = ac.Layout2DCI(shape_2d=(5, 5), region_list=[(0, 3, 0, 3)])

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

    simulator = ac.SimulatorImagingCI(
        pixel_scales=1.0, norm=100.0, row_slope=0.0, column_sigma=1.0, ci_seed=1
    )

    layout = ac.Layout2DCI(shape_2d=(5, 5), region_list=[(1, 2, 1, 3), (3, 4, 1, 3)])

    image = simulator.pre_cti_data_non_uniform_from(layout=layout)

    image[:] = np.round(image[:], 1)

    assert (
        image.native
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 101.6, 99.4, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 101.6, 99.4, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    ).all()

    simulator = ac.SimulatorImagingCI(
        pixel_scales=1.0,
        norm=100.0,
        row_slope=0.0,
        column_sigma=100.0,
        max_norm=100.0,
        ci_seed=1,
    )

    layout = ac.Layout2DCI(shape_2d=(5, 5), region_list=[(0, 5, 0, 5)])

    image = simulator.pre_cti_data_non_uniform_from(layout=layout)

    image = np.round(image, 1)

    # Checked ci_seed to ensure the max is above 100.0 without a max_normalization
    assert np.max(image) < 100.0


def test__pre_cti_data_from__non_uniformity_in_rows():
    simulator = ac.SimulatorImagingCI(
        pixel_scales=1.0, norm=100.0, row_slope=-0.01, column_sigma=0.0
    )

    layout = ac.Layout2DCI(shape_2d=(5, 5), region_list=[(0, 3, 0, 3)])

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

    simulator = ac.SimulatorImagingCI(
        pixel_scales=1.0, norm=100.0, row_slope=-0.01, column_sigma=0.0
    )

    layout = ac.Layout2DCI(shape_2d=(5, 5), region_list=[(0, 2, 1, 4), (3, 5, 1, 4)])

    image = simulator.pre_cti_data_non_uniform_from(layout=layout)

    image[:] = np.round(image[:], 1)

    assert (
        image.native
        == np.array(
            [
                [0.0, 100.0, 100.0, 100.0, 0.0],
                [0.0, 99.3, 99.3, 99.3, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 100.0, 100.0, 100.0, 0.0],
                [0.0, 99.3, 99.3, 99.3, 0.0],
            ]
        )
    ).all()

    simulator = ac.SimulatorImagingCI(
        pixel_scales=1.0,
        norm=100.0,
        row_slope=-0.01,
        column_sigma=1.0,
        read_noise=0.0,
        ci_seed=1,
    )

    layout = ac.Layout2DCI(shape_2d=(5, 5), region_list=[(0, 2, 0, 5), (3, 5, 0, 5)])

    image = simulator.pre_cti_data_non_uniform_from(layout=layout)

    image[:] = np.round(image[:], 1)

    assert (image.native[0:2, 0:5] == image.native[3:5, 0:5]).all()


def test__no_instrumental_effects_input__only_cti_simulated(
    parallel_clocker_2d, traps_x2, ccd
):
    layout = ac.Layout2DCI(
        shape_2d=(5, 5),
        region_list=[(0, 1, 0, 5)],
        serial_overscan=ac.Region2D((1, 2, 1, 2)),
    )

    simulator = ac.SimulatorImagingCI(norm=10.0, pixel_scales=1.0)

    cti = ac.CTI2D(parallel_trap_list=traps_x2, parallel_ccd=ccd)

    dataset = simulator.via_layout_from(
        layout=layout, clocker=parallel_clocker_2d, cti=cti
    )

    assert dataset.data[0, 0:5] == pytest.approx(
        np.array([9.43, 9.43, 9.43, 9.43, 9.43]), 1e-1
    )
    assert dataset.layout == layout


def test__include_charge_noise__is_added_before_cti(parallel_clocker_2d, traps_x2, ccd):
    layout = ac.Layout2DCI(
        shape_2d=(3, 3),
        region_list=[(0, 1, 0, 3)],
        serial_overscan=ac.Region2D((1, 2, 1, 2)),
    )

    simulator = ac.SimulatorImagingCI(
        pixel_scales=1.0, norm=10.0, charge_noise=1.0, noise_seed=1, ci_seed=1
    )

    cti = ac.CTI2D(parallel_trap_list=traps_x2, parallel_ccd=ccd)

    dataset = simulator.via_layout_from(
        layout=layout, clocker=parallel_clocker_2d, cti=cti
    )

    data_no_noise = simulator.pre_cti_data_uniform_from(layout=layout)

    # assert dataset.data - data_no_noise.native == pytest.approx(
    #     np.array([[1.01064, -1.1632, -1.08214], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    #     1e-1,
    # )
    # assert dataset.layout == layout


def test__include_read_noise__is_added_after_cti(parallel_clocker_2d, traps_x2, ccd):
    layout = ac.Layout2DCI(
        shape_2d=(3, 3),
        region_list=[(0, 1, 0, 3)],
        serial_overscan=ac.Region2D((1, 2, 1, 2)),
    )

    simulator = ac.SimulatorImagingCI(
        pixel_scales=1.0, norm=10.0, read_noise=1.0, noise_seed=1
    )

    cti = ac.CTI2D(parallel_trap_list=traps_x2, parallel_ccd=ccd)

    dataset = simulator.via_layout_from(
        layout=layout, clocker=parallel_clocker_2d, cti=cti
    )

    data_no_noise = simulator.pre_cti_data_uniform_from(layout=layout)

    # assert dataset.data - data_no_noise.native == pytest.approx(
    #     np.array(
    #         [[1.055, -1.180, -1.097], [-1.073, 0.865, -2.301], [1.744, -0.761, 0.319]]
    #     ),
    #     1e-1,
    # )
    # assert dataset.layout == layout


def test__include_cosmics__is_added_to_image_and_trailed(
    parallel_clocker_2d, traps_x2, ccd
):
    layout = ac.Layout2DCI(
        shape_2d=(5, 5),
        region_list=[(0, 1, 0, 5)],
        serial_overscan=ac.Region2D((1, 2, 1, 2)),
    )

    simulator = ac.SimulatorImagingCI(pixel_scales=1.0, norm=10.0)

    cosmic_ray_map = ac.Array2D.zeros(shape_native=(5, 5), pixel_scales=0.1).native
    cosmic_ray_map[2, 2] = 100.0

    cti = ac.CTI2D(parallel_trap_list=traps_x2, parallel_ccd=ccd)

    dataset = simulator.via_layout_from(
        layout=layout,
        clocker=parallel_clocker_2d,
        cti=cti,
        cosmic_ray_map=cosmic_ray_map,
    )

    assert dataset.data[0, 0:5] == pytest.approx(
        np.array([9.43, 9.43, 9.43, 9.43, 9.43]), 1e-1
    )
    assert 0.0 < dataset.data[1, 1] < 100.0
    assert dataset.data[2, 2] > 94.0
    assert (dataset.data[1, 1:4] > 0.0).all()
    assert (
        dataset.cosmic_ray_map
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
    assert dataset.layout == layout

    dataset = simulator.via_layout_from(
        layout=layout,
        clocker=None,
        cti=None,
        cosmic_ray_map=cosmic_ray_map,
    )

    assert dataset.data[2, 2] > 94.0


def test__from_pre_cti_data(parallel_clocker_2d, traps_x2, ccd):
    layout = ac.Layout2DCI(
        shape_2d=(5, 5),
        region_list=[(0, 1, 0, 5)],
        serial_overscan=ac.Region2D((1, 2, 1, 2)),
    )

    simulator = ac.SimulatorImagingCI(
        pixel_scales=1.0, norm=1.0, read_noise=4.0, noise_seed=1
    )

    cosmic_ray_map = ac.Array2D.zeros(shape_native=(5, 5), pixel_scales=0.1).native
    cosmic_ray_map[2, 2] = 100.0

    cti = ac.CTI2D(parallel_trap_list=traps_x2, parallel_ccd=ccd)

    dataset = simulator.via_layout_from(
        layout=layout,
        clocker=parallel_clocker_2d,
        cti=cti,
        cosmic_ray_map=cosmic_ray_map,
    )

    pre_cti_data = simulator.pre_cti_data_uniform_from(layout=layout)

    dataset_via = simulator.via_pre_cti_data_from(
        pre_cti_data=pre_cti_data.native,
        layout=layout,
        clocker=parallel_clocker_2d,
        cti=cti,
        cosmic_ray_map=cosmic_ray_map,
    )

    assert (dataset.data == dataset_via.image).all()
    assert (dataset.noise_map == dataset_via.noise_map).all()
    assert (dataset.pre_cti_data == dataset_via.pre_cti_data).all()
    assert (dataset.cosmic_ray_map == dataset_via.cosmic_ray_map).all()


def test__from_post_cti_data(parallel_clocker_2d, traps_x2, ccd):
    layout = ac.Layout2DCI(
        shape_2d=(5, 5),
        region_list=[(0, 1, 0, 5)],
        serial_overscan=ac.Region2D((1, 2, 1, 2)),
    )

    simulator = ac.SimulatorImagingCI(
        pixel_scales=1.0, norm=1.0, read_noise=4.0, noise_seed=1
    )

    cosmic_ray_map = ac.Array2D.zeros(shape_native=(5, 5), pixel_scales=0.1).native
    cosmic_ray_map[2, 2] = 100.0

    cti = ac.CTI2D(parallel_trap_list=traps_x2, parallel_ccd=ccd)

    dataset = simulator.via_layout_from(
        layout=layout,
        clocker=parallel_clocker_2d,
        cti=cti,
        cosmic_ray_map=cosmic_ray_map,
    )

    pre_cti_data = simulator.pre_cti_data_uniform_from(layout=layout).native
    pre_cti_data += cosmic_ray_map

    cti = ac.CTI2D(parallel_trap_list=traps_x2, parallel_ccd=ccd)

    post_cti_data = parallel_clocker_2d.add_cti(data=pre_cti_data, cti=cti)

    pre_cti_data -= cosmic_ray_map

    dataset_via = simulator.via_post_cti_data_from(
        post_cti_data=post_cti_data,
        pre_cti_data=pre_cti_data.native,
        layout=layout,
        cosmic_ray_map=cosmic_ray_map,
    )

    assert (dataset.data == dataset_via.image).all()
    assert (dataset.noise_map == dataset_via.noise_map).all()
    assert (dataset.pre_cti_data == dataset_via.pre_cti_data).all()
    assert (dataset.cosmic_ray_map == dataset_via.cosmic_ray_map).all()
