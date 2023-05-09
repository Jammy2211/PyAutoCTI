import numpy as np
import pytest

import autocti as ac


def test__no_instrumental_effects_input__only_cti_simulated(clocker_1d, traps_x2, ccd):
    layout = ac.Layout1D(shape_1d=(5,), region_list=[(0, 5)])

    simulator = ac.SimulatorDataset1D(
        pixel_scales=1.0, norm=10.0, add_poisson_noise=False
    )

    cti = ac.CTI1D(trap_list=traps_x2, ccd=ccd)

    dataset_1d = simulator.via_layout_from(layout=layout, clocker=clocker_1d, cti=cti)

    assert dataset_1d.data == pytest.approx(
        np.array([9.43, 9.43, 9.43, 9.43, 9.43]), 1e-1
    )
    assert dataset_1d.layout == layout


def test__include_charge_noise__is_added_before_cti(clocker_1d, traps_x2, ccd):
    layout = ac.Layout1D(shape_1d=(5,), region_list=[(0, 3)])

    simulator = ac.SimulatorDataset1D(
        pixel_scales=1.0,
        norm=10.0,
        charge_noise=1.0,
        add_poisson_noise=False,
        noise_seed=1,
    )

    cti = ac.CTI1D(trap_list=traps_x2, ccd=ccd)

    dataset_1d = simulator.via_layout_from(layout=layout, clocker=clocker_1d, cti=cti)

    print(dataset_1d.data)

    # Use seed to give us a known read noises map we'll test_autocti for

    assert dataset_1d.data == pytest.approx(
        np.array([11.01064, 9.4526, 9.4990, 1.0969, 0.59858]), 1e-2
    )
    assert dataset_1d.layout == layout


def test__include_read_noise__is_added_after_cti(clocker_1d, traps_x2, ccd):
    layout = ac.Layout1D(shape_1d=(5,), region_list=[(0, 5)])

    simulator = ac.SimulatorDataset1D(
        pixel_scales=1.0,
        norm=10.0,
        read_noise=1.0,
        add_poisson_noise=False,
        noise_seed=1,
    )

    cti = ac.CTI1D(trap_list=traps_x2, ccd=ccd)

    dataset_1d = simulator.via_layout_from(layout=layout, clocker=clocker_1d, cti=cti)

    # Use seed to give us a known read noises map we'll test_autocti for

    assert dataset_1d.data == pytest.approx(
        np.array([11.05513, 9.36790, 9.47129, 8.92700, 10.8654]), 1e-2
    )
    assert dataset_1d.layout == layout


def test__pre_cti_data_from():
    simulator = ac.SimulatorDataset1D(norm=10.0, pixel_scales=1.0)

    layout = ac.Layout1D(shape_1d=(3,), region_list=[(0, 2)])

    pre_cti_data = simulator.pre_cti_data_from(layout=layout, pixel_scales=1.0)

    assert (pre_cti_data.native == np.array([10.0, 10.0, 0.0])).all()


def test__from_pre_cti_data(clocker_1d, traps_x2, ccd):
    layout = ac.Layout1D(shape_1d=(5,), region_list=[(0, 5)])

    simulator = ac.SimulatorDataset1D(
        pixel_scales=1.0,
        norm=10.0,
        read_noise=4.0,
        add_poisson_noise=False,
        noise_seed=1,
    )

    cti = ac.CTI1D(trap_list=traps_x2, ccd=ccd)

    dataset_1d = simulator.via_layout_from(layout=layout, clocker=clocker_1d, cti=cti)

    pre_cti_data = simulator.pre_cti_data_from(layout=layout, pixel_scales=1.0)

    dataset_1d_via_pre_cti_data = simulator.via_pre_cti_data_from(
        pre_cti_data=pre_cti_data.native, layout=layout, clocker=clocker_1d, cti=cti
    )

    assert (dataset_1d.data == dataset_1d_via_pre_cti_data.data).all()
    assert (dataset_1d.noise_map == dataset_1d_via_pre_cti_data.noise_map).all()
    assert (dataset_1d.pre_cti_data == dataset_1d_via_pre_cti_data.pre_cti_data).all()


def test__from_post_cti_data(clocker_1d, traps_x2, ccd):
    layout = ac.Layout1D(shape_1d=(5,), region_list=[(0, 5)])
    simulator = ac.SimulatorDataset1D(
        pixel_scales=1.0,
        norm=10.0,
        read_noise=4.0,
        add_poisson_noise=False,
        noise_seed=1,
    )

    cti = ac.CTI1D(trap_list=traps_x2, ccd=ccd)

    dataset_1d = simulator.via_layout_from(layout=layout, clocker=clocker_1d, cti=cti)

    pre_cti_data = simulator.pre_cti_data_from(layout=layout, pixel_scales=1.0).native

    post_cti_data = clocker_1d.add_cti(data=pre_cti_data, cti=cti)

    dataset_1d_via_post_cti_data = simulator.via_post_cti_data_from(
        post_cti_data=post_cti_data, pre_cti_data=pre_cti_data.native, layout=layout
    )

    assert (dataset_1d.data == dataset_1d_via_post_cti_data.data).all()
    assert (dataset_1d.noise_map == dataset_1d_via_post_cti_data.noise_map).all()
    assert (dataset_1d.pre_cti_data == dataset_1d_via_post_cti_data.pre_cti_data).all()
