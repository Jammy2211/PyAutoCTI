import numpy as np
import pytest

import autocti as ac


class TestApplyMask:
    def test__construtor__masks_arrays_correctly(self, dataset_line_7):

        mask = ac.Mask1D.unmasked(
            shape_slim=dataset_line_7.data.shape_slim,
            pixel_scales=dataset_line_7.data.pixel_scales,
        )

        mask[0] = True

        masked_dataset_line = dataset_line_7.apply_mask(mask=mask)

        assert (masked_dataset_line.mask == mask).all()

        masked_data = dataset_line_7.data
        masked_data[0] = 0.0

        assert (masked_dataset_line.data == masked_data).all()

        masked_noise_map = dataset_line_7.noise_map
        masked_noise_map[0] = 0.0

        assert (masked_dataset_line.noise_map == masked_noise_map).all()

        assert (masked_dataset_line.pre_cti_line == dataset_line_7.pre_cti_line).all()


class TestSimulatorLineDataset:
    def test__no_instrumental_effects_input__only_cti_simulated(
        self, clocker_1d, traps_x2, ccd
    ):

        layout = ac.Layout1DLine(
            shape_1d=(5,),
            normalization=10.0,
            region_list=[(0, 5)],
            overscan=ac.Region1D((4, 5)),
        )

        simulator = ac.SimulatorDatasetLine(pixel_scales=1.0, add_poisson_noise=False)

        dataset_line = simulator.from_layout(
            layout=layout, clocker=clocker_1d, traps=traps_x2, ccd=ccd
        )

        print(dataset_line.data)

        assert dataset_line.data == pytest.approx(
            np.array([9.43, 9.43, 9.43, 9.43, 9.43]), 1e-1
        )
        assert dataset_line.layout == layout


#
#     def test__include_read_noise__is_added_after_cti(
#         self, clocker_1d, traps_x2, ccd
#     ):
#
#         layout = ac.Layout1DLine(
#             shape_2d=(3, 3),
#             normalization=10.0,
#             region_list=[(0, 1, 0, 3)],
#             serial_overscan=ac.Region2D((1, 2, 1, 2)),
#         )
#
#         simulator = ac.SimulatorDatasetLine(
#             pixel_scales=1.0, read_noise=1.0, add_poisson_noise=True, noise_seed=1
#         )
#
#         dataset_line = simulator.from_layout(
#             layout=layout,
#             clocker=clocker_1d,
#             parallel_traps=traps_x2,
#             parallel_ccd=ccd,
#         )
#
#         data_no_noise = layout.pre_cti_data_from(
#             shape_native=(3, 3), pixel_scales=1.0
#         )
#
#         # Use seed to give us a known read noises map we'll test_autocti for
#
#         assert dataset_line.data - data_no_noise.native == pytest.approx(
#             np.array(
#                 [
#                     [1.055, -1.180, -1.097],
#                     [-0.780, 1.1574, -2.009],
#                     [1.863, -0.642, 0.437],
#                 ]
#             ),
#             1e-2,
#         )
#         assert dataset_line.layout == layout
#
#     def test__include_cosmics__is_added_to_data_and_trailed(
#         self, clocker_1d, traps_x2, ccd
#     ):
#
#         layout = ac.Layout1DLine(
#             shape_2d=(5, 5),
#             normalization=10.0,
#             region_list=[(0, 1, 0, 5)],
#             serial_overscan=ac.Region2D((1, 2, 1, 2)),
#         )
#
#         simulator = ac.SimulatorDatasetLine(pixel_scales=1.0, add_poisson_noise=False)
#
#         cosmic_ray_map = np.zeros((5, 5))
#         cosmic_ray_map[2, 2] = 100.0
#
#         dataset_line = simulator.from_layout(
#             layout=layout,
#             clocker=clocker_1d,
#             parallel_traps=traps_x2,
#             parallel_ccd=ccd,
#             cosmic_ray_map=cosmic_ray_map,
#         )
#
#         assert dataset_line.data[0, 0:5] == pytest.approx(
#             np.array([9.43, 9.43, 9.43, 9.43, 9.43]), 1e-1
#         )
#         assert 0.0 < dataset_line.data[1, 1] < 100.0
#         assert dataset_line.data[2, 2] > 94.0
#         assert (dataset_line.data[1, 1:4] > 0.0).all()
#         assert (
#             dataset_line.cosmic_ray_map
#             == np.array(
#                 [
#                     [0.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 100.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0],
#                 ]
#             )
#         ).all()
#         assert dataset_line.layout == layout
#
#     def test__from_pre_cti_data(self, clocker_1d, traps_x2, ccd):
#
#         layout = ac.Layout1DLine(
#             shape_2d=(5, 5),
#             normalization=10.0,
#             region_list=[(0, 1, 0, 5)],
#             serial_overscan=ac.Region2D((1, 2, 1, 2)),
#         )
#
#         simulator = ac.SimulatorDatasetLine(
#             pixel_scales=1.0, read_noise=4.0, add_poisson_noise=False, noise_seed=1
#         )
#
#         cosmic_ray_map = np.zeros((5, 5))
#         cosmic_ray_map[2, 2] = 100.0
#
#         dataset_line = simulator.from_layout(
#             layout=layout,
#             clocker=clocker_1d,
#             parallel_traps=traps_x2,
#             parallel_ccd=ccd,
#             cosmic_ray_map=cosmic_ray_map,
#         )
#
#         pre_cti_data = layout.pre_cti_data_from(
#             shape_native=(5, 5), pixel_scales=1.0
#         )
#
#         dataset_line_via_pre_cti_data = simulator.from_pre_cti_data(
#             pre_cti_data=pre_cti_data.native,
#             layout=layout,
#             clocker=clocker_1d,
#             parallel_traps=traps_x2,
#             parallel_ccd=ccd,
#             cosmic_ray_map=cosmic_ray_map,
#         )
#
#         assert (dataset_line.data == dataset_line_via_pre_cti_data.data).all()
#         assert (dataset_line.noise_map == dataset_line_via_pre_cti_data.noise_map).all()
#         assert (dataset_line.pre_cti_data == dataset_line_via_pre_cti_data.pre_cti_data).all()
#         assert (
#             dataset_line.cosmic_ray_map == dataset_line_via_pre_cti_data.cosmic_ray_map
#         ).all()
#
#     def test__from_post_cti_data(self, clocker_1d, traps_x2, ccd):
#
#         layout = ac.Layout1DLine(
#             shape_2d=(5, 5),
#             normalization=10.0,
#             region_list=[(0, 1, 0, 5)],
#             serial_overscan=ac.Region2D((1, 2, 1, 2)),
#         )
#
#         simulator = ac.SimulatorDatasetLine(
#             pixel_scales=1.0, read_noise=4.0, add_poisson_noise=False, noise_seed=1
#         )
#
#         cosmic_ray_map = np.zeros((5, 5))
#         cosmic_ray_map[2, 2] = 100.0
#
#         dataset_line = simulator.from_layout(
#             layout=layout,
#             clocker=clocker_1d,
#             parallel_traps=traps_x2,
#             parallel_ccd=ccd,
#             cosmic_ray_map=cosmic_ray_map,
#         )
#
#         pre_cti_data = layout.pre_cti_data_from(
#             shape_native=(5, 5), pixel_scales=1.0
#         ).native
#         pre_cti_data += cosmic_ray_map
#
#         post_cti_data = clocker_1d.add_cti(
#             data_pre_cti=pre_cti_data, parallel_traps=traps_x2, parallel_ccd=ccd
#         )
#
#         pre_cti_data -= cosmic_ray_map
#
#         dataset_line_via_post_cti_data = simulator.from_post_cti_data(
#             post_cti_data=post_cti_data,
#             pre_cti_data=pre_cti_data.native,
#             layout=layout,
#             cosmic_ray_map=cosmic_ray_map,
#         )
#
#         assert (dataset_line.data == dataset_line_via_post_cti_data.data).all()
#         assert (dataset_line.noise_map == dataset_line_via_post_cti_data.noise_map).all()
#         assert (dataset_line.pre_cti_data == dataset_line_via_post_cti_data.pre_cti_data).all()
#         assert (
#             dataset_line.cosmic_ray_map == dataset_line_via_post_cti_data.cosmic_ray_map
#         ).all()
