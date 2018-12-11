import pytest
import numpy as np

from autocti.data import cti_image
from autocti.model import arctic_settings
from autocti.model import arctic_params

@pytest.fixture(scope='class')
def arctic_parallel():

    parallel_settings = arctic_settings.ParallelSettings(well_depth=84700, niter=1, express=5,n_levels=2000,
                                                        readout_offset=0)
    arctic_parallel = arctic_settings.ArcticSettings(neomode='NEO', parallel=parallel_settings)

    return arctic_parallel

@pytest.fixture(scope='class')
def arctic_serial():

    serial_settings = arctic_settings.SerialSettings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                                    readout_offset=0)

    arctic_serial = arctic_settings.ArcticSettings(neomode='NEO', serial=serial_settings)

    return arctic_serial

@pytest.fixture(scope='class')
def arctic_both():

    parallel_settings = arctic_settings.ParallelSettings(well_depth=84700, niter=1, express=5,n_levels=2000,
                                                        readout_offset=0)

    serial_settings = arctic_settings.SerialSettings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                                    readout_offset=0)

    arctic_both = arctic_settings.ArcticSettings(neomode='NEO', parallel=parallel_settings,
                                                serial=serial_settings)

    return arctic_both

@pytest.fixture(scope='class')
def params_parallel():

    params_parallel = arctic_params.ParallelOneSpecies(trap_densities=(0.1,), trap_lifetimes=(1.0,),
                                                      well_notch_depth=0.000001, well_fill_beta=0.8)

    params_parallel = arctic_params.ArcticParams(parallel_species=params_parallel)

    return params_parallel

@pytest.fixture(scope='class')
def params_serial():

    params_serial = arctic_params.SerialOneSpecies(trap_densities=(0.2,), trap_lifetimes=(2.0,),
                                                  well_notch_depth=0.000001, well_fill_beta=0.4)

    params_serial = arctic_params.ArcticParams(serial_species=params_serial)

    return params_serial

@pytest.fixture(scope='class')
def params_both():

    params_parallel = arctic_params.ParallelOneSpecies(trap_densities=(0.4,), trap_lifetimes=(1.0,),
                                                      well_notch_depth=0.000001, well_fill_beta=0.8)

    params_serial = arctic_params.SerialOneSpecies(trap_densities=(0.2,), trap_lifetimes=(2.0,),
                                                  well_notch_depth=0.000001, well_fill_beta=0.4)

    params_both = arctic_params.ArcticParams(parallel_species=params_parallel, serial_species=params_serial)

    return params_both


class TestQuadrantGeometryEuclidBottomLeft:

    class TestAddCTI:

        class TestParallelCTI:

            def test__horizontal_charge_line__loses_charge_and_trails_form_below_line(self, arctic_parallel,
                                                                                      params_parallel):

                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, :] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBL(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_parallel, cti_settings=arctic_parallel)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[0:2, :] == 0.0).all()  # No trails closest to read out
                assert (image_difference[2, :] < 0.0).all()  # line loses charge
                assert (image_difference[3:5, :] > 0.0).all()  # trails appear below line

            def test__vertical_charge_line__loses_charge_no_trails(self, arctic_parallel, params_parallel):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[:, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBL(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_parallel, cti_settings=arctic_parallel)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[:, 0:2] == 0.0).all()  # No trails closest to read out
                assert (image_difference[:, 3:-1] == 0.0).all()  # line loses charge
                assert (image_difference[:, 2] < 0.0).all()  # No trails away from read-out either, as wrong direction

        class TestSerialCTI:

            def test__horizontal_charge_line__loses_charge_no_trails(self, arctic_serial, params_serial):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, :] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBL(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_serial, cti_settings=arctic_serial)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[0:2, :] == 0.0).all()  # No trails closest to read out
                assert (image_difference[2, :] < 0.0).all()  # line loses charge
                assert (image_difference[3:5,
                        :] == 0.0).all()  # No trails away from read-out either, as wrong direction

            def test__vertical_charge_line__loses_charge_and_trails_form_to_right_of_line(self, arctic_serial,
                                                                                          params_serial):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[:, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBL(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_serial, cti_settings=arctic_serial)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[:, 0:2] == 0.0).all()  # No trails closest to read out
                assert (image_difference[:, 2] < 0.0).all()  # line loses charge
                assert (image_difference[:, 3:5] > 0.0).all()  # Trails appear away from read-out

        class TestParallelAndSerialCTI:

            def test__horizontal_charge_line__loses_charge_trails_form_both_directions(self, arctic_both,
                                                                                       params_both):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, 1:4] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBL(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_both, cti_settings=arctic_both)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[0:2, :] == 0.0).all()  # No change in front of charge
                assert (image_difference[2, 1:4] < 0.0).all()  # charge lost in charge

                assert (image_difference[3:5, 1:4] > 0.0).all()  # Parallel trails behind charge

                assert (image_difference[2:5, 0] == 0.0).all()  # no serial cti trail to left
                assert (image_difference[2:5, 4] > 0.0).all()  # serial cti trail to right including parallel cti trails

                assert (image_difference[3, 1:4] > image_difference[4,
                                                   1:4]).all()  # check parallel cti trails decreasing.

                assert (image_difference[3, 4] > image_difference[
                    4, 4])  # Check serial trails of paralle trails decreasing.

            def test__vertical_charge_line__loses_charge_trails_form_in_serial_directions(self, arctic_both,
                                                                                          params_both):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[1:4, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBL(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_both, cti_settings=arctic_both)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[0, 0:5] == 0.0).all()  # No change in front of charge
                assert (image_difference[1:4, 2] < 0.0).all()  # charge lost in charge

                assert (image_difference[4, 2] > 0.0).all()  # Parallel trail behind charge

                assert (image_difference[0:5, 0:2] == 0.0).all()  # no serial cti trail to left
                assert (image_difference[1:5,
                        3:5] > 0.0).all()  # serial cti trail to right including parallel cti trails

                assert (image_difference[3, 3] > image_difference[3, 4])  # Check serial trails decreasing.
                assert (image_difference[4, 3] > image_difference[
                    4, 4])  # Check serial trails of parallel trails decreasing.

            def test__individual_pixel_trails_form_cross_around_it(self, params_both, arctic_both):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBL(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_both, cti_settings=arctic_both)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[0:2, :] == 0.0).all()  # First two rows should all remain zero
                assert (image_difference[:, 0:2] == 0.0).all()  # First tow columns should all remain zero
                assert (image_difference[2, 2] < 0.0)  # pixel which had charge should lose it due to cti.
                assert (image_difference[3:5, 2] > 0.0).all()  # Parallel trail increases charge above pixel
                assert (image_difference[2, 3:5] > 0.0).all()  # Serial trail increases charge to right of pixel
                assert (image_difference[3:5,
                        3:5] > 0.0).all()  # Serial trailing of parallel trail increases charge up-right of pixel

            def test__individual_pixel_double_density__more_captures_so_brighter_trails(self, arctic_both):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBL(), array=image_pre_cti)

                params_0 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                                  p_well_notch_depth=0.0000001, p_well_fill_beta=0.8,
                                                  include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                                  s_well_notch_depth=0.0000001, s_well_fill_beta=0.8)

                image_post_cti_0 = im.add_cti_to_image(cti_params=params_0, cti_settings=arctic_both)

                params_1 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.2,), p_trap_lifetimes=(1.0,),
                                                  p_well_notch_depth=0.0000001, p_well_fill_beta=0.8,
                                                  include_serial=True, s_trap_densities=(0.2,), s_trap_lifetimes=(1.0,),
                                                  s_well_notch_depth=0.0000001, s_well_fill_beta=0.8)

                image_post_cti_1 = im.add_cti_to_image(cti_params=params_1, cti_settings=arctic_both)

                image_difference = image_post_cti_1 - image_post_cti_0

                assert (image_difference[0:2, :] == 0.0).all()  # First two rows remain zero
                assert (image_difference[:, 0:2] == 0.0).all()  # First tow columns remain zero
                assert (image_difference[
                            2, 2] < 0.0)  # More captures in second ci_pre_ctis, so more charge in central pixel lost
                assert (image_difference[3:5,
                        2] > 0.0).all()  # More captpures in ci_pre_ctis 2, so brighter parallel trail
                assert (image_difference[2,
                        3:5] > 0.0).all()  # More captpures in ci_pre_ctis 2, so brighter serial trail
                assert (image_difference[3:5, 3:5] > 0.0).all()  # Brighter serial trails from parallel trail trails

            def test__individual_pixel_increase_lifetime_longer_release_so_fainter_trails(self, arctic_both):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBL(), array=image_pre_cti)

                params_0 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                                  p_well_notch_depth=0.0000001, p_well_fill_beta=0.8,
                                                  include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                                  s_well_notch_depth=0.0000001, s_well_fill_beta=0.8)

                image_post_cti_0 = im.add_cti_to_image(cti_params=params_0, cti_settings=arctic_both)

                params_1 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(20.0,),
                                                  p_well_notch_depth=0.0000001, p_well_fill_beta=0.8,
                                                  include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(20.0,),
                                                  s_well_notch_depth=0.0000001, s_well_fill_beta=0.8)

                image_post_cti_1 = im.add_cti_to_image(cti_params=params_1, cti_settings=arctic_both)

                image_difference = image_post_cti_1 - image_post_cti_0

                assert (image_difference[0:2, :] == 0.0).all()  # First two rows remain zero
                assert (image_difference[:, 0:2] == 0.0).all()  # First tow columns remain zero
                assert (image_difference[2, 2] == 0.0)  # Same density so same captures
                assert (image_difference[3:5,
                        2] < 0.0).all()  # Longer release in ci_pre_ctis 2, so fainter parallel trail
                assert (image_difference[2, 3:5] < 0.0).all()  # Longer release in ci_pre_ctis 2, so fainter serial trail
                assert (image_difference[3:5,
                        3:5] < 0.0).all()  # Longer release in ci_pre_ctis 2, so fainter parallel trail trails

            def test__individual_pixel_increase_beta__fewer_captures_so_fainter_trails(self, arctic_both):

                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBL(), array=image_pre_cti)

                params_0 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                                  p_well_notch_depth=0.0000001, p_well_fill_beta=0.8,
                                                  include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                                  s_well_notch_depth=0.0000001, s_well_fill_beta=0.8)

                image_post_cti_0 = im.add_cti_to_image(cti_params=params_0, cti_settings=arctic_both)

                params_1 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                                  p_well_notch_depth=0.0000001, p_well_fill_beta=0.9,
                                                  include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                                  s_well_notch_depth=0.0000001, s_well_fill_beta=0.9)

                image_post_cti_1 = im.add_cti_to_image(cti_params=params_1, cti_settings=arctic_both)

                image_difference = image_post_cti_1 - image_post_cti_0

                assert (image_difference[0:2, :] == 0.0).all()  # First two rows remain zero
                assert (image_difference[:, 0:2] == 0.0).all()  # First tow columns remain zero
                assert (image_difference[2, 2] > 0.0)  # Higher beta in 2, so fewer captures
                assert (image_difference[3:5, 2] < 0.0).all()  # Fewer catprues in 2, so fainter parallel trail
                assert (image_difference[2, 3:5] < 0.0).all()  # Fewer captures in 2, so fainter serial trail
                assert (image_difference[3:5, 3:5] < 0.0).all()  # fewer captures in 2, so fainter trails trail region

    class TestCorrectCTI:
        class TestParallelCTI:

            def test__array_of_values__corrected_image_more_like_original(self, arctic_parallel, params_parallel):
                image_pre_cti = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 9.0, 5.0, 9.5, 3.2, 9.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 9.0, 5.0, 9.5, 352, 9.4, 0.0],
                                          [0.0, 9.0, 5.0, 9.5, 0.0, 9.0, 0.0],
                                          [0.0, 9.0, 9.1, 9.3, 9.2, 9.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBL(),
                                        array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_parallel, cti_settings=arctic_parallel)
                image_difference_1 = image_post_cti - image_pre_cti

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBL(),
                                        array=image_post_cti)

                image_correct_cti = im.correct_cti_from_image(cti_params=params_parallel, cti_settings=arctic_parallel)
                image_difference_2 = image_correct_cti - image_pre_cti

                assert (abs(image_difference_2) <= abs(image_difference_1)).all()

        class TestSerialCTI:

            def test__array_of_values__corrected_image_more_like_original(self, arctic_serial, params_serial):
                image_pre_cti = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 9.0, 5.0, 9.5, 3.2, 9.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 9.0, 5.0, 9.5, 352, 9.4, 0.0],
                                          [0.0, 9.0, 5.0, 9.5, 0.0, 9.0, 0.0],
                                          [0.0, 9.0, 9.1, 9.3, 9.2, 9.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBL(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_serial, cti_settings=arctic_serial)
                image_difference_1 = image_post_cti - image_pre_cti

                # Update ci_pre_ctis ci_data so that it corrects the cti added ci_pre_ctis
                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBL(),
                                        array=image_post_cti)

                image_correct_cti = im.correct_cti_from_image(cti_params=params_serial, cti_settings=arctic_serial)
                image_difference_2 = image_correct_cti - image_pre_cti

                assert (abs(image_difference_2) <= abs(image_difference_1)).all()

        # class TestSerialAndParallelCTI:
        # 
        #     def test__array_of_values__corrected_image_more_like_original(self, cti_params=params_both, cti_settings=arctic_both):
        # 
        #         pre_im = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #                                   [0.0, 9.0, 5.0, 9.5, 3.2, 9.0, 0.0],
        #                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #                                   [0.0, 9.0, 5.0, 9.5, 352, 9.4, 0.0],
        #                                   [0.0, 9.0, 5.0, 9.5, 0.0, 9.0, 0.0],
        #                                   [0.0, 9.0, 9.1, 9.3, 9.2, 9.0, 0.0],
        #                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        # 
        #         ci_image = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBL(), ci_pre_ctis=pre_im, euclid_checks=False)
        # 
        #         image_post_cti = ci_image.add_cti(cti_params=params_both, cti_settings=arctic_both)
        # 
        #         image_difference_1 = image_post_cti - pre_im
        # 
        #         image_correct_cti = ci_image.correct_cti(cti_params=params_both, cti_settings=arctic_both)
        # 
        #         image_difference_2 = image_correct_cti - pre_im
        # 
        #         assert (abs(image_difference_2) <= abs(image_difference_1)).all()


class TestQuadrantGeometryEuclidBottomRight:

    class TestAddCTI:
        class TestParallelCTI:

            def test__horizontal_charge_line__loses_charge_and_trails_form_below(self, arctic_parallel,
                                                                                 params_parallel):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, :] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBR(),
                                        array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_parallel, cti_settings=arctic_parallel)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[0:2, :] == 0.0).all()  # No trails closest to read out
                assert (image_difference[2, :] < 0.0).all()  # line loses charge
                assert (image_difference[3:5, :] > 0.0).all()  # Trails appear away from read-out

            def test__vertical_charge_line__loses_charge_no_trails(self, arctic_parallel, params_parallel):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[:, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBR(),
                                        array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_parallel, cti_settings=arctic_parallel)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[:, 0:2] == 0.0).all()  # No trails closest to read out
                assert (image_difference[:, 2] < 0.0).all()  # line loses charge
                assert (image_difference[:,
                        3:5] == 0.0).all()  # No trails away from read-out either, as wrong direction

        class TestSerialCTI:

            def test__horizontal_charge_line__loses_charge_no_trails(self, arctic_serial, params_serial):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, :] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBR(),
                                        array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_serial, cti_settings=arctic_serial)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[0:2, :] == 0.0).all()  # No trails closest to read out
                assert (image_difference[2, :] < 0.0).all()  # line loses charge
                assert (image_difference[3:5,
                        :] == 0.0).all()  # No trails away from read-out either, as wrong direction

            def test__vertical_charge_line__loses_charge_and_trails_form_to_left_of_line(self, arctic_serial,
                                                                                         params_serial):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[:, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBR(),
                                        array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_serial, cti_settings=arctic_serial)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[:, 0:2] > 0.0).all()  # No trails closest to read out
                assert (image_difference[:, 2] < 0.0).all()  # line loses charge
                assert (image_difference[:, 3:5] == 0.0).all()  # Trails appear away from read-out

        class TestParallelAndSerialCTI:

            def test__horizontal_charge_line__loses_charge_trails_form_both_directions(self, arctic_both,
                                                                                       params_both):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, 1:4] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBR(),
                                        array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_both, cti_settings=arctic_both)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[0:2, :] == 0.0).all()  # No change in front of charge
                assert (image_difference[2, 1:4] < 0.0).all()  # charge lost in charge

                assert (image_difference[3:5, 1:4] > 0.0).all()  # Parallel trails behind charge

                assert (image_difference[2:5, 0] > 0.0).all()  # serial cti trail to left including parallel trails
                assert (image_difference[2:5, 4] == 0.0).all()  # no trails to right

                assert (image_difference[3, 1:4] > image_difference[4,
                                                   1:4]).all()  # check parallel cti trails decreasing.

                assert (image_difference[3, 0] > image_difference[
                    4, 0])  # Check serial trails of paralle trails decreasing.

            def test__vertical_charge_line__loses_charge_trails_form_in_serial_direction(self, arctic_both,
                                                                                         params_both):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[1:4, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBR(),
                                        array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_both, cti_settings=arctic_both)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[0, 0:5] == 0.0).all()  # No change in front of charge
                assert (image_difference[1:4, 2] < 0.0).all()  # charge lost in charge

                assert (image_difference[4, 2] > 0.0).all()  # Parallel trail behind charge

                assert (image_difference[1:5,
                        0:2] > 0.0).all()  # serial cti trail is to the left including parallel trails
                assert (image_difference[0:5, 3:5] == 0.0).all()  # no serial cti trail to the right

                assert (image_difference[3, 1] > image_difference[3, 0])  # Check serial trails decreasing.
                assert (image_difference[4, 1] > image_difference[
                    4, 0])  # Check serial trails of parallel trails decreasing.

            def test__individual_pixel_trails_form_cross_around_it(self, params_both, arctic_both):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBR(),
                                        array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_both, cti_settings=arctic_both)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[0:2, :] == 0.0).all()  # First two rows should all remain zero
                assert (image_difference[:, 3:5] == 0.0).all()  # First tow columns should all remain zero
                assert (image_difference[2, 2] < 0.0)  # pixel which had charge should lose it due to cti.
                assert (image_difference[3:5, 2] > 0.0).all()  # Parallel trail increases charge above pixel
                assert (image_difference[2, 0:2] > 0.0).all()  # Serial trail increases charge to right of pixel
                assert (image_difference[3:5,
                        0:2] > 0.0).all()  # Serial trailing of parallel trail increases charge up-right of pixel

            def test__individual_pixel_double_density__more_captures_so_brighter_trails(self, arctic_both):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBR(),
                                        array=image_pre_cti)

                params_0 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                                  p_well_notch_depth=0.0000001, p_well_fill_beta=0.8,
                                                  include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                                  s_well_notch_depth=0.0000001, s_well_fill_beta=0.8)

                image_post_cti_0 = im.add_cti_to_image(cti_params=params_0, cti_settings=arctic_both)

                params_1 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.2,), p_trap_lifetimes=(1.0,),
                                                  p_well_notch_depth=0.0000001, p_well_fill_beta=0.8,
                                                  include_serial=True, s_trap_densities=(0.2,), s_trap_lifetimes=(1.0,),
                                                  s_well_notch_depth=0.0000001, s_well_fill_beta=0.8)

                image_post_cti_1 = im.add_cti_to_image(cti_params=params_1, cti_settings=arctic_both)

                image_difference = image_post_cti_1 - image_post_cti_0

                assert (image_difference[0:2, :] == 0.0).all()  # First two rows remain zero
                assert (image_difference[:, 3:5] == 0.0).all()  # First tow columns remain zero
                assert (image_difference[
                            2, 2] < 0.0)  # More captures in second ci_pre_ctis, so more charge in central pixel lost
                assert (image_difference[3:5,
                        2] > 0.0).all()  # More captpures in ci_pre_ctis 2, so brighter parallel trail
                assert (image_difference[2,
                        0:2] > 0.0).all()  # More captpures in ci_pre_ctis 2, so brighter serial trail
                assert (image_difference[3:5, 0:2] > 0.0).all()  # Brighter serial trails from parallel trail trails

            def test__individual_pixel_increase_lifetime_longer_release_so_fainter_trails(self, arctic_both):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBR(),
                                        array=image_pre_cti)

                params_0 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                                  p_well_notch_depth=0.0000001, p_well_fill_beta=0.8,
                                                  include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                                  s_well_notch_depth=0.0000001, s_well_fill_beta=0.8)

                image_post_cti_0 = im.add_cti_to_image(cti_params=params_0, cti_settings=arctic_both)

                params_1 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(20.0,),
                                                  p_well_notch_depth=0.0000001, p_well_fill_beta=0.8,
                                                  include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(20.0,),
                                                  s_well_notch_depth=0.0000001, s_well_fill_beta=0.8)

                image_post_cti_1 = im.add_cti_to_image(cti_params=params_1, cti_settings=arctic_both)

                image_difference = image_post_cti_1 - image_post_cti_0

                assert (image_difference[0:2, :] == 0.0).all()  # First two rows remain zero
                assert (image_difference[:, 3:5] == 0.0).all()  # First tow columns remain zero
                assert (image_difference[2, 2] == 0.0)  # Same density so same captures
                assert (image_difference[3:5,
                        2] < 0.0).all()  # Longer release in ci_pre_ctis 2, so fainter parallel trail
                assert (image_difference[2, 0:2] < 0.0).all()  # Longer release in ci_pre_ctis 2, so fainter serial trail
                assert (image_difference[3:5,
                        0:2] < 0.0).all()  # Longer release in ci_pre_ctis 2, so fainter parallel trail trails

            def test__individual_pixel_increase_beta__fewer_captures_so_fainter_trails(self, arctic_both):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBR(),
                                        array=image_pre_cti)

                params_0 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                                  p_well_notch_depth=0.0000001, p_well_fill_beta=0.8,
                                                  include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                                  s_well_notch_depth=0.0000001, s_well_fill_beta=0.8)

                image_post_cti_0 = im.add_cti_to_image(cti_params=params_0, cti_settings=arctic_both)

                params_1 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                                  p_well_notch_depth=0.0000001, p_well_fill_beta=0.9,
                                                  include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                                  s_well_notch_depth=0.0000001, s_well_fill_beta=0.9)

                image_post_cti_1 = im.add_cti_to_image(cti_params=params_1, cti_settings=arctic_both)

                image_difference = image_post_cti_1 - image_post_cti_0

                assert (image_difference[0:2, :] == 0.0).all()  # First two rows remain zero
                assert (image_difference[:, 3:5] == 0.0).all()  # First tow columns remain zero
                assert (image_difference[2, 2] > 0.0)  # Higher beta in 2, so fewer captures
                assert (image_difference[3:5, 2] < 0.0).all()  # Fewer catprues in 2, so fainter parallel trail
                assert (image_difference[2, 0:2] < 0.0).all()  # Fewer captures in 2, so fainter serial trail
                assert (image_difference[3:5, 0:2] < 0.0).all()  # fewer captures in 2, so fainter trails trail region

    class TestCorrectCTI:
        class TestParallelCTI:

            def test__array_of_values__corrected_image_more_like_original(self, arctic_parallel, params_parallel):
                image_pre_cti = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 9.0, 5.0, 9.5, 3.2, 9.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 9.0, 5.0, 9.5, 352, 9.4, 0.0],
                                          [0.0, 9.0, 5.0, 9.5, 0.0, 9.0, 0.0],
                                          [0.0, 9.0, 9.1, 9.3, 9.2, 9.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBR(),
                                        array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_parallel, cti_settings=arctic_parallel)
                image_difference_1 = image_post_cti - image_pre_cti

                # Update ci_pre_ctis ci_data so that it corrects the cti added ci_pre_ctis
                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBR(),
                                        array=image_post_cti)

                image_correct_cti = im.correct_cti_from_image(cti_params=params_parallel, cti_settings=arctic_parallel)
                image_difference_2 = image_correct_cti - image_pre_cti

                assert (abs(image_difference_2) <= abs(image_difference_1)).all()

        class TestSerialCTI:

            def test__array_of_values__corrected_image_more_like_original(self, arctic_serial, params_serial):
                image_pre_cti = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 9.0, 5.0, 9.5, 3.2, 9.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 9.0, 5.0, 9.5, 352, 9.4, 0.0],
                                          [0.0, 9.0, 5.0, 9.5, 0.0, 9.0, 0.0],
                                          [0.0, 9.0, 9.1, 9.3, 9.2, 9.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBR(),
                                        array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_serial, cti_settings=arctic_serial)
                image_difference_1 = image_post_cti - image_pre_cti

                # Update ci_pre_ctis ci_data so that it corrects the cti added ci_pre_ctis
                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBR(),
                                        array=image_post_cti)

                image_correct_cti = im.correct_cti_from_image(cti_params=params_serial, cti_settings=arctic_serial)
                image_difference_2 = image_correct_cti - image_pre_cti

                assert (abs(image_difference_2) <= abs(image_difference_1)).all()

        # class TestSerialAndParallelCTI:
        #
        #     def test__array_of_values__corrected_image_more_like_original(self, cti_params=params_both, cti_settings=arctic_both):
        #         pre_im = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #                                   [0.0, 9.0, 5.0, 9.5, 3.2, 9.0, 0.0],
        #                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #                                   [0.0, 9.0, 5.0, 9.5, 352, 9.4, 0.0],
        #                                   [0.0, 9.0, 5.0, 9.5, 0.0, 9.0, 0.0],
        #                                   [0.0, 9.0, 9.1, 9.3, 9.2, 9.0, 0.0],
        #                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        #
        #         ci_image = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidBR(), ci_pre_ctis=pre_im, euclid_checks=False)
        #
        #         image_post_cti = ci_image.add_cti_to_image(cti_params=params_both, cti_settings=arctic_both)
        #         image_difference_1 = image_post_cti - pre_im
        #
        #         ci_image = image_post_cti # Update ci_pre_ctis ci_data so that it corrects the cti added ci_pre_ctis
        #
        #         image_correct_cti = ci_image.correct_cti(cti_params=params_both, cti_settings=arctic_both)
        #         image_difference_2 = image_correct_cti - pre_im
        #
        #         assert (abs(image_difference_2) <= abs(image_difference_1)).all()


class TestQuadrantGeometryEuclidTopLeft:

    class TestAddCTI:
        class TestParallelCTI:

            def test__horizontal_charge_line__loses_charge_and_trails_form_above_line(self, arctic_parallel,
                                                                                      params_parallel):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, :] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTL(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_parallel, cti_settings=arctic_parallel)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[3:5, :] == 0.0).all()  # No trails closest to read out
                assert (image_difference[2, :] < 0.0).all()  # line loses charge
                assert (image_difference[0:2, :] > 0.0).all()  # trails appear behind line

            def test__vertical_charge_line__loses_charge_no_trails(self, arctic_parallel, params_parallel):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[:, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTL(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_parallel, cti_settings=arctic_parallel)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[:, 0:2] == 0.0).all()  # No trails closest to read out
                assert (image_difference[:, 3:-1] == 0.0).all()  # line loses charge
                assert (image_difference[:, 2] < 0.0).all()  # No trails away from read-out either, as wrong direction

        class TestSerialCTI:

            def test__horizontal_charge_line__loses_charge_no_trails(self, arctic_serial, params_serial):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, :] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTL(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_serial, cti_settings=arctic_serial)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[0:2, :] == 0.0).all()  # No trails closest to read out
                assert (image_difference[2, :] < 0.0).all()  # line loses charge
                assert (image_difference[3:5,
                        :] == 0.0).all()  # No trails away from read-out either, as wrong direction

            def test__vertical_charge_line__loses_charge_and_trails_form_to_right_of_line(self, arctic_serial,
                                                                                          params_serial):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[:, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTL(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_serial, cti_settings=arctic_serial)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[:, 0:2] == 0.0).all()  # No trails closest to read out
                assert (image_difference[:, 2] < 0.0).all()  # line loses charge
                assert (image_difference[:, 3:5] > 0.0).all()  # Trails appear away from read-out

        class TestParallelAndSerialCTI:

            def test__horizontal_charge_line__loses_charge_trails_form_both_directions(self, arctic_both,
                                                                                       params_both):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, 1:4] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTL(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_both, cti_settings=arctic_both)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[3:5, :] == 0.0).all()  # No change in front of charge
                assert (image_difference[2, 1:4] < 0.0).all()  # charge lost in charge

                assert (image_difference[0:2, 1:4] > 0.0).all()  # Parallel trails behind charge

                assert (image_difference[0:5, 0] == 0.0).all()  # no serial cti trail to left
                assert (image_difference[0:3, 4] > 0.0).all()  # serial cti trail to right including parallel cti trails

                assert (image_difference[1, 1:4] > image_difference[0,
                                                   1:4]).all()  # check parallel cti trails decreasing.

                assert (image_difference[1, 4] > image_difference[
                    0, 4])  # Check serial trails of paralle trails decreasing.

            def test__vertical_charge_line__loses_charge_trails_form_in_serial_direction(self, arctic_both,
                                                                                         params_both):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[1:4, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTL(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_both, cti_settings=arctic_both)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[4, 0:5] == 0.0).all()  # No change in front of charge
                assert (image_difference[1:4, 2] < 0.0).all()  # charge lost in charge

                assert (image_difference[0, 2] > 0.0).all()  # Parallel trail behind charge

                assert (image_difference[1:5, 0:2] == 0.0).all()  # no serial cti trail to left
                assert (image_difference[0:4,
                        3:5] > 0.0).all()  # serial cti trail to right including parallel cti trails

                assert (image_difference[3, 3] > image_difference[3, 4])  # Check serial trails decreasing.
                assert (image_difference[0, 3] > image_difference[
                    0, 4])  # Check serial trails of parallel trails decreasing.

            def test__individual_pixel__trails_form_cross_around_it(self, params_both, arctic_both):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTL(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_both, cti_settings=arctic_both)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[3:5, :] == 0.0).all()  # First two rows should all remain zero
                assert (image_difference[:, 0:2] == 0.0).all()  # First tow columns should all remain zero
                assert (image_difference[2, 2] < 0.0)  # pixel which had charge should lose it due to cti.
                assert (image_difference[0:2, 2] > 0.0).all()  # Parallel trail increases charge above pixel
                assert (image_difference[2, 3:5] > 0.0).all()  # Serial trail increases charge to right of pixel
                assert (image_difference[0:2,
                        3:5] > 0.0).all()  # Serial trailing of parallel trail increases charge up-right of pixel

            def test__individual_pixel_double_density__more_captures_so_brighter_trails(self, arctic_both):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTL(), array=image_pre_cti)

                params_0 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                                  p_well_notch_depth=0.0000001, p_well_fill_beta=0.8,
                                                  include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                                  s_well_notch_depth=0.0000001, s_well_fill_beta=0.8)

                image_post_cti_0 = im.add_cti_to_image(cti_params=params_0, cti_settings=arctic_both)

                params_1 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.2,), p_trap_lifetimes=(1.0,),
                                                  p_well_notch_depth=0.0000001, p_well_fill_beta=0.8,
                                                  include_serial=True, s_trap_densities=(0.2,), s_trap_lifetimes=(1.0,),
                                                  s_well_notch_depth=0.0000001, s_well_fill_beta=0.8)

                image_post_cti_1 = im.add_cti_to_image(cti_params=params_1, cti_settings=arctic_both)

                image_difference = image_post_cti_1 - image_post_cti_0

                assert (image_difference[3:5, :] == 0.0).all()  # First two rows remain zero
                assert (image_difference[:, 0:2] == 0.0).all()  # First tow columns remain zero
                assert (image_difference[
                            2, 2] < 0.0)  # More captures in second ci_pre_ctis, so more charge in central pixel lost
                assert (image_difference[0:2,
                        2] > 0.0).all()  # More captpures in ci_pre_ctis 2, so brighter parallel trail
                assert (image_difference[2,
                        3:5] > 0.0).all()  # More captpures in ci_pre_ctis 2, so brighter serial trail
                assert (image_difference[0:2, 3:5] > 0.0).all()  # Brighter serial trails from parallel trail trails

            def test__individual_pixel_increase_lifetime_longer_release_so_fainter_trails(self, arctic_both):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTL(), array=image_pre_cti)

                params_0 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                                  p_well_notch_depth=0.0000001, p_well_fill_beta=0.8,
                                                  include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                                  s_well_notch_depth=0.0000001, s_well_fill_beta=0.8)

                image_post_cti_0 = im.add_cti_to_image(cti_params=params_0, cti_settings=arctic_both)

                params_1 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(20.0,),
                                                  p_well_notch_depth=0.0000001, p_well_fill_beta=0.8,
                                                  include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(20.0,),
                                                  s_well_notch_depth=0.0000001, s_well_fill_beta=0.8)

                image_post_cti_1 = im.add_cti_to_image(cti_params=params_1, cti_settings=arctic_both)

                image_difference = image_post_cti_1 - image_post_cti_0

                assert (image_difference[3:5, :] == 0.0).all()  # First two rows remain zero
                assert (image_difference[:, 0:2] == 0.0).all()  # First tow columns remain zero
                assert (image_difference[2, 2] == 0.0)  # Same density so same captures
                assert (image_difference[0:2,
                        2] < 0.0).all()  # Longer release in ci_pre_ctis 2, so fainter parallel trail
                assert (image_difference[2, 3:5] < 0.0).all()  # Longer release in ci_pre_ctis 2, so fainter serial trail
                assert (image_difference[0:2,
                        3:5] < 0.0).all()  # Longer release in ci_pre_ctis 2, so fainter parallel trail trails

            def test__individual_pixel_increase_beta__fewer_captures_so_fainter_trails(self, arctic_both):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTL(), array=image_pre_cti)

                params_0 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                                  p_well_notch_depth=0.0000001, p_well_fill_beta=0.8,
                                                  include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                                  s_well_notch_depth=0.0000001, s_well_fill_beta=0.8)

                image_post_cti_0 = im.add_cti_to_image(cti_params=params_0, cti_settings=arctic_both)

                params_1 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                                  p_well_notch_depth=0.0000001, p_well_fill_beta=0.9,
                                                  include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                                  s_well_notch_depth=0.0000001, s_well_fill_beta=0.9)

                image_post_cti_1 = im.add_cti_to_image(cti_params=params_1, cti_settings=arctic_both)

                image_difference = image_post_cti_1 - image_post_cti_0

                assert (image_difference[3:5, :] == 0.0).all()  # First two rows remain zero
                assert (image_difference[:, 0:2] == 0.0).all()  # First tow columns remain zero
                assert (image_difference[2, 2] > 0.0)  # Higher beta in 2, so fewer captures
                assert (image_difference[0:2, 2] < 0.0).all()  # Fewer catprues in 2, so fainter parallel trail
                assert (image_difference[2, 3:5] < 0.0).all()  # Fewer captures in 2, so fainter serial trail
                assert (image_difference[0:2, 3:5] < 0.0).all()  # fewer captures in 2, so fainter trails trail region

    class TestCorrectCTI:
        class TestParallelCTI:

            def test__array_of_values__corrected_image_more_like_original(self, arctic_parallel, params_parallel):
                image_pre_cti = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 9.0, 5.0, 9.5, 3.2, 9.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 9.0, 5.0, 9.5, 352, 9.4, 0.0],
                                          [0.0, 9.0, 5.0, 9.5, 0.0, 9.0, 0.0],
                                          [0.0, 9.0, 9.1, 9.3, 9.2, 9.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTL(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_parallel, cti_settings=arctic_parallel)
                image_difference_1 = image_post_cti - image_pre_cti

                # Update ci_pre_ctis ci_data so that it corrects the cti added ci_pre_ctis
                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTL(),
                                        array=image_post_cti)

                image_correct_cti = im.correct_cti_from_image(cti_params=params_parallel, cti_settings=arctic_parallel)
                image_difference_2 = image_correct_cti - image_pre_cti

                assert (abs(image_difference_2) <= abs(image_difference_1)).all()

        class TestSerialCTI:

            def test__array_of_values__corrected_image_more_like_original(self, arctic_serial, params_serial):
                image_pre_cti = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 9.0, 5.0, 9.5, 3.2, 9.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 9.0, 5.0, 9.5, 352, 9.4, 0.0],
                                          [0.0, 9.0, 5.0, 9.5, 0.0, 9.0, 0.0],
                                          [0.0, 9.0, 9.1, 9.3, 9.2, 9.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTL(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_serial, cti_settings=arctic_serial)
                image_difference_1 = image_post_cti - image_pre_cti

                # Update ci_pre_ctis ci_data so that it corrects the cti added ci_pre_ctis
                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTL(),
                                        array=image_post_cti)

                image_correct_cti = im.correct_cti_from_image(cti_params=params_serial, cti_settings=arctic_serial)
                image_difference_2 = image_correct_cti - image_pre_cti

                assert (abs(image_difference_2) <= abs(image_difference_1)).all()

        # class TestSerialAndParallelCTI:
        #
        #     def test__array_of_values__corrected_image_more_like_original(self, cti_params=params_both, cti_settings=arctic_both):
        #         pre_im = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #                                   [0.0, 9.0, 5.0, 9.5, 3.2, 9.0, 0.0],
        #                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #                                   [0.0, 9.0, 5.0, 9.5, 352, 9.4, 0.0],
        #                                   [0.0, 9.0, 5.0, 9.5, 0.0, 9.0, 0.0],
        #                                   [0.0, 9.0, 9.1, 9.3, 9.2, 9.0, 0.0],
        #                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        #
        #         ci_image = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTL(), ci_pre_ctis=pre_im, euclid_checks=False)
        #
        #         image_post_cti = ci_image.add_cti_to_image(cti_params=params_both, cti_settings=arctic_both)
        #         image_difference_1 = image_post_cti - pre_im
        #
        #         ci_image = image_post_cti # Update ci_pre_ctis ci_data so that it corrects the cti added ci_pre_ctis
        #
        #         image_correct_cti = ci_image.correct_cti(cti_params=params_both, cti_settings=arctic_both)
        #         image_difference_2 = image_correct_cti - pre_im
        #
        #         assert (abs(image_difference_2) <= abs(image_difference_1)).all()


class TestQuadrantGeometryEuclidTopRight:

    class TestAddCTI:
        class TestParallelCTI:

            def test__horizontal_charge_line__loses_charge_and_trails_form_above_line(self, arctic_parallel,
                                                                                      params_parallel):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, :] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTR(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_parallel, cti_settings=arctic_parallel)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[3:5, :] == 0.0).all()  # No trails closest to read out
                assert (image_difference[2, :] < 0.0).all()  # line loses charge
                assert (image_difference[0:2, :] > 0.0).all()  # Trails appear away from read-out

            def test__vertical_charge_line__loses_charge_no_trails(self, arctic_parallel, params_parallel):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[:, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTR(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_parallel, cti_settings=arctic_parallel)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[:, 0:2] == 0.0).all()  # No trails closest to read out
                assert (image_difference[:, 2] < 0.0).all()  # line loses charge
                assert (image_difference[:,
                        3:5] == 0.0).all()  # No trails away from read-out either, as wrong direction

        class TestSerialCTI:

            def test__horizontal_charge_line__loses_charge_no_trails(self, arctic_serial, params_serial):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, :] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTR(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_serial, cti_settings=arctic_serial)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[0:2, :] == 0.0).all()  # No trails closest to read out
                assert (image_difference[2, :] < 0.0).all()  # line loses charge
                assert (image_difference[3:5,
                        :] == 0.0).all()  # No trails away from read-out either, as wrong direction

            def test__vertical_charge_line__loses_charge_and_trails_form_to_right_of_line(self, arctic_serial,
                                                                                          params_serial):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[:, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTR(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_serial, cti_settings=arctic_serial)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[:, 0:2] > 0.0).all()  # No trails closest to read out
                assert (image_difference[:, 2] < 0.0).all()  # line loses charge
                assert (image_difference[:, 3:5] == 0.0).all()  # Trails appear away from read-out

        class TestParallelAndSerialCTI:

            def test__horizontal_charge_line__loses_charge_trails_form_both_directions(self, arctic_both,
                                                                                       params_both):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, 1:4] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTR(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_both, cti_settings=arctic_both)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[3:5, :] == 0.0).all()  # No change in front of charge
                assert (image_difference[2, 1:4] < 0.0).all()  # charge lost in charge

                assert (image_difference[0:2, 1:4] > 0.0).all()  # Parallel trails behind charge

                assert (image_difference[0:2, 0] > 0.0).all()  # serial cti trail to left including parallel trails
                assert (image_difference[2:5, 4] == 0.0).all()  # no trails to right

                assert (image_difference[1, 1:4] > image_difference[0,
                                                   1:4]).all()  # check parallel cti trails decreasing.

                assert (image_difference[1, 0] > image_difference[
                    0, 0])  # Check serial trails of paralle trails decreasing.

            def test__vertical_charge_line__loses_charge_trails_form_in_serial_direction(self, arctic_both,
                                                                                         params_both):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[1:4, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTR(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_both, cti_settings=arctic_both)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[4, 0:5] == 0.0).all()  # No change in front of charge
                assert (image_difference[1:4, 2] < 0.0).all()  # charge lost in charge

                assert (image_difference[0, 2] > 0.0).all()  # Parallel trail behind charge

                assert (image_difference[0:4,
                        0:2] > 0.0).all()  # serial cti trail is to the left including parallel trails
                assert (image_difference[0:5, 3:5] == 0.0).all()  # no serial cti trail to the right

                assert (image_difference[2, 1] > image_difference[2, 0])  # Check serial trails decreasing.
                assert (image_difference[1, 1] > image_difference[
                    1, 0])  # Check serial trails of parallel trails decreasing.

            def test__individual_pixel__trails_form_cross_around_it(self, params_both, arctic_both):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTR(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_both, cti_settings=arctic_both)

                image_difference = image_post_cti - image_pre_cti

                assert (image_difference[3:5, :] == 0.0).all()  # First two rows should all remain zero
                assert (image_difference[:, 3:5] == 0.0).all()  # First tow columns should all remain zero
                assert (image_difference[2, 2] < 0.0)  # pixel which had charge should lose it due to cti.
                assert (image_difference[0:2, 2] > 0.0).all()  # Parallel trail increases charge above pixel
                assert (image_difference[2, 0:2] > 0.0).all()  # Serial trail increases charge to right of pixel
                assert (image_difference[0:2,
                        0:2] > 0.0).all()  # Serial trailing of parallel trail increases charge up-right of pixel

            def test__individual_pixel_double_density__more_captures_so_brighter_trails(self, arctic_both):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTR(), array=image_pre_cti)

                params_0 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                                  p_well_notch_depth=0.0000001, p_well_fill_beta=0.8,
                                                  include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                                  s_well_notch_depth=0.0000001, s_well_fill_beta=0.8)

                image_post_cti_0 = im.add_cti_to_image(cti_params=params_0, cti_settings=arctic_both)

                params_1 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.2,), p_trap_lifetimes=(1.0,),
                                                  p_well_notch_depth=0.0000001, p_well_fill_beta=0.8,
                                                  include_serial=True, s_trap_densities=(0.2,), s_trap_lifetimes=(1.0,),
                                                  s_well_notch_depth=0.0000001, s_well_fill_beta=0.8)

                image_post_cti_1 = im.add_cti_to_image(cti_params=params_1, cti_settings=arctic_both)

                image_difference = image_post_cti_1 - image_post_cti_0

                assert (image_difference[3:5, :] == 0.0).all()  # First two rows remain zero
                assert (image_difference[:, 3:5] == 0.0).all()  # First tow columns remain zero
                assert (image_difference[
                            2, 2] < 0.0)  # More captures in second ci_pre_ctis, so more charge in central pixel lost
                assert (image_difference[0:2,
                        2] > 0.0).all()  # More captpures in ci_pre_ctis 2, so brighter parallel trail
                assert (image_difference[2,
                        0:2] > 0.0).all()  # More captpures in ci_pre_ctis 2, so brighter serial trail
                assert (image_difference[0:2, 0:2] > 0.0).all()  # Brighter serial trails from parallel trail trails

            def test__individual_pixel_increase_lifetime_longer_release_so_fainter_trails(self, arctic_both):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTR(), array=image_pre_cti)

                params_0 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                                  p_well_notch_depth=0.0000001, p_well_fill_beta=0.8,
                                                  include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                                  s_well_notch_depth=0.0000001, s_well_fill_beta=0.8)

                image_post_cti_0 = im.add_cti_to_image(cti_params=params_0, cti_settings=arctic_both)

                params_1 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(20.0,),
                                                  p_well_notch_depth=0.0000001, p_well_fill_beta=0.8,
                                                  include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(20.0,),
                                                  s_well_notch_depth=0.0000001, s_well_fill_beta=0.8)

                image_post_cti_1 = im.add_cti_to_image(cti_params=params_1, cti_settings=arctic_both)

                image_difference = image_post_cti_1 - image_post_cti_0

                assert (image_difference[3:5, :] == 0.0).all()  # First two rows remain zero
                assert (image_difference[:, 3:5] == 0.0).all()  # First tow columns remain zero
                assert (image_difference[2, 2] == 0.0)  # Same density so same captures
                assert (image_difference[0:2,
                        2] < 0.0).all()  # Longer release in ci_pre_ctis 2, so fainter parallel trail
                assert (image_difference[2, 0:2] < 0.0).all()  # Longer release in ci_pre_ctis 2, so fainter serial trail
                assert (image_difference[0:2,
                        0:2] < 0.0).all()  # Longer release in ci_pre_ctis 2, so fainter parallel trail trails

            def test__individual_pixel_increase_beta__fewer_captures_so_fainter_trails(self, arctic_both):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, 2] = + 100

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTR(), array=image_pre_cti)

                params_0 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                                  p_well_notch_depth=0.0000001, p_well_fill_beta=0.8,
                                                  include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                                  s_well_notch_depth=0.0000001, s_well_fill_beta=0.8)

                image_post_cti_0 = im.add_cti_to_image(cti_params=params_0, cti_settings=arctic_both)

                params_1 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                                  p_well_notch_depth=0.0000001, p_well_fill_beta=0.9,
                                                  include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                                  s_well_notch_depth=0.0000001, s_well_fill_beta=0.9)

                image_post_cti_1 = im.add_cti_to_image(cti_params=params_1, cti_settings=arctic_both)

                image_difference = image_post_cti_1 - image_post_cti_0

                assert (image_difference[3:5, :] == 0.0).all()  # First two rows remain zero
                assert (image_difference[:, 3:5] == 0.0).all()  # First tow columns remain zero
                assert (image_difference[2, 2] > 0.0)  # Higher beta in 2, so fewer captures
                assert (image_difference[0:2, 2] < 0.0).all()  # Fewer catprues in 2, so fainter parallel trail
                assert (image_difference[2, 0:2] < 0.0).all()  # Fewer captures in 2, so fainter serial trail
                assert (image_difference[0:2, 0:2] < 0.0).all()  # fewer captures in 2, so fainter trails trail region

    class TestCorrectCTI:
        class TestParallelCTI:

            def test__array_of_values__corrected_image_more_like_original(self, arctic_parallel, params_parallel):
                image_pre_cti = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 9.0, 5.0, 9.5, 3.2, 9.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 9.0, 5.0, 9.5, 352, 9.4, 0.0],
                                          [0.0, 9.0, 5.0, 9.5, 0.0, 9.0, 0.0],
                                          [0.0, 9.0, 9.1, 9.3, 9.2, 9.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTR(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_parallel, cti_settings=arctic_parallel)
                image_difference_1 = image_post_cti - image_pre_cti

                # Update ci_pre_ctis ci_data so that it corrects the cti added ci_pre_ctis
                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTR(),
                                        array=image_post_cti)

                image_correct_cti = im.correct_cti_from_image(cti_params=params_parallel, cti_settings=arctic_parallel)
                image_difference_2 = image_correct_cti - image_pre_cti

                assert (abs(image_difference_2) <= abs(image_difference_1)).all()

        class TestSerialCTI:

            def test__array_of_values__corrected_image_more_like_original(self, arctic_serial, params_serial):
                image_pre_cti = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 9.0, 5.0, 9.5, 3.2, 9.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 9.0, 5.0, 9.5, 352, 9.4, 0.0],
                                          [0.0, 9.0, 5.0, 9.5, 0.0, 9.0, 0.0],
                                          [0.0, 9.0, 9.1, 9.3, 9.2, 9.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTR(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_serial, cti_settings=arctic_serial)
                image_difference_1 = image_post_cti - image_pre_cti

                # Update ci_pre_ctis ci_data so that it corrects the cti added ci_pre_ctis
                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTR(),
                                        array=image_post_cti)

                image_correct_cti = im.correct_cti_from_image(cti_params=params_serial, cti_settings=arctic_serial)
                image_difference_2 = image_correct_cti - image_pre_cti

                assert (abs(image_difference_2) <= abs(image_difference_1)).all()

        class TestSerialAndParallelCTI:

            def test__simple_case__corrected_image_more_like_original(self, params_both, arctic_both):
                image_pre_cti = np.zeros((5, 5))
                image_pre_cti[2, 2] = 1000.

                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTR(), array=image_pre_cti)

                image_post_cti = im.add_cti_to_image(cti_params=params_both, cti_settings=arctic_both)
                image_difference_1 = image_post_cti - image_pre_cti

                # Update ci_pre_ctis ci_data so that it corrects the cti added ci_pre_ctis
                im = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTR(),
                                        array=image_post_cti)

                image_correct_cti = im.correct_cti_from_image(cti_params=params_both, cti_settings=arctic_both)
                image_difference_2 = image_correct_cti - image_pre_cti

                assert (abs(image_difference_2) <= abs(image_difference_1)).all()

            # def test__array_of_values__corrected_image_more_like_original(self, cti_params=params_both, cti_settings=arctic_both):
            #
            #     pre_im = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                               [0.0, 9.0, 5.0, 9.5, 3.2, 9.0, 0.0],
            #                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                               [0.0, 9.0, 5.0, 9.5, 352, 9.4, 0.0],
            #                               [0.0, 9.0, 5.0, 9.5, 0.0, 9.0, 0.0],
            #                               [0.0, 9.0, 9.1, 9.3, 9.2, 9.0, 0.0],
            #                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            #
            #     ci_image = cti_image.CTIImage(frame_geometry=cti_image.QuadGeometryEuclidTR(), ci_pre_ctis=pre_im, euclid_checks=False)
            #
            #     image_post_cti = ci_image.add_cti_to_image(cti_params=params_both, cti_settings=arctic_both)
            #     image_difference_1 = image_post_cti - pre_im
            #
            #     ci_image = image_post_cti # Update ci_pre_ctis ci_data so that it corrects the cti added ci_pre_ctis
            #
            #     image_correct_cti = ci_image.correct_cti(cti_params=params_both, cti_settings=arctic_both)
            #     image_difference_2 = image_correct_cti - pre_im
            #
            #     assert (abs(image_difference_2) <= abs(image_difference_1)).all()