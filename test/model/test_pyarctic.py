#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
#
# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3.0 of the License, or (at your option)
# any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
#

"""
File: tests/python/PyArCTIC_test.py

Created on: 02/13/18
Author: James Nightingale
"""

import pytest
import numpy as np

from autocti.model import pyarctic
from autocti.model import arctic_settings
from autocti.model import arctic_params


@pytest.fixture(scope='class')
def arctic_parallel():
    parallel_settings = arctic_settings.ParallelSettings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                                         charge_injection_mode=True, readout_offset=0)

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
    parallel_settings = arctic_settings.ParallelSettings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                                         readout_offset=0)

    serial_settings = arctic_settings.SerialSettings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                                     readout_offset=0)

    arctic_both = arctic_settings.ArcticSettings(neomode='NEO', parallel=parallel_settings,
                                                 serial=serial_settings)

    return arctic_both


class TestArcticAddCTI:
    class TestParallelCTI:

        def test__horizontal_line__line_loses_charge_trails_appear(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_alpha=1.0, p_well_fill_beta=0.8,
                                             p_well_fill_gamma=0.0)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[0:2, :] == 0.0).all()  # First four rows should all remain zero
            assert (image_difference[2,
                    :] < 0.0).all()  # All pixels in the charge line should lose charge due to capture
            assert (image_difference[3:-1, :] > 0.0).all()  # All other pixels should have charge trailed into them

        def test__vertical_line__no_trails(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[:, 2] += 100

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_alpha=1.0, p_well_fill_beta=0.8,
                                             p_well_fill_gamma=0.0)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[:, 0:2] == 0.0).all()  # Most pixels unchaged
            assert (image_difference[:, 3:-1] == 0.0).all()
            assert (image_difference[:, 2] < 0.0).all()  # charge line still loses charge

        def test__double_trap_density__more_captures_so_brighter_trails(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            ### SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED ###

            parameters_0 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                               p_well_notch_depth=0.01, p_well_fill_alpha=1.0, p_well_fill_beta=0.8,
                                               p_well_fill_gamma=0.0)

            parameters_1 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.2,), p_trap_lifetimes=(1.0,),
                                               p_well_notch_depth=0.01, p_well_fill_alpha=1.0, p_well_fill_beta=0.8,
                                               p_well_fill_gamma=0.0)

            ### NOW GENERATE THE IMAGE POST CTI OF EACH SET

            image_post_cti_0 = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters_0,
                                                                  settings=arctic_parallel)
            image_post_cti_1 = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters_1,
                                                                  settings=arctic_parallel)

            assert (image_post_cti_0[0:2, :] == 0.0).all()  # First four rows should all remain zero
            assert (image_post_cti_1[0:2, :] == 0.0).all()  # First four rows should all remain zero
            assert (image_post_cti_0[2, :] > image_post_cti_1[4, :]).all()  # charge line loses less charge in image 1
            assert (image_post_cti_0[3:-1, :] < image_post_cti_1[5:-1,
                                                :]).all()  # fewer pixels trailed behind in image 2

        def test__double_trap_lifetime__longer_release_so_fainter_trails(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            ### SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED ###

            parameters_0 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                               p_well_notch_depth=0.01, p_well_fill_alpha=1.0, p_well_fill_beta=0.8,
                                               p_well_fill_gamma=0.0)

            parameters_1 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(2.0,),
                                               p_well_notch_depth=0.01, p_well_fill_alpha=1.0, p_well_fill_beta=0.8,
                                               p_well_fill_gamma=0.0)
            ### NOW GENERATE THE IMAGE POST CTI OF EACH SET

            image_post_cti_0 = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters_0,
                                                                  settings=arctic_parallel)
            image_post_cti_1 = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters_1,
                                                                  settings=arctic_parallel)

            assert (image_post_cti_0[0:2, :] == 0.0).all()  # First four rows should all remain zero
            assert (image_post_cti_1[0:2, :] == 0.0).all()  # First four rows should all remain zero
            assert (image_post_cti_0[2, :] == image_post_cti_1[2, :]).all()  # charge line loses equal amount of charge
            assert (image_post_cti_0[3:-1, :] > image_post_cti_1[3:-1,
                                                :]).all()  # each trail in pixel 2 is 'longer' so fainter

        # def test__increase_alpha__more_captures_and_brighter_trails(self, arctic_parallel):
        #
        #     image_pre_cti = np.zeros((5, 5))
        #     image_pre_cti[2, :] += 100
        #
        #     ### SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED ###
        #
        #     parameters_0 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
        #                                       p_well_notch_depth=0.01, p_well_fill_alpha=0.8, p_well_fill_beta=0.2,
        #                                       p_well_fill_gamma=0.1)
        #
        #     parameters_1 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
        #                                       p_well_notch_depth=0.01, p_well_fill_alpha=1.0, p_well_fill_beta=0.2,
        #                                       p_well_fill_gamma=0.1)
        #
        #     ### NOW GENERATE THE IMAGE POST CTI OF EACH SET
        #
        #     image_post_cti_0 = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters_0,
        #                                                           settings=arctic_parallel)
        #     image_post_cti_1 = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters_1,
        #                                                           settings=arctic_parallel)
        #
        #     assert (image_post_cti_0[0:2, :] == 0.0).all()  # First four rows should all remain zero
        #     assert (image_post_cti_1[0:2, :] == 0.0).all()  # First four rows should all remain zero
        #     assert (image_post_cti_0[2, :] > image_post_cti_1[2,:]).all()  # charge line loses less charge with higher beta
        #     assert (image_post_cti_0[3:-1, :] < image_post_cti_1[3:-1, :]).all()  # so less electrons trailed into image

        def test__increase_beta__fewer_captures_fainter_trail(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            ### SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED ###

            parameters_0 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            parameters_1 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                               p_well_notch_depth=0.01, p_well_fill_beta=0.9)

            ### NOW GENERATE THE IMAGE POST CTI OF EACH SET

            image_post_cti_0 = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters_0,
                                                                  settings=arctic_parallel)
            image_post_cti_1 = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters_1,
                                                                  settings=arctic_parallel)

            assert (image_post_cti_0[0:2, :] == 0.0).all()  # First four rows should all remain zero
            assert (image_post_cti_1[0:2, :] == 0.0).all()  # First four rows should all remain zero
            assert (image_post_cti_0[2, :] < image_post_cti_1[2,
                                             :]).all()  # charge line loses less charge with higher beta
            assert (image_post_cti_0[3:-1, :] > image_post_cti_1[3:-1, :]).all()  # so less electrons trailed into image

        # def test__increase_gamma__more_captures_so_brighter_trails(self, arctic_parallel):
        #
        #     image_pre_cti = np.zeros((5, 5))
        #     image_pre_cti[2, :] += 100
        #
        #     ### SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED ###
        #
        #     parameters_0 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
        #                                       p_well_notch_depth=0.01, p_well_fill_alpha=1.0, p_well_fill_beta=0.8,
        #                                       p_well_fill_gamma=0.0)
        #
        #     parameters_1 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
        #                                       p_well_notch_depth=0.01, p_well_fill_alpha=1.0, p_well_fill_beta=0.8,
        #                                       p_well_fill_gamma=0.2)
        #
        #     ### NOW GENERATE THE IMAGE POST CTI OF EACH SET
        #
        #     image_post_cti_0 = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, cti_params=parameters_0,
        #                                                           settings=arctic_parallel)
        #     image_post_cti_1 = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, cti_params=parameters_1,
        #                                                           settings=arctic_parallel)
        #
        #     assert (image_post_cti_0[0:2, :] == 0.0).all()  # First four rows should all remain zero
        #     assert (image_post_cti_1[0:2, :] == 0.0).all()  # First four rows should all remain zero
        #     assert (image_post_cti_0[2, :] > image_post_cti_1[2,:]).all()  # charge line loses less charge with higher beta
        #     assert (image_post_cti_0[3:-1, :] < image_post_cti_1[3:-1, :]).all()  # so less electrons trailed into image

        def test__two_traps_half_density_of_one__same_trails(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            ### SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED ###

            parameters_0 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            parameters_1 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.05, 0.05),
                                               p_trap_lifetimes=(1.0, 1.0),
                                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            ### NOW GENERATE THE IMAGE POST CTI OF EACH SET

            image_post_cti_0 = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters_0,
                                                                  settings=arctic_parallel)
            image_post_cti_1 = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters_1,
                                                                  settings=arctic_parallel)

            assert (image_post_cti_0 == image_post_cti_1).all()

        def test__three_traps_third_density_of_one__same_trails(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            ### SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED ###

            parameters_0 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.15,), p_trap_lifetimes=(1.0,),
                                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            parameters_1 = arctic_params.setup(include_parallel=True, p_trap_densities=(0.05, 0.05, 0.05,),
                                               p_trap_lifetimes=(1.0, 1.0, 1.0),
                                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            ### NOW GENERATE THE IMAGE POST CTI OF EACH SET

            image_post_cti_0 = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters_0,
                                                                  settings=arctic_parallel)
            image_post_cti_1 = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters_1,
                                                                  settings=arctic_parallel)
            image_difference = abs(image_post_cti_0 - image_post_cti_1)

            assert (image_difference < 1e-9).all()

        def test__delta_functions__add_cti_only_behind_them(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[1, 1] += 100  # Delta 1
            image_pre_cti[3, 3] += 100  # Delta 2
            image_pre_cti[2, 4] += 100  # Delta 3

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[:, 0] == 0.0).all()  # No Delta, no charge

            assert (image_difference[0, 1] == 0.0)  # No charge in front of Delta 1
            assert (image_difference[1, 1] < 0.0)  # Delta 1 loses charge
            assert (image_difference[2:5, 1] > 0.0).all()  # Delta 1 trails

            assert (image_difference[:, 2] == 0.0).all()  # No Delta, no charge

            assert (image_difference[0:3, 3] == 0.0).all()  # No charge in front of Delta 2
            assert image_difference[3, 3] < 0.0  # Delta 2 loses charge
            assert image_difference[4, 3] > 0.0  # Delta 2 trail

            assert (image_difference[0:2, 4] == 0.0).all()  # No charge in front of Delta 3
            assert image_difference[2, 4] < 0.0  # Delta 3 loses charge
            assert (image_difference[3:5, 4] > 0.0).all()  # Delta 3 trail

    class TestSerialCTI:

        def test__horizontal_line__line_loses_charge_trails_appear(self, arctic_serial):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            parameters = arctic_params.setup(include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                             s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
                                             s_well_fill_gamma=0.0)

            image_post_cti = pyarctic.add_serial_cti_to_image(image=image_pre_cti, params=parameters,
                                                              settings=arctic_serial)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[0:2, :] == 0.0).all()  # First four rows should all remain zero
            assert (image_difference[2,
                    :] < 0.0).all()  # All pixels in the charge line should lose charge due to capture
            assert (image_difference[3:-1, :] > 0.0).all()  # All other pixels should have charge trailed into them

        def test__vertical_line__no_trails(self, arctic_serial):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[:, 2] += 100

            parameters = arctic_params.setup(include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                             s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
                                             s_well_fill_gamma=0.0)

            image_post_cti = pyarctic.add_serial_cti_to_image(image=image_pre_cti, params=parameters,
                                                              settings=arctic_serial)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[:, 0:2] == 0.0).all()  # Most pixels unchaged
            assert (image_difference[:, 3:-1] == 0.0).all()
            assert (image_difference[:, 2] < 0.0).all()  # charge line still loses charge

        def test__double_trap_density__more_captures_so_brighter_trails(self, arctic_serial):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            ### SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED ###

            parameters_0 = arctic_params.setup(include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                               s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
                                               s_well_fill_gamma=0.0)

            parameters_1 = arctic_params.setup(include_serial=True, s_trap_densities=(0.2,), s_trap_lifetimes=(1.0,),
                                               s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
                                               s_well_fill_gamma=0.0)

            ### NOW GENERATE THE IMAGE POST CTI OF EACH SET

            image_post_cti_0 = pyarctic.add_serial_cti_to_image(image=image_pre_cti, params=parameters_0,
                                                                settings=arctic_serial)
            image_post_cti_1 = pyarctic.add_serial_cti_to_image(image=image_pre_cti, params=parameters_1,
                                                                settings=arctic_serial)

            assert (image_post_cti_0[0:2, :] == 0.0).all()  # First four rows should all remain zero
            assert (image_post_cti_1[0:2, :] == 0.0).all()  # First four rows should all remain zero
            assert (image_post_cti_0[2, :] > image_post_cti_1[4, :]).all()  # charge line loses less charge in image 1
            assert (image_post_cti_0[3:-1, :] < image_post_cti_1[5:-1,
                                                :]).all()  # fewer pixels trailed behind in image 2

        def test__double_trap_lifetime__longer_release_so_fainter_trails(self, arctic_serial):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            ### SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED ###

            parameters_0 = arctic_params.setup(include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                               s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
                                               s_well_fill_gamma=0.0)

            parameters_1 = arctic_params.setup(include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(2.0,),
                                               s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
                                               s_well_fill_gamma=0.0)

            ### NOW GENERATE THE IMAGE POST CTI OF EACH SET

            image_post_cti_0 = pyarctic.add_serial_cti_to_image(image=image_pre_cti, params=parameters_0,
                                                                settings=arctic_serial)
            image_post_cti_1 = pyarctic.add_serial_cti_to_image(image=image_pre_cti, params=parameters_1,
                                                                settings=arctic_serial)

            assert (image_post_cti_0[0:2, :] == 0.0).all()  # First four rows should all remain zero
            assert (image_post_cti_1[0:2, :] == 0.0).all()  # First four rows should all remain zero
            assert (image_post_cti_0[2, :] == image_post_cti_1[2, :]).all()  # charge line loses equal amount of charge
            assert (image_post_cti_0[3:-1, :] > image_post_cti_1[3:-1,
                                                :]).all()  # each trail in pixel 2 is 'longer' so fainter

        # def test__increase_alpha__more_captures_brighter_trail(self, arctic_serial):
        #
        #     image_pre_cti = np.zeros((5, 5))
        #     image_pre_cti[2, :] += 100
        #
        #     ### SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED ###
        #
        #     parameters_0 = arctic_params.setup(include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
        #             s_well_notch_depth=0.01, s_well_fill_alpha=0.8, s_well_fill_beta=0.8, s_well_fill_gamma=0.1)
        #
        #     parameters_1 = arctic_params.setup(include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
        #                                       s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
        #                                       s_well_fill_gamma=0.1)
        #
        #     ### NOW GENERATE THE IMAGE POST CTI OF EACH SET
        #
        #     image_post_cti_0 = pyarctic.add_serial_cti_to_image(image=image_pre_cti, params=parameters_0,
        #                                                         settings=arctic_serial)
        #     image_post_cti_1 = pyarctic.add_serial_cti_to_image(image=image_pre_cti, params=parameters_1,
        #                                                         settings=arctic_serial)
        #
        #     assert (image_post_cti_0[0:2, :] == 0.0).all()  # First four rows should all remain zero
        #     assert (image_post_cti_1[0:2, :] == 0.0).all()  # First four rows should all remain zero
        #     assert (image_post_cti_0[2, :] > image_post_cti_1[2,:]).all()  # charge line loses less charge with higher beta
        #     assert (image_post_cti_0[3:-1, :] < image_post_cti_1[3:-1, :]).all()  # so less electrons trailed into image

        def test__increase_beta__fewer_captures_fainter_trail(self, arctic_serial):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            ### SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED ###

            parameters_0 = arctic_params.setup(include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                               s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
                                               s_well_fill_gamma=0.0)

            parameters_1 = arctic_params.setup(include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                               s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.9,
                                               s_well_fill_gamma=0.0)

            ### NOW GENERATE THE IMAGE POST CTI OF EACH SET

            image_post_cti_0 = pyarctic.add_serial_cti_to_image(image=image_pre_cti, params=parameters_0,
                                                                settings=arctic_serial)
            image_post_cti_1 = pyarctic.add_serial_cti_to_image(image=image_pre_cti, params=parameters_1,
                                                                settings=arctic_serial)

            assert (image_post_cti_0[0:2, :] == 0.0).all()  # First four rows should all remain zero
            assert (image_post_cti_1[0:2, :] == 0.0).all()  # First four rows should all remain zero
            assert (image_post_cti_0[2, :] < image_post_cti_1[2,
                                             :]).all()  # charge line loses less charge with higher beta
            assert (image_post_cti_0[3:-1, :] > image_post_cti_1[3:-1, :]).all()  # so less electrons trailed into image

        # def test__increase_gamma__more_captures_brighter_trail(self, arctic_serial):
        #
        #     image_pre_cti = np.zeros((5, 5))
        #     image_pre_cti[2, :] += 100
        #
        #     ### SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED ###
        #
        #     parameters_0 = arctic_params.setup(include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
        #             s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8, s_well_fill_gamma=0.0)
        #
        #     parameters_1 = arctic_params.setup(include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
        #                                       s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
        #                                       s_well_fill_gamma=0.2)
        #
        #     ### NOW GENERATE THE IMAGE POST CTI OF EACH SET
        #
        #     image_post_cti_0 = pyarctic.add_serial_cti_to_image(image=image_pre_cti, cti_params=parameters_0,
        #                                                         settings=arctic_serial)
        #     image_post_cti_1 = pyarctic.add_serial_cti_to_image(image=image_pre_cti, cti_params=parameters_1,
        #                                                         settings=arctic_serial)
        #
        #     assert (image_post_cti_0[0:2, :] == 0.0).all()  # First four rows should all remain zero
        #     assert (image_post_cti_1[0:2, :] == 0.0).all()  # First four rows should all remain zero
        #     assert (image_post_cti_0[2, :] > image_post_cti_1[2,:]).all()  # charge line loses less charge with higher beta
        #     assert (image_post_cti_0[3:-1, :] < image_post_cti_1[3:-1, :]).all()  # so less electrons trailed into image

        def test__two_traps_half_density_of_one__same_trails(self, arctic_serial):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            ### SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED ###

            parameters_0 = arctic_params.setup(include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                               s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
                                               s_well_fill_gamma=0.0)

            parameters_1 = arctic_params.setup(include_serial=True, s_trap_densities=(0.05, 0.05,),
                                               s_trap_lifetimes=(1.0, 1.0),
                                               s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
                                               s_well_fill_gamma=0.0)

            ### NOW GENERATE THE IMAGE POST CTI OF EACH SET

            image_post_cti_0 = pyarctic.add_serial_cti_to_image(image=image_pre_cti, params=parameters_0,
                                                                settings=arctic_serial)
            image_post_cti_1 = pyarctic.add_serial_cti_to_image(image=image_pre_cti, params=parameters_1,
                                                                settings=arctic_serial)

            assert (image_post_cti_0 == image_post_cti_1).all()

        def test__three_traps_third_density_of_one__same_trails(self, arctic_serial):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            ### SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED ###

            parameters_0 = arctic_params.setup(include_serial=True, s_trap_densities=(0.15,), s_trap_lifetimes=(1.0,),
                                               s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
                                               s_well_fill_gamma=0.0)

            parameters_1 = arctic_params.setup(include_serial=True, s_trap_densities=(0.05, 0.05, 0.05,),
                                               s_trap_lifetimes=(1.0, 1.0, 1.0),
                                               s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
                                               s_well_fill_gamma=0.0)

            ### NOW GENERATE THE IMAGE POST CTI OF EACH SET

            image_post_cti_0 = pyarctic.add_serial_cti_to_image(image=image_pre_cti, params=parameters_0,
                                                                settings=arctic_serial)
            image_post_cti_1 = pyarctic.add_serial_cti_to_image(image=image_pre_cti, params=parameters_1,
                                                                settings=arctic_serial)

            image_difference = abs(image_post_cti_0 - image_post_cti_1)

            assert (image_difference < 1e-9).all()

        def test__delta_functions__add_cti_only_behind_them(self, arctic_serial):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[1, 1] += 100  # Delta 1
            image_pre_cti[3, 3] += 100  # Delta 2
            image_pre_cti[2, 4] += 100  # Delta 3

            parameters = arctic_params.setup(include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                             s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
                                             s_well_fill_gamma=0.0)

            image_post_cti = pyarctic.add_serial_cti_to_image(image=image_pre_cti, params=parameters,
                                                              settings=arctic_serial)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[:, 0] == 0.0).all()  # No Delta, no charge

            assert (image_difference[0, 1] == 0.0)  # No charge in front of Delta 1
            assert (image_difference[1, 1] < 0.0)  # Delta 1 loses charge
            assert (image_difference[2:5, 1] > 0.0).all()  # Delta 1 trails

            assert (image_difference[:, 2] == 0.0).all()  # No Delta, no charge

            assert (image_difference[0:3, 3] == 0.0).all()  # No charge in front of Delta 2
            assert image_difference[3, 3] < 0.0  # Delta 2 loses charge
            assert image_difference[4, 3] > 0.0  # Delta 2 trail

            assert (image_difference[0:2, 4] == 0.0).all()  # No charge in front of Delta 3
            assert image_difference[2, 4] < 0.0  # Delta 3 loses charge
            assert (image_difference[3:5, 4] > 0.0).all()  # Delta 3 trail

    class TestRectangularImageDimensions:

        def test__horizontal_line__rectangular_image_odd_x_odd(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 7))
            image_pre_cti[2, :] += 100

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[0:2, :] == 0.0).all()
            assert (image_difference[2, :] < 0.0).all()
            assert (image_difference[3:-1, :] > 0.0).all()

        def test__horizontal_line__rectangular_image_even_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((4, 6))
            image_pre_cti[2, :] += 100

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[0:2, :] == 0.0).all()
            assert (image_difference[2, :] < 0.0).all()
            assert (image_difference[3:-1, :] > 0.0).all()

        def test__horizontal_line__rectangular_image_even_x_odd(self, arctic_parallel):
            image_pre_cti = np.zeros((4, 7))
            image_pre_cti[2, :] += 100

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[0:2, :] == 0.0).all()
            assert (image_difference[2, :] < 0.0).all()
            assert (image_difference[3:-1, :] > 0.0).all()

        def test__horizontal_line__rectangular_image_odd_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 6))
            image_pre_cti[2, :] += 100

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[0:2, :] == 0.0).all()
            assert (image_difference[2, :] < 0.0).all()
            assert (image_difference[3:-1, :] > 0.0).all()

        def test__vertical_line__rectangular_image_odd_x_odd(self, arctic_parallel):
            image_pre_cti = np.zeros((3, 5))
            image_pre_cti[:, 2] += 100

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[:, 0:2] == 0.0).all()
            assert (image_difference[:, 3:-1] == 0.0).all()
            assert (image_difference[:, 2] < 0.0).all()

        def test__vertical_line__rectangular_image_even_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((4, 6))
            image_pre_cti[:, 2] += 100

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[:, 0:2] == 0.0).all()
            assert (image_difference[:, 3:-1] == 0.0).all()
            assert (image_difference[:, 2] < 0.0).all()

        def test__vertical_line__rectangular_image_even_x_odd(self, arctic_parallel):
            image_pre_cti = np.zeros((4, 7))
            image_pre_cti[:, 2] += 100

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[:, 0:2] == 0.0).all()
            assert (image_difference[:, 3:-1] == 0.0).all()
            assert (image_difference[:, 2] < 0.0).all()

        def test__vertical_line__rectangular_image_odd_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 6))
            image_pre_cti[:, 2] += 100

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[:, 0:2] == 0.0).all()
            assert (image_difference[:, 3:-1] == 0.0).all()
            assert (image_difference[:, 2] < 0.0).all()

        def test__delta_functions__add_cti_only_behind_them__odd_x_odd(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 7))
            image_pre_cti[1, 1] += 100  # Delta 1
            image_pre_cti[3, 3] += 100  # Delta 2
            image_pre_cti[2, 4] += 100  # Delta 3

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[:, 0] == 0.0).all()  # No Delta, no charge

            assert (image_difference[0, 1] == 0.0)  # No charge in front of Delta 1
            assert (image_difference[1, 1] < 0.0)  # Delta 1 loses charge
            assert (image_difference[2:5, 1] > 0.0).all()  # Delta 1 trails

            assert (image_difference[:, 2] == 0.0).all()  # No Delta, no charge

            assert (image_difference[0:3, 3] == 0.0).all()  # No charge in front of Delta 2
            assert image_difference[3, 3] < 0.0  # Delta 2 loses charge
            assert image_difference[4, 3] > 0.0  # Delta 2 trail

            assert (image_difference[0:2, 4] == 0.0).all()  # No charge in front of Delta 3
            assert image_difference[2, 4] < 0.0  # Delta 3 loses charge
            assert (image_difference[3:5, 4] > 0.0).all()  # Delta 3 trail

            assert (image_difference[:, 5] == 0.0).all()  # No Delta, no charge
            assert (image_difference[:, 6] == 0.0).all()  # No Delta, no charge

        def test__delta_functions__add_cti_only_behind_them__even_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((6, 8))
            image_pre_cti[1, 1] += 100  # Delta 1
            image_pre_cti[3, 3] += 100  # Delta 2
            image_pre_cti[2, 4] += 100  # Delta 3

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[:, 0] == 0.0).all()  # No Delta, no charge

            assert (image_difference[0, 1] == 0.0)  # No charge in front of Delta 1
            assert (image_difference[1, 1] < 0.0)  # Delta 1 loses charge
            assert (image_difference[2:5, 1] > 0.0).all()  # Delta 1 trails

            assert (image_difference[:, 2] == 0.0).all()  # No Delta, no charge

            assert (image_difference[0:3, 3] == 0.0).all()  # No charge in front of Delta 2
            assert image_difference[3, 3] < 0.0  # Delta 2 loses charge
            assert image_difference[4, 3] > 0.0  # Delta 2 trail

            assert (image_difference[0:2, 4] == 0.0).all()  # No charge in front of Delta 3
            assert image_difference[2, 4] < 0.0  # Delta 3 loses charge
            assert (image_difference[3:5, 4] > 0.0).all()  # Delta 3 trail

            assert (image_difference[:, 5] == 0.0).all()  # No Delta, no charge
            assert (image_difference[:, 6] == 0.0).all()  # No Delta, no charge
            assert (image_difference[:, 7] == 0.0).all()  # No Delta, no charge

        def test__delta_functions__add_cti_only_behind_them__even_x_odd(self, arctic_parallel):
            image_pre_cti = np.zeros((6, 7))
            image_pre_cti[1, 1] += 100  # Delta 1
            image_pre_cti[3, 3] += 100  # Delta 2
            image_pre_cti[2, 4] += 100  # Delta 3

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[:, 0] == 0.0).all()  # No Delta, no charge

            assert (image_difference[0, 1] == 0.0)  # No charge in front of Delta 1
            assert (image_difference[1, 1] < 0.0)  # Delta 1 loses charge
            assert (image_difference[2:5, 1] > 0.0).all()  # Delta 1 trails

            assert (image_difference[:, 2] == 0.0).all()  # No Delta, no charge

            assert (image_difference[0:3, 3] == 0.0).all()  # No charge in front of Delta 2
            assert image_difference[3, 3] < 0.0  # Delta 2 loses charge
            assert image_difference[4, 3] > 0.0  # Delta 2 trail

            assert (image_difference[0:2, 4] == 0.0).all()  # No charge in front of Delta 3
            assert image_difference[2, 4] < 0.0  # Delta 3 loses charge
            assert (image_difference[3:5, 4] > 0.0).all()  # Delta 3 trail

            assert (image_difference[:, 5] == 0.0).all()  # No Delta, no charge
            assert (image_difference[:, 6] == 0.0).all()  # No Delta, no charge

        def test__delta_functions__add_cti_only_behind_them__odd_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 6))

            image_pre_cti[1, 1] += 100  # Delta 1
            image_pre_cti[3, 3] += 100  # Delta 2
            image_pre_cti[2, 4] += 100  # Delta 3

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[:, 0] == 0.0).all()  # No Delta, no charge

            assert (image_difference[0, 1] == 0.0)  # No charge in front of Delta 1
            assert (image_difference[1, 1] < 0.0)  # Delta 1 loses charge
            assert (image_difference[2:5, 1] > 0.0).all()  # Delta 1 trails

            assert (image_difference[:, 2] == 0.0).all()  # No Delta, no charge

            assert (image_difference[0:3, 3] == 0.0).all()  # No charge in front of Delta 2
            assert image_difference[3, 3] < 0.0  # Delta 2 loses charge
            assert image_difference[4, 3] > 0.0  # Delta 2 trail

            assert (image_difference[0:2, 4] == 0.0).all()  # No charge in front of Delta 3
            assert image_difference[2, 4] < 0.0  # Delta 3 loses charge
            assert (image_difference[3:5, 4] > 0.0).all()  # Delta 3 trail

            assert (image_difference[:, 5] == 0.0).all()  # No Delta, no charge

    class TestChargeInjectionMode:

        def test__charge_injection_parameters_on__makes_trails_longer(self):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            settings = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=1, p_express=5,
                                             p_n_levels=2000,
                                             p_charge_injection_mode=False, p_readout_offset=0)

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=settings)

            settings_charge_inj = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=1,
                                                        p_express=5, p_n_levels=2000,
                                                        p_charge_injection_mode=True, p_readout_offset=0)

            image_post_cti_charge_inj = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                           settings=settings_charge_inj)

            image_difference = image_post_cti_charge_inj - image_post_cti

            assert (image_difference[0:2, :] == 0.0).all()  # First four rows should all remain zero
            assert (image_difference[2,
                    :] < 0.0).all()  # With charge injection mode on, more transfers, so more charge lost
            assert (image_difference[3:-1, :] > 0.0).all()  # And therefore, trails are brighter

    class TestDensityVary:

        def test__horizontal_line__different_density_in_each_column(self):
            settings = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=1, p_express=5,
                                             p_n_levels=2000,
                                             p_charge_injection_mode=False, p_readout_offset=0)

            parallel_vary = arctic_params.ParallelDensityVary(
                trap_densities=[(10.0,), (20.0,), (30.0,), (40.0,), (50.0,)],
                trap_lifetimes=(1.0,), well_fill_beta=0.8, well_notch_depth=0.01)

            parameters = arctic_params.ArcticParams(parallel_species=parallel_vary)

            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=settings)

            assert (image_post_cti[2:5, 0] != image_post_cti[2:5, 1]).all()
            assert (image_post_cti[2:5, 1] != image_post_cti[2:5, 2]).all()
            assert (image_post_cti[2:5, 2] != image_post_cti[2:5, 3]).all()
            assert (image_post_cti[2:5, 3] != image_post_cti[2:5, 4]).all()


class TestArcticCorrectCTI:
    class TestParallel:

        def test__horizontal_line__corrected_image_more_like_original(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = pyarctic.correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                         settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__vertical_line__corrected_image_more_like_original(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[:, 2] += 100

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = pyarctic.correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                         settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__delta_functions__corrected_image_more_like_original(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[1, 1] += 100  # Delta 1
            image_pre_cti[3, 3] += 100  # Delta 2
            image_pre_cti[2, 4] += 100  # Delta 3

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = pyarctic.correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                         settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__decrease_niter__worse_correction(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            arctic_niter_5 = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=5, p_express=5,
                                                   p_n_levels=2000,
                                                   p_charge_injection_mode=False, p_readout_offset=0)

            image_correct_cti = pyarctic.correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                         settings=arctic_niter_5)

            image_difference_niter_5 = image_correct_cti - image_pre_cti

            arctic_niter_3 = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=3, p_express=5,
                                                   p_n_levels=2000,
                                                   p_charge_injection_mode=False, p_readout_offset=0)

            image_correct_cti = pyarctic.correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                         settings=arctic_niter_3)

            image_difference_niter_3 = image_correct_cti - image_pre_cti

            assert (abs(image_difference_niter_5) <= abs(
                image_difference_niter_3)).all()  # First four rows should all remain zero

    class TestSerial:

        def test__horizontal_line__corrected_image_more_like_original(self, arctic_serial):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] = 100

            parameters = arctic_params.setup(include_serial=True, s_trap_densities=(0.2,), s_trap_lifetimes=(2.0,),
                                             s_well_notch_depth=0.02, s_well_fill_beta=0.4)

            image_post_cti = pyarctic.add_serial_cti_to_image(image=image_pre_cti, params=parameters,
                                                              settings=arctic_serial)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = pyarctic.correct_serial_cti_from_image(image=image_post_cti, params=parameters,
                                                                       settings=arctic_serial)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()

        def test__vertical_line__corrected_image_more_like_original(self, arctic_serial):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[:, 2] += 100

            parameters = arctic_params.setup(include_serial=True, s_trap_densities=(0.2,), s_trap_lifetimes=(2.0,),
                                             s_well_notch_depth=0.02, s_well_fill_beta=0.4)

            image_post_cti = pyarctic.add_serial_cti_to_image(image=image_pre_cti, params=parameters,
                                                              settings=arctic_serial)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = pyarctic.correct_serial_cti_from_image(image=image_post_cti, params=parameters,
                                                                       settings=arctic_serial)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()

        def test__delta_functions__corrected_image_more_like_original(self, arctic_serial):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[1, 1] += 100  # Delta 1
            image_pre_cti[3, 3] += 100  # Delta 2
            image_pre_cti[2, 4] += 100  # Delta 3

            parameters = arctic_params.setup(include_serial=True, s_trap_densities=(0.2,), s_trap_lifetimes=(2.0,),
                                             s_well_notch_depth=0.02, s_well_fill_beta=0.4)

            image_post_cti = pyarctic.add_serial_cti_to_image(image=image_pre_cti, params=parameters,
                                                              settings=arctic_serial)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = pyarctic.correct_serial_cti_from_image(image=image_post_cti, params=parameters,
                                                                       settings=arctic_serial)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__decrease_niter__worse_correction(self, arctic_serial):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[:, 2] += 100

            parameters = arctic_params.setup(include_serial=True, s_trap_densities=(0.2,), s_trap_lifetimes=(2.0,),
                                             s_well_notch_depth=0.02, s_well_fill_beta=0.4)

            image_post_cti = pyarctic.add_serial_cti_to_image(image=image_pre_cti, params=parameters,
                                                              settings=arctic_serial)

            arctic_niter_5 = arctic_settings.setup(include_serial=True, s_well_depth=84700, s_niter=5, s_express=5,
                                                   s_n_levels=2000,
                                                   s_charge_injection_mode=False, s_readout_offset=0)

            image_correct_cti = pyarctic.correct_serial_cti_from_image(image=image_post_cti, params=parameters,
                                                                       settings=arctic_niter_5)

            image_difference_niter_5 = image_correct_cti - image_pre_cti

            arctic_niter_3 = arctic_settings.setup(include_serial=True, s_well_depth=84700, s_niter=3, s_express=5,
                                                   s_n_levels=2000,
                                                   s_charge_injection_mode=False, s_readout_offset=0)

            image_correct_cti = pyarctic.correct_serial_cti_from_image(image=image_post_cti, params=parameters,
                                                                       settings=arctic_niter_3)

            image_difference_niter_3 = image_correct_cti - image_pre_cti

            assert (abs(image_difference_niter_5) <= abs(
                image_difference_niter_3)).all()  # First four rows should all remain zero

    class TestRectangularImageDimensions:

        def test__horizontal_line__odd_x_odd(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 3))
            image_pre_cti[2, :] += 100

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = pyarctic.correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                         settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__horizontal_line__even_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((6, 4))
            image_pre_cti[2, :] += 100

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = pyarctic.correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                         settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__horizontal_line__even_x_odd(self, arctic_parallel):
            image_pre_cti = np.zeros((6, 3))
            image_pre_cti[2, :] += 100

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = pyarctic.correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                         settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__horizontal_line__odd_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 4))
            image_pre_cti[2, :] += 100

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = pyarctic.correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                         settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__veritcal_line__odd_x_odd(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 3))
            image_pre_cti[:, 2] += 100

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = pyarctic.correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                         settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__veritcal_line__even_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((6, 4))
            image_pre_cti[:, 2] += 100

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = pyarctic.correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                         settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__veritcal_line__even_x_odd(self, arctic_parallel):
            image_pre_cti = np.zeros((6, 3))
            image_pre_cti[:, 2] += 100

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = pyarctic.correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                         settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__veritcal_line__odd_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 4))
            image_pre_cti[:, 2] += 100

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = pyarctic.correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                         settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__delta_functions__odd_x_odd(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 7))
            image_pre_cti[1, 1] += 100  # Delta 1
            image_pre_cti[3, 3] += 100  # Delta 2
            image_pre_cti[2, 4] += 100  # Delta 3

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = pyarctic.correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                         settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__delta_functions__even_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((6, 8))
            image_pre_cti[1, 1] += 100  # Delta 1
            image_pre_cti[3, 3] += 100  # Delta 2
            image_pre_cti[2, 4] += 100  # Delta 3

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = pyarctic.correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                         settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__delta_functions__even_x_odd(self, arctic_parallel):
            image_pre_cti = np.zeros((6, 7))
            image_pre_cti[1, 1] += 100  # Delta 1
            image_pre_cti[3, 3] += 100  # Delta 2
            image_pre_cti[2, 4] += 100  # Delta 3

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = pyarctic.correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                         settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__delta_functions__odd_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 8))
            image_pre_cti[1, 1] += 100  # Delta 1
            image_pre_cti[3, 3] += 100  # Delta 2
            image_pre_cti[2, 4] += 100  # Delta 3

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = pyarctic.correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                         settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__decrease_niter__worse_correction__odd_x_odd(self, arctic_parallel):
            image_pre_cti = np.zeros((3, 5))
            image_pre_cti[2, :] += 100

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            arctic_niter_5 = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=5, p_express=5,
                                                   p_n_levels=2000,
                                                   p_charge_injection_mode=False, p_readout_offset=0)

            image_correct_cti = pyarctic.correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                         settings=arctic_niter_5)

            image_difference_niter_5 = image_correct_cti - image_pre_cti

            arctic_niter_3 = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=3, p_express=5,
                                                   p_n_levels=2000,
                                                   p_charge_injection_mode=False, p_readout_offset=0)

            image_correct_cti = pyarctic.correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                         settings=arctic_niter_3)

            image_difference_niter_3 = image_correct_cti - image_pre_cti

            assert (abs(image_difference_niter_5) <= abs(
                image_difference_niter_3)).all()  # First four rows should all remain zero

        def test__dencrease_niter__worse_correction__even_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((4, 4))
            image_pre_cti[2, :] += 100

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            arctic_niter_5 = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=5, p_express=5,
                                                   p_n_levels=2000,
                                                   p_charge_injection_mode=False, p_readout_offset=0)

            image_correct_cti = pyarctic.correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                         settings=arctic_niter_5)

            image_difference_niter_5 = image_correct_cti - image_pre_cti

            arctic_niter_3 = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=3, p_express=5,
                                                   p_n_levels=2000,
                                                   p_charge_injection_mode=False, p_readout_offset=0)

            image_correct_cti = pyarctic.correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                         settings=arctic_niter_3)

            image_difference_niter_3 = image_correct_cti - image_pre_cti

            assert (abs(image_difference_niter_5) <= abs(
                image_difference_niter_3)).all()  # First four rows should all remain zero

        def test__dencrease_niter__worse_correction__even_x_odd(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 4))
            image_pre_cti[2, :] += 100

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            arctic_niter_5 = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=5, p_express=5,
                                                   p_n_levels=2000,
                                                   p_charge_injection_mode=False, p_readout_offset=0)

            image_correct_cti = pyarctic.correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                         settings=arctic_niter_5)

            image_difference_niter_5 = image_correct_cti - image_pre_cti

            arctic_niter_3 = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=3, p_express=5,
                                                   p_n_levels=2000,
                                                   p_charge_injection_mode=False, p_readout_offset=0)

            image_correct_cti = pyarctic.correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                         settings=arctic_niter_3)

            image_difference_niter_3 = image_correct_cti - image_pre_cti

            assert (abs(image_difference_niter_5) <= abs(
                image_difference_niter_3)).all()  # First four rows should all remain zero

        def test__dencrease_niter__worse_correction__odd_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((3, 6))
            image_pre_cti[2, :] += 100

            parameters = arctic_params.setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                             p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = pyarctic.add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                                settings=arctic_parallel)

            arctic_niter_5 = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=5, p_express=5,
                                                   p_n_levels=2000,
                                                   p_charge_injection_mode=False, p_readout_offset=0)

            image_correct_cti = pyarctic.correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                         settings=arctic_niter_5)

            image_difference_niter_5 = image_correct_cti - image_pre_cti

            arctic_niter_3 = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=3, p_express=5,
                                                   p_n_levels=2000,
                                                   p_charge_injection_mode=False, p_readout_offset=0)

            image_correct_cti = pyarctic.correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                         settings=arctic_niter_3)

            image_difference_niter_3 = image_correct_cti - image_pre_cti

            assert (abs(image_difference_niter_5) <= abs(
                image_difference_niter_3)).all()  # First four rows should all remain zero
