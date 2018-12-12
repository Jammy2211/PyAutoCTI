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

import numpy as np
import pytest

from autocti.model import arctic_params
from autocti.model import arctic_settings
from autocti.model import pyarctic


def setup(include_parallel=False, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
          p_well_notch_depth=0.01, p_well_fill_alpha=1.0, p_well_fill_beta=0.8,
          p_well_fill_gamma=0.0, include_serial=False, s_trap_densities=(0.05, 0.05,),
          s_trap_lifetimes=(1.0, 1.0),
          s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
          s_well_fill_gamma=0.0):
    serial_species = []
    parallel_species = []
    serial_ccd = None
    parallel_ccd = None

    if include_serial:
        serial_ccd = arctic_params.CCD(well_notch_depth=s_well_notch_depth, well_fill_alpha=s_well_fill_alpha,
                                       well_fill_beta=s_well_fill_beta, well_fill_gamma=s_well_fill_gamma)
        serial_species = [arctic_params.Species(trap_density=s_trap_densities[i], trap_lifetime=s_trap_lifetimes[i]) for
                          i in range(len(s_trap_densities))]
    if include_parallel:
        parallel_ccd = arctic_params.CCD(well_notch_depth=p_well_notch_depth, well_fill_alpha=p_well_fill_alpha,
                                         well_fill_beta=p_well_fill_beta, well_fill_gamma=p_well_fill_gamma)
        parallel_species = [arctic_params.Species(trap_density=p_trap_densities[i], trap_lifetime=p_trap_lifetimes[i])
                            for
                            i in range(len(p_trap_densities))]

    return arctic_params.ArcticParams(parallel_ccd=parallel_ccd, serial_ccd=serial_ccd, serial_species=serial_species,
                                      parallel_species=parallel_species)


@pytest.fixture(scope='class', name='arctic_parallel')
def make_arctic_parallel():
    parallel_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                                 charge_injection_mode=True, readout_offset=0)

    return arctic_settings.ArcticSettings(neomode='NEO', parallel=parallel_settings)


@pytest.fixture(scope='class', name='arctic_serial')
def make_arctic_serial():
    serial_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                               readout_offset=0)
    return arctic_settings.ArcticSettings(neomode='NEO', serial=serial_settings)


@pytest.fixture(scope='class', name='arctic_both')
def make_arctic_both():
    parallel_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                                 readout_offset=0)

    serial_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                               readout_offset=0)
    return arctic_settings.ArcticSettings(neomode='NEO', parallel=parallel_settings,
                                          serial=serial_settings)


def add_parallel_cti_to_image(image, params, settings):
    return pyarctic.call_arctic(image, params.parallel_species, params.parallel_ccd, settings.parallel)


def add_serial_cti_to_image(image, params, settings):
    return pyarctic.call_arctic(image, params.serial_species, params.serial_ccd, settings.serial)


def correct_parallel_cti_from_image(image, params, settings):
    return pyarctic.call_arctic(image, params.parallel_species, params.parallel_ccd, settings.parallel,
                                correct_cti=True)


def correct_serial_cti_from_image(image, params, settings):
    return pyarctic.call_arctic(image, params.serial_species, params.serial_ccd, settings.serial, correct_cti=True)


class TestArcticAddCTI:
    class TestParallelCTI:

        def test__horizontal_line__line_loses_charge_trails_appear(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_alpha=1.0, p_well_fill_beta=0.8,
                               p_well_fill_gamma=0.0)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[0:2, :] == 0.0).all()  # First four rows should all remain zero
            assert (image_difference[2,
                    :] < 0.0).all()  # All pixels in the charge line should lose charge due to capture
            assert (image_difference[3:-1, :] > 0.0).all()  # All other pixels should have charge trailed into them

        def test__vertical_line__no_trails(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[:, 2] += 100

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_alpha=1.0, p_well_fill_beta=0.8,
                               p_well_fill_gamma=0.0)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[:, 0:2] == 0.0).all()  # Most pixels unchaged
            assert (image_difference[:, 3:-1] == 0.0).all()
            assert (image_difference[:, 2] < 0.0).all()  # charge line still loses charge

        def test__double_trap_density__more_captures_so_brighter_trails(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            ### SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED ###

            parameters_0 = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                 p_well_notch_depth=0.01, p_well_fill_alpha=1.0, p_well_fill_beta=0.8,
                                 p_well_fill_gamma=0.0)

            parameters_1 = setup(include_parallel=True, p_trap_densities=(0.2,), p_trap_lifetimes=(1.0,),
                                 p_well_notch_depth=0.01, p_well_fill_alpha=1.0, p_well_fill_beta=0.8,
                                 p_well_fill_gamma=0.0)

            ### NOW GENERATE THE IMAGE POST CTI OF EACH SET

            image_post_cti_0 = add_parallel_cti_to_image(image=image_pre_cti, params=parameters_0,
                                                         settings=arctic_parallel)
            image_post_cti_1 = add_parallel_cti_to_image(image=image_pre_cti, params=parameters_1,
                                                         settings=arctic_parallel)

            assert (image_post_cti_0[0:2, :] == 0.0).all()  # First four rows should all remain zero
            assert (image_post_cti_1[0:2, :] == 0.0).all()  # First four rows should all remain zero
            # noinspection PyUnresolvedReferences
            assert (image_post_cti_0[2, :] > image_post_cti_1[4, :]).all()  # charge line loses less charge in image 1
            # noinspection PyUnresolvedReferences
            assert (image_post_cti_0[3:-1, :] < image_post_cti_1[5:-1,
                                                :]).all()  # fewer pixels trailed behind in image 2

        def test__double_trap_lifetime__longer_release_so_fainter_trails(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            ### SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED ###

            parameters_0 = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                 p_well_notch_depth=0.01, p_well_fill_alpha=1.0, p_well_fill_beta=0.8,
                                 p_well_fill_gamma=0.0)

            parameters_1 = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(2.0,),
                                 p_well_notch_depth=0.01, p_well_fill_alpha=1.0, p_well_fill_beta=0.8,
                                 p_well_fill_gamma=0.0)
            ### NOW GENERATE THE IMAGE POST CTI OF EACH SET

            image_post_cti_0 = add_parallel_cti_to_image(image=image_pre_cti, params=parameters_0,
                                                         settings=arctic_parallel)
            image_post_cti_1 = add_parallel_cti_to_image(image=image_pre_cti, params=parameters_1,
                                                         settings=arctic_parallel)

            assert (image_post_cti_0[0:2, :] == 0.0).all()  # First four rows should all remain zero
            assert (image_post_cti_1[0:2, :] == 0.0).all()  # First four rows should all remain zero
            # noinspection PyUnresolvedReferences
            assert (image_post_cti_0[2, :] == image_post_cti_1[2, :]).all()  # charge line loses equal amount of charge
            # noinspection PyUnresolvedReferences
            assert (image_post_cti_0[3:-1, :] > image_post_cti_1[3:-1,
                                                :]).all()  # each trail in pixel 2 is 'longer' so fainter

        def test__increase_beta__fewer_captures_fainter_trail(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            ### SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED ###

            parameters_0 = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                 p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            parameters_1 = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                 p_well_notch_depth=0.01, p_well_fill_beta=0.9)

            ### NOW GENERATE THE IMAGE POST CTI OF EACH SET

            image_post_cti_0 = add_parallel_cti_to_image(image=image_pre_cti, params=parameters_0,
                                                         settings=arctic_parallel)
            image_post_cti_1 = add_parallel_cti_to_image(image=image_pre_cti, params=parameters_1,
                                                         settings=arctic_parallel)

            assert (image_post_cti_0[0:2, :] == 0.0).all()  # First four rows should all remain zero
            assert (image_post_cti_1[0:2, :] == 0.0).all()  # First four rows should all remain zero
            # noinspection PyUnresolvedReferences
            assert (image_post_cti_0[2, :] < image_post_cti_1[2,
                                             :]).all()  # charge line loses less charge with higher beta
            # noinspection PyUnresolvedReferences
            assert (image_post_cti_0[3:-1, :] > image_post_cti_1[3:-1, :]).all()  # so less electrons trailed into image

        def test__two_traps_half_density_of_one__same_trails(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            ### SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED ###

            parameters_0 = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                 p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            parameters_1 = setup(include_parallel=True, p_trap_densities=(0.05, 0.05),
                                 p_trap_lifetimes=(1.0, 1.0),
                                 p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            ### NOW GENERATE THE IMAGE POST CTI OF EACH SET

            image_post_cti_0 = add_parallel_cti_to_image(image=image_pre_cti, params=parameters_0,
                                                         settings=arctic_parallel)
            image_post_cti_1 = add_parallel_cti_to_image(image=image_pre_cti, params=parameters_1,
                                                         settings=arctic_parallel)
            # noinspection PyUnresolvedReferences
            assert (image_post_cti_0 == image_post_cti_1).all()

        def test__three_traps_third_density_of_one__same_trails(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            ### SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED ###

            parameters_0 = setup(include_parallel=True, p_trap_densities=(0.15,), p_trap_lifetimes=(1.0,),
                                 p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            parameters_1 = setup(include_parallel=True, p_trap_densities=(0.05, 0.05, 0.05,),
                                 p_trap_lifetimes=(1.0, 1.0, 1.0),
                                 p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            ### NOW GENERATE THE IMAGE POST CTI OF EACH SET

            image_post_cti_0 = add_parallel_cti_to_image(image=image_pre_cti, params=parameters_0,
                                                         settings=arctic_parallel)
            image_post_cti_1 = add_parallel_cti_to_image(image=image_pre_cti, params=parameters_1,
                                                         settings=arctic_parallel)
            image_difference = abs(image_post_cti_0 - image_post_cti_1)

            assert (image_difference < 1e-9).all()

        def test__delta_functions__add_cti_only_behind_them(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[1, 1] += 100  # Delta 1
            image_pre_cti[3, 3] += 100  # Delta 2
            image_pre_cti[2, 4] += 100  # Delta 3

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
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

            parameters = setup(include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                               s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
                               s_well_fill_gamma=0.0)

            image_post_cti = add_serial_cti_to_image(image=image_pre_cti, params=parameters,
                                                     settings=arctic_serial)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[0:2, :] == 0.0).all()  # First four rows should all remain zero
            assert (image_difference[2,
                    :] < 0.0).all()  # All pixels in the charge line should lose charge due to capture
            assert (image_difference[3:-1, :] > 0.0).all()  # All other pixels should have charge trailed into them

        def test__vertical_line__no_trails(self, arctic_serial):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[:, 2] += 100

            parameters = setup(include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                               s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
                               s_well_fill_gamma=0.0)

            image_post_cti = add_serial_cti_to_image(image=image_pre_cti, params=parameters,
                                                     settings=arctic_serial)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[:, 0:2] == 0.0).all()  # Most pixels unchaged
            assert (image_difference[:, 3:-1] == 0.0).all()
            assert (image_difference[:, 2] < 0.0).all()  # charge line still loses charge

        def test__double_trap_density__more_captures_so_brighter_trails(self, arctic_serial):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            ### SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED ###

            parameters_0 = setup(include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                 s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
                                 s_well_fill_gamma=0.0)

            parameters_1 = setup(include_serial=True, s_trap_densities=(0.2,), s_trap_lifetimes=(1.0,),
                                 s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
                                 s_well_fill_gamma=0.0)

            ### NOW GENERATE THE IMAGE POST CTI OF EACH SET

            image_post_cti_0 = add_serial_cti_to_image(image=image_pre_cti, params=parameters_0,
                                                       settings=arctic_serial)
            image_post_cti_1 = add_serial_cti_to_image(image=image_pre_cti, params=parameters_1,
                                                       settings=arctic_serial)

            assert (image_post_cti_0[0:2, :] == 0.0).all()  # First four rows should all remain zero
            assert (image_post_cti_1[0:2, :] == 0.0).all()  # First four rows should all remain zero
            # noinspection PyUnresolvedReferences
            assert (image_post_cti_0[2, :] > image_post_cti_1[4, :]).all()  # charge line loses less charge in image 1
            # noinspection PyUnresolvedReferences
            assert (image_post_cti_0[3:-1, :] < image_post_cti_1[5:-1,
                                                :]).all()  # fewer pixels trailed behind in image 2

        def test__double_trap_lifetime__longer_release_so_fainter_trails(self, arctic_serial):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            ### SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED ###

            parameters_0 = setup(include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                 s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
                                 s_well_fill_gamma=0.0)

            parameters_1 = setup(include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(2.0,),
                                 s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
                                 s_well_fill_gamma=0.0)

            ### NOW GENERATE THE IMAGE POST CTI OF EACH SET

            image_post_cti_0 = add_serial_cti_to_image(image=image_pre_cti, params=parameters_0,
                                                       settings=arctic_serial)
            image_post_cti_1 = add_serial_cti_to_image(image=image_pre_cti, params=parameters_1,
                                                       settings=arctic_serial)

            assert (image_post_cti_0[0:2, :] == 0.0).all()  # First four rows should all remain zero
            assert (image_post_cti_1[0:2, :] == 0.0).all()  # First four rows should all remain zero
            # noinspection PyUnresolvedReferences
            assert (image_post_cti_0[2, :] == image_post_cti_1[2, :]).all()  # charge line loses equal amount of charge
            # noinspection PyUnresolvedReferences
            assert (image_post_cti_0[3:-1, :] > image_post_cti_1[3:-1,
                                                :]).all()  # each trail in pixel 2 is 'longer' so fainter

        def test__increase_beta__fewer_captures_fainter_trail(self, arctic_serial):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            ### SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED ###

            parameters_0 = setup(include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                 s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
                                 s_well_fill_gamma=0.0)

            parameters_1 = setup(include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                 s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.9,
                                 s_well_fill_gamma=0.0)

            ### NOW GENERATE THE IMAGE POST CTI OF EACH SET

            image_post_cti_0 = add_serial_cti_to_image(image=image_pre_cti, params=parameters_0,
                                                       settings=arctic_serial)
            image_post_cti_1 = add_serial_cti_to_image(image=image_pre_cti, params=parameters_1,
                                                       settings=arctic_serial)

            assert (image_post_cti_0[0:2, :] == 0.0).all()  # First four rows should all remain zero
            assert (image_post_cti_1[0:2, :] == 0.0).all()  # First four rows should all remain zero
            # noinspection PyUnresolvedReferences
            assert (image_post_cti_0[2, :] < image_post_cti_1[2,
                                             :]).all()  # charge line loses less charge with higher beta
            # noinspection PyUnresolvedReferences
            assert (image_post_cti_0[3:-1, :] > image_post_cti_1[3:-1, :]).all()  # so less electrons trailed into image

        def test__two_traps_half_density_of_one__same_trails(self, arctic_serial):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            ### SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED ###

            parameters_0 = setup(include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                                 s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
                                 s_well_fill_gamma=0.0)

            parameters_1 = setup(include_serial=True, s_trap_densities=(0.05, 0.05,),
                                 s_trap_lifetimes=(1.0, 1.0),
                                 s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
                                 s_well_fill_gamma=0.0)

            ### NOW GENERATE THE IMAGE POST CTI OF EACH SET

            image_post_cti_0 = add_serial_cti_to_image(image=image_pre_cti, params=parameters_0,
                                                       settings=arctic_serial)
            image_post_cti_1 = add_serial_cti_to_image(image=image_pre_cti, params=parameters_1,
                                                       settings=arctic_serial)
            # noinspection PyUnresolvedReferences
            assert (image_post_cti_0 == image_post_cti_1).all()

        def test__three_traps_third_density_of_one__same_trails(self, arctic_serial):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            ### SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED ###

            parameters_0 = setup(include_serial=True, s_trap_densities=(0.15,), s_trap_lifetimes=(1.0,),
                                 s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
                                 s_well_fill_gamma=0.0)

            parameters_1 = setup(include_serial=True, s_trap_densities=(0.05, 0.05, 0.05,),
                                 s_trap_lifetimes=(1.0, 1.0, 1.0),
                                 s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
                                 s_well_fill_gamma=0.0)

            ### NOW GENERATE THE IMAGE POST CTI OF EACH SET

            image_post_cti_0 = add_serial_cti_to_image(image=image_pre_cti, params=parameters_0,
                                                       settings=arctic_serial)
            image_post_cti_1 = add_serial_cti_to_image(image=image_pre_cti, params=parameters_1,
                                                       settings=arctic_serial)

            image_difference = abs(image_post_cti_0 - image_post_cti_1)

            assert (image_difference < 1e-9).all()

        def test__delta_functions__add_cti_only_behind_them(self, arctic_serial):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[1, 1] += 100  # Delta 1
            image_pre_cti[3, 3] += 100  # Delta 2
            image_pre_cti[2, 4] += 100  # Delta 3

            parameters = setup(include_serial=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
                               s_well_notch_depth=0.01, s_well_fill_alpha=1.0, s_well_fill_beta=0.8,
                               s_well_fill_gamma=0.0)

            image_post_cti = add_serial_cti_to_image(image=image_pre_cti, params=parameters,
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

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[0:2, :] == 0.0).all()
            assert (image_difference[2, :] < 0.0).all()
            assert (image_difference[3:-1, :] > 0.0).all()

        def test__horizontal_line__rectangular_image_even_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((4, 6))
            image_pre_cti[2, :] += 100

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[0:2, :] == 0.0).all()
            assert (image_difference[2, :] < 0.0).all()
            assert (image_difference[3:-1, :] > 0.0).all()

        def test__horizontal_line__rectangular_image_even_x_odd(self, arctic_parallel):
            image_pre_cti = np.zeros((4, 7))
            image_pre_cti[2, :] += 100

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[0:2, :] == 0.0).all()
            assert (image_difference[2, :] < 0.0).all()
            assert (image_difference[3:-1, :] > 0.0).all()

        def test__horizontal_line__rectangular_image_odd_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 6))
            image_pre_cti[2, :] += 100

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[0:2, :] == 0.0).all()
            assert (image_difference[2, :] < 0.0).all()
            assert (image_difference[3:-1, :] > 0.0).all()

        def test__vertical_line__rectangular_image_odd_x_odd(self, arctic_parallel):
            image_pre_cti = np.zeros((3, 5))
            image_pre_cti[:, 2] += 100

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[:, 0:2] == 0.0).all()
            assert (image_difference[:, 3:-1] == 0.0).all()
            assert (image_difference[:, 2] < 0.0).all()

        def test__vertical_line__rectangular_image_even_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((4, 6))
            image_pre_cti[:, 2] += 100

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[:, 0:2] == 0.0).all()
            assert (image_difference[:, 3:-1] == 0.0).all()
            assert (image_difference[:, 2] < 0.0).all()

        def test__vertical_line__rectangular_image_even_x_odd(self, arctic_parallel):
            image_pre_cti = np.zeros((4, 7))
            image_pre_cti[:, 2] += 100

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            image_difference = image_post_cti - image_pre_cti

            assert (image_difference[:, 0:2] == 0.0).all()
            assert (image_difference[:, 3:-1] == 0.0).all()
            assert (image_difference[:, 2] < 0.0).all()

        def test__vertical_line__rectangular_image_odd_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 6))
            image_pre_cti[:, 2] += 100

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
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

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
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

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
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

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
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

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
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

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=settings)

            settings_charge_inj = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=1,
                                                        p_express=5, p_n_levels=2000,
                                                        p_charge_injection_mode=True, p_readout_offset=0)

            image_post_cti_charge_inj = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
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

            ccd = arctic_params.CCD(well_fill_beta=0.8, well_notch_depth=0.01)
            species = [arctic_params.Species(trap_density=10., trap_lifetime=1.),
                       arctic_params.Species(trap_density=20., trap_lifetime=1.),
                       arctic_params.Species(trap_density=30., trap_lifetime=1.),
                       arctic_params.Species(trap_density=40., trap_lifetime=1.),
                       arctic_params.Species(trap_density=50., trap_lifetime=1.), ]

            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            image_post_cti = pyarctic.call_arctic_parallel_density_vary(image=image_pre_cti, species=species, ccd=ccd,
                                                                        settings=settings.parallel)

            # noinspection PyUnresolvedReferences
            assert (image_post_cti[2:5, 0] != image_post_cti[2:5, 1]).all()
            # noinspection PyUnresolvedReferences
            assert (image_post_cti[2:5, 1] != image_post_cti[2:5, 2]).all()
            # noinspection PyUnresolvedReferences
            assert (image_post_cti[2:5, 2] != image_post_cti[2:5, 3]).all()
            # noinspection PyUnresolvedReferences
            assert (image_post_cti[2:5, 3] != image_post_cti[2:5, 4]).all()


class TestArcticCorrectCTI:
    class TestParallel:

        def test__horizontal_line__corrected_image_more_like_original(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__vertical_line__corrected_image_more_like_original(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[:, 2] += 100

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__delta_functions__corrected_image_more_like_original(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[1, 1] += 100  # Delta 1
            image_pre_cti[3, 3] += 100  # Delta 2
            image_pre_cti[2, 4] += 100  # Delta 3

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__decrease_niter__worse_correction(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] += 100

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            arctic_niter_5 = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=5, p_express=5,
                                                   p_n_levels=2000,
                                                   p_charge_injection_mode=False, p_readout_offset=0)

            image_correct_cti = correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                settings=arctic_niter_5)

            image_difference_niter_5 = image_correct_cti - image_pre_cti

            arctic_niter_3 = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=3, p_express=5,
                                                   p_n_levels=2000,
                                                   p_charge_injection_mode=False, p_readout_offset=0)

            image_correct_cti = correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                settings=arctic_niter_3)

            image_difference_niter_3 = image_correct_cti - image_pre_cti

            # noinspection PyUnresolvedReferences
            assert (abs(image_difference_niter_5) <= abs(
                image_difference_niter_3)).all()  # First four rows should all remain zero

    class TestSerial:

        def test__horizontal_line__corrected_image_more_like_original(self, arctic_serial):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[2, :] = 100

            parameters = setup(include_serial=True, s_trap_densities=(0.2,), s_trap_lifetimes=(2.0,),
                               s_well_notch_depth=0.02, s_well_fill_beta=0.4)

            image_post_cti = add_serial_cti_to_image(image=image_pre_cti, params=parameters,
                                                     settings=arctic_serial)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = correct_serial_cti_from_image(image=image_post_cti, params=parameters,
                                                              settings=arctic_serial)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()

        def test__vertical_line__corrected_image_more_like_original(self, arctic_serial):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[:, 2] += 100

            parameters = setup(include_serial=True, s_trap_densities=(0.2,), s_trap_lifetimes=(2.0,),
                               s_well_notch_depth=0.02, s_well_fill_beta=0.4)

            image_post_cti = add_serial_cti_to_image(image=image_pre_cti, params=parameters,
                                                     settings=arctic_serial)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = correct_serial_cti_from_image(image=image_post_cti, params=parameters,
                                                              settings=arctic_serial)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()

        def test__delta_functions__corrected_image_more_like_original(self, arctic_serial):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[1, 1] += 100  # Delta 1
            image_pre_cti[3, 3] += 100  # Delta 2
            image_pre_cti[2, 4] += 100  # Delta 3

            parameters = setup(include_serial=True, s_trap_densities=(0.2,), s_trap_lifetimes=(2.0,),
                               s_well_notch_depth=0.02, s_well_fill_beta=0.4)

            image_post_cti = add_serial_cti_to_image(image=image_pre_cti, params=parameters,
                                                     settings=arctic_serial)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = correct_serial_cti_from_image(image=image_post_cti, params=parameters,
                                                              settings=arctic_serial)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__decrease_niter__worse_correction(self, arctic_serial):
            image_pre_cti = np.zeros((5, 5))
            image_pre_cti[:, 2] += 100

            parameters = setup(include_serial=True, s_trap_densities=(0.2,), s_trap_lifetimes=(2.0,),
                               s_well_notch_depth=0.02, s_well_fill_beta=0.4)

            image_post_cti = add_serial_cti_to_image(image=image_pre_cti, params=parameters,
                                                     settings=arctic_serial)

            arctic_niter_5 = arctic_settings.setup(include_serial=True, s_well_depth=84700, s_niter=5, s_express=5,
                                                   s_n_levels=2000,
                                                   s_charge_injection_mode=False, s_readout_offset=0)

            image_correct_cti = correct_serial_cti_from_image(image=image_post_cti, params=parameters,
                                                              settings=arctic_niter_5)

            image_difference_niter_5 = image_correct_cti - image_pre_cti

            arctic_niter_3 = arctic_settings.setup(include_serial=True, s_well_depth=84700, s_niter=3, s_express=5,
                                                   s_n_levels=2000,
                                                   s_charge_injection_mode=False, s_readout_offset=0)

            image_correct_cti = correct_serial_cti_from_image(image=image_post_cti, params=parameters,
                                                              settings=arctic_niter_3)

            image_difference_niter_3 = image_correct_cti - image_pre_cti

            # noinspection PyUnresolvedReferences
            assert (abs(image_difference_niter_5) <= abs(
                image_difference_niter_3)).all()  # First four rows should all remain zero

    class TestRectangularImageDimensions:

        def test__horizontal_line__odd_x_odd(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 3))
            image_pre_cti[2, :] += 100

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__horizontal_line__even_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((6, 4))
            image_pre_cti[2, :] += 100

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__horizontal_line__even_x_odd(self, arctic_parallel):
            image_pre_cti = np.zeros((6, 3))
            image_pre_cti[2, :] += 100

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__horizontal_line__odd_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 4))
            image_pre_cti[2, :] += 100

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__veritcal_line__odd_x_odd(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 3))
            image_pre_cti[:, 2] += 100

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__veritcal_line__even_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((6, 4))
            image_pre_cti[:, 2] += 100

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__veritcal_line__even_x_odd(self, arctic_parallel):
            image_pre_cti = np.zeros((6, 3))
            image_pre_cti[:, 2] += 100

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__veritcal_line__odd_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 4))
            image_pre_cti[:, 2] += 100

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__delta_functions__odd_x_odd(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 7))
            image_pre_cti[1, 1] += 100  # Delta 1
            image_pre_cti[3, 3] += 100  # Delta 2
            image_pre_cti[2, 4] += 100  # Delta 3

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__delta_functions__even_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((6, 8))
            image_pre_cti[1, 1] += 100  # Delta 1
            image_pre_cti[3, 3] += 100  # Delta 2
            image_pre_cti[2, 4] += 100  # Delta 3

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__delta_functions__even_x_odd(self, arctic_parallel):
            image_pre_cti = np.zeros((6, 7))
            image_pre_cti[1, 1] += 100  # Delta 1
            image_pre_cti[3, 3] += 100  # Delta 2
            image_pre_cti[2, 4] += 100  # Delta 3

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__delta_functions__odd_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 8))
            image_pre_cti[1, 1] += 100  # Delta 1
            image_pre_cti[3, 3] += 100  # Delta 2
            image_pre_cti[2, 4] += 100  # Delta 3

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            image_difference_1 = image_post_cti - image_pre_cti

            image_correct_cti = correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                settings=arctic_parallel)

            image_difference_2 = image_correct_cti - image_pre_cti

            assert (image_difference_2 <= abs(image_difference_1)).all()  # First four rows should all remain zero

        def test__decrease_niter__worse_correction__odd_x_odd(self, arctic_parallel):
            image_pre_cti = np.zeros((3, 5))
            image_pre_cti[2, :] += 100

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            arctic_niter_5 = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=5, p_express=5,
                                                   p_n_levels=2000,
                                                   p_charge_injection_mode=False, p_readout_offset=0)

            image_correct_cti = correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                settings=arctic_niter_5)

            image_difference_niter_5 = image_correct_cti - image_pre_cti

            arctic_niter_3 = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=3, p_express=5,
                                                   p_n_levels=2000,
                                                   p_charge_injection_mode=False, p_readout_offset=0)

            image_correct_cti = correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                settings=arctic_niter_3)

            image_difference_niter_3 = image_correct_cti - image_pre_cti

            # noinspection PyUnresolvedReferences
            assert (abs(image_difference_niter_5) <= abs(
                image_difference_niter_3)).all()  # First four rows should all remain zero

        def test__dencrease_niter__worse_correction__even_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((4, 4))
            image_pre_cti[2, :] += 100

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            arctic_niter_5 = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=5, p_express=5,
                                                   p_n_levels=2000,
                                                   p_charge_injection_mode=False, p_readout_offset=0)

            image_correct_cti = correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                settings=arctic_niter_5)

            image_difference_niter_5 = image_correct_cti - image_pre_cti

            arctic_niter_3 = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=3, p_express=5,
                                                   p_n_levels=2000,
                                                   p_charge_injection_mode=False, p_readout_offset=0)

            image_correct_cti = correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                settings=arctic_niter_3)

            image_difference_niter_3 = image_correct_cti - image_pre_cti

            # noinspection PyUnresolvedReferences
            assert (abs(image_difference_niter_5) <= abs(
                image_difference_niter_3)).all()  # First four rows should all remain zero

        def test__dencrease_niter__worse_correction__even_x_odd(self, arctic_parallel):
            image_pre_cti = np.zeros((5, 4))
            image_pre_cti[2, :] += 100

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            arctic_niter_5 = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=5, p_express=5,
                                                   p_n_levels=2000,
                                                   p_charge_injection_mode=False, p_readout_offset=0)

            image_correct_cti = correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                settings=arctic_niter_5)

            image_difference_niter_5 = image_correct_cti - image_pre_cti

            arctic_niter_3 = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=3, p_express=5,
                                                   p_n_levels=2000,
                                                   p_charge_injection_mode=False, p_readout_offset=0)

            image_correct_cti = correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                settings=arctic_niter_3)

            image_difference_niter_3 = image_correct_cti - image_pre_cti

            # noinspection PyUnresolvedReferences
            assert (abs(image_difference_niter_5) <= abs(
                image_difference_niter_3)).all()  # First four rows should all remain zero

        def test__dencrease_niter__worse_correction__odd_x_even(self, arctic_parallel):
            image_pre_cti = np.zeros((3, 6))
            image_pre_cti[2, :] += 100

            parameters = setup(include_parallel=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                               p_well_notch_depth=0.01, p_well_fill_beta=0.8)

            image_post_cti = add_parallel_cti_to_image(image=image_pre_cti, params=parameters,
                                                       settings=arctic_parallel)

            arctic_niter_5 = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=5, p_express=5,
                                                   p_n_levels=2000,
                                                   p_charge_injection_mode=False, p_readout_offset=0)

            image_correct_cti = correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                settings=arctic_niter_5)

            image_difference_niter_5 = image_correct_cti - image_pre_cti

            arctic_niter_3 = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=3, p_express=5,
                                                   p_n_levels=2000,
                                                   p_charge_injection_mode=False, p_readout_offset=0)

            image_correct_cti = correct_parallel_cti_from_image(image=image_post_cti, params=parameters,
                                                                settings=arctic_niter_3)

            image_difference_niter_3 = image_correct_cti - image_pre_cti

            # noinspection PyUnresolvedReferences
            assert (abs(image_difference_niter_5) <= abs(
                image_difference_niter_3)).all()  # First four rows should all remain zero
