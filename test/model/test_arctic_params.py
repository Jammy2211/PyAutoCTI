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
File: tests/python/ArcticParams_test.py

Created on: 02/13/18
Author: James Nightingale
"""

import os
import shutil

import pytest

from autocti.model import arctic_params


@pytest.fixture(name='info_path')
def test_info():
    info_path = "{}/files/cti_params/info/".format(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(info_path):
        shutil.rmtree(info_path)

    os.mkdir(info_path)

    return info_path


@pytest.fixture(name='hdr_path')
def test_header_info():
    hdr_path = "{}/files/cti_params/header_info/".format(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(hdr_path):
        shutil.rmtree(hdr_path)

    os.mkdir(hdr_path)

    return hdr_path


class TestParallelParams:
    class TestConstructor:

        def test__1_species__sets_values_correctly(self):
            parallel_1_species = arctic_params.Species(trap_density=0.1, trap_lifetime=1.0)

            parameters = arctic_params.ArcticParams(parallel_species=parallel_1_species)

            assert type(parameters) == arctic_params.ArcticParams
            assert type(parameters.parallel_species) == arctic_params.Species

            assert parameters.parallel_species.trap_density == 0.1
            assert parameters.parallel_species.trap_lifetime == 1.0

    class TestInfoFile:

        def test__species__output_info_file_follows_the_correct_format(self, info_path):
            parallel_1_species = arctic_params.Species(trap_density=0.1, trap_lifetime=1.0)

            parameters = arctic_params.ArcticParams(parallel_species=[parallel_1_species])

            parameters.output_info_file(path=info_path)

            parameters_file = open(info_path + 'ArcticParams.info')

            parameters = parameters_file.readlines()

            assert parameters[0] == r'parallel_trap_density = 0.1' + '\n'
            assert parameters[1] == r'parallel_trap_lifetime = 1.0' + '\n'

        def test__ccd__output_info_file_follows_the_correct_format(self, info_path):
            ccd = arctic_params.CCD(
                well_notch_depth=0.01, well_fill_alpha=0.2,
                well_fill_beta=0.8, well_fill_gamma=2.0)

            parameters = arctic_params.ArcticParams(parallel_ccd=ccd)

            parameters.output_info_file(path=info_path)

            parameters_file = open(info_path + 'ArcticParams.info')

            parameters = parameters_file.readlines()

            assert parameters[0] == r'parallel_well_notch_depth = 0.01' + '\n'
            assert parameters[1] == r'parallel_well_fill_alpha = 0.2' + '\n'
            assert parameters[2] == r'parallel_well_fill_beta = 0.8' + '\n'
            assert parameters[3] == r'parallel_well_fill_gamma = 2.0' + '\n'


class TestParallelAndSerialParams:
    class TestInfoFile:

        def test__1_species__output_info_file_follows_the_correct_format(self):
            parallel_1_species = arctic_params.Species(trap_density=0.1, trap_lifetime=1.0)

            parallel_ccd = arctic_params.CCD(well_notch_depth=0.01, well_fill_alpha=0.2,
                                             well_fill_beta=0.8, well_fill_gamma=2.0)

            serial_1_species = arctic_params.Species(trap_density=0.2, trap_lifetime=2.0)

            serial_ccd = arctic_params.CCD(
                well_notch_depth=0.02, well_fill_alpha=0.1,
                well_fill_beta=0.4, well_fill_gamma=0.6)

            parameters = arctic_params.ArcticParams(parallel_species=[parallel_1_species],
                                                    serial_species=[serial_1_species], parallel_ccd=parallel_ccd,
                                                    serial_ccd=serial_ccd)

            parameters = parameters.generate_info().split("\n")

            assert parameters[0] == r'parallel_trap_density = 0.1'
            assert parameters[1] == r'parallel_trap_lifetime = 1.0'
            assert parameters[3] == r'serial_trap_density = 0.2'
            assert parameters[4] == r'serial_trap_lifetime = 2.0'
            assert parameters[5] == r''
            assert parameters[6] == r'parallel_well_notch_depth = 0.01'
            assert parameters[7] == r'parallel_well_fill_alpha = 0.2'
            assert parameters[8] == r'parallel_well_fill_beta = 0.8'
            assert parameters[9] == r'parallel_well_fill_gamma = 2.0'
            assert parameters[10] == r''
            assert parameters[11] == r'serial_well_notch_depth = 0.02'
            assert parameters[12] == r'serial_well_fill_alpha = 0.1'
            assert parameters[13] == r'serial_well_fill_beta = 0.4'
            assert parameters[14] == r'serial_well_fill_gamma = 0.6'


class TestParallelDensityVary:

    def test_1_species__density_01__1000_column_pixels__1_row_pixel_so_100_traps__posison_density_near_01(self):
        parallel_vary = arctic_params.ParallelDensityVary.poisson_densities(trap_densities=(0.1,),
                                                                            trap_lifetimes=(1.0,),
                                                                            well_notch_depth=0.01, well_fill_alpha=1.0,
                                                                            well_fill_beta=0.8, well_fill_gamma=0.0,
                                                                            shape=(1000, 1),
                                                                            seed=1)

        assert parallel_vary.trap_densities == [(0.098,)]

    def test__1_species__density_1__1000_column_pixels_so_1000_traps__1_row_pixel__posison_value_is_near_1(self):
        parallel_vary = arctic_params.ParallelDensityVary.poisson_densities(trap_densities=(1.0,),
                                                                            trap_lifetimes=(1.0,),
                                                                            well_notch_depth=0.01, well_fill_alpha=1.0,
                                                                            well_fill_beta=0.8,
                                                                            well_fill_gamma=0.0, shape=(1000, 1),
                                                                            seed=1)

        assert parallel_vary.trap_densities == [(0.992,)]

    def test__1_species__density_1___2_row_pixels__posison_value_is_near_1(self):
        parallel_vary = arctic_params.ParallelDensityVary.poisson_densities(trap_densities=(1.0,),
                                                                            trap_lifetimes=(1.0,),
                                                                            well_notch_depth=0.01, well_fill_alpha=1.0,
                                                                            well_fill_beta=0.8,
                                                                            well_fill_gamma=0.0, shape=(1000, 2),
                                                                            seed=1)

        assert parallel_vary.trap_densities == [(0.992,), (0.962,)]

    def test__2_species__1_row_pixel__poisson_for_each_species_drawn(self):
        parallel_vary = arctic_params.ParallelDensityVary.poisson_densities(trap_densities=(1.0, 2.0),
                                                                            trap_lifetimes=(1.0, 2.0),
                                                                            well_notch_depth=0.01, well_fill_alpha=1.0,
                                                                            well_fill_beta=0.8,
                                                                            well_fill_gamma=0.0, shape=(1000, 1),
                                                                            seed=1)

        assert parallel_vary.trap_densities == [(0.992, 1.946)]

    def test__2_species__2_row_pixel__poisson_for_each_species_drawn(self):
        parallel_vary = arctic_params.ParallelDensityVary.poisson_densities(trap_densities=(1.0, 2.0),
                                                                            trap_lifetimes=(1.0, 2.0),
                                                                            well_notch_depth=0.01, well_fill_alpha=1.0,
                                                                            well_fill_beta=0.8,
                                                                            well_fill_gamma=0.0, shape=(1000, 2),
                                                                            seed=1)

        assert parallel_vary.trap_densities == [(0.992, 1.946), (0.968, 1.987)]

    def test__same_as_above_but_3_species_and_new_values(self):
        parallel_vary = arctic_params.ParallelDensityVary.poisson_densities(trap_densities=(1.0, 2.0, 0.1),
                                                                            trap_lifetimes=(1.0, 2.0, 3.0),
                                                                            well_fill_alpha=1.0, well_fill_beta=0.8,
                                                                            well_fill_gamma=0.0,
                                                                            well_notch_depth=0.01, shape=(1000, 3),
                                                                            seed=1)

        assert parallel_vary.trap_densities == [(0.992, 1.946, 0.09), (0.991, 1.99, 0.098), (0.961, 1.975, 0.113)]
