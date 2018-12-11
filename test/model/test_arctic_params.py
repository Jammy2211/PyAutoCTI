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

from __future__ import division, print_function
import sys
import os

if sys.version_info[0] < 3:
    from future_builtins import *

import pytest
import shutil
from astropy.io import fits
import numpy as np

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


class TestFactories:


    class TestSetupParallel:

        def test__1_species_in__sets_up_correct_class_instance(self):

            parameters = arctic_params.setup(p=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
            p_well_notch_depth=0.01, p_well_fill_alpha=0.1, p_well_fill_beta=0.8, p_well_fill_gamma=0.5)

            assert type(parameters) == arctic_params.ArcticParams
            assert type(parameters.parallel) == arctic_params.ParallelOneSpecies

            assert parameters.parallel.trap_densities[0] == (0.1)
            assert parameters.parallel.trap_lifetimes[0] == (1.0)
            assert parameters.parallel.well_notch_depth == 0.01
            assert parameters.parallel.well_fill_alpha == 0.1
            assert parameters.parallel.well_fill_beta == 0.8
            assert parameters.parallel.well_fill_gamma == 0.5

        def test__2_species_in__sets_up_correct_class_instance(self):

            parameters = arctic_params.setup(p=True, p_trap_densities=(0.1, 0.3), p_trap_lifetimes=(1.0, 10.0),
                                            p_well_notch_depth=0.01, p_well_fill_alpha=0.1, p_well_fill_beta=0.8,
                                            p_well_fill_gamma=0.5)

            assert type(parameters) == arctic_params.ArcticParams
            assert type(parameters.parallel) == arctic_params.ParallelTwoSpecies

            assert parameters.parallel.trap_densities == (0.1, 0.3)
            assert parameters.parallel.trap_lifetimes == (1.0, 10.0)
            assert parameters.parallel.well_notch_depth == 0.01
            assert parameters.parallel.well_fill_alpha == 0.1
            assert parameters.parallel.well_fill_beta == 0.8
            assert parameters.parallel.well_fill_gamma == 0.5

        def test__3_species_in__sets_up_correct_class_instance(self):

            parameters = arctic_params.setup(p=True, p_trap_densities=(0.1, 0.3, 0.5), p_trap_lifetimes=(1.0, 10.0, 100.0),
                                            p_well_notch_depth=0.01, p_well_fill_alpha=0.1, p_well_fill_beta=0.8,
                                            p_well_fill_gamma=0.5)

            assert type(parameters) == arctic_params.ArcticParams
            assert type(parameters.parallel) == arctic_params.ParallelThreeSpecies

            assert parameters.parallel.trap_densities == (0.1, 0.3, 0.5)
            assert parameters.parallel.trap_lifetimes == (1.0, 10.0, 100.0)
            assert parameters.parallel.well_notch_depth == 0.01
            assert parameters.parallel.well_fill_alpha == 0.1
            assert parameters.parallel.well_fill_beta == 0.8
            assert parameters.parallel.well_fill_gamma == 0.5

        def test__different_number_of_trap_densities_and_lifetimes__raises_error(self):

            with pytest.raises(AttributeError):
                parameters = arctic_params.setup(p=True, p_trap_densities=(0.2,), p_trap_lifetimes=(2.0, 4.0),
                                                p_well_notch_depth=0.02, p_well_fill_beta=0.4)


    class TestSetupSerial:

        def test__1_species_in__sets_up_correct_class_instance(self):

            parameters = arctic_params.setup(s=True, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,),
            s_well_notch_depth=0.01, s_well_fill_alpha=0.6, s_well_fill_beta=0.8, s_well_fill_gamma=-0.1)

            assert type(parameters) == arctic_params.ArcticParams
            assert type(parameters.serial) == arctic_params.SerialOneSpecies

            assert parameters.serial.trap_densities[0] == (0.1)
            assert parameters.serial.trap_lifetimes[0] == (1.0)
            assert parameters.serial.well_notch_depth == 0.01
            assert parameters.serial.well_fill_alpha == 0.6
            assert parameters.serial.well_fill_beta == 0.8
            assert parameters.serial.well_fill_gamma == -0.1

        def test__2_species_in__sets_up_correct_class_instance(self):
            
            parameters = arctic_params.setup(s=True, s_trap_densities=(0.1, 0.3), s_trap_lifetimes=(1.0, 10.0),
                                            s_well_notch_depth=0.01, s_well_fill_alpha=0.6, s_well_fill_beta=0.8,
                                            s_well_fill_gamma=-0.1)

            assert type(parameters) == arctic_params.ArcticParams
            assert type(parameters.serial) == arctic_params.SerialTwoSpecies

            assert parameters.serial.trap_densities == (0.1, 0.3)
            assert parameters.serial.trap_lifetimes == (1.0, 10.0)
            assert parameters.serial.well_notch_depth == 0.01
            assert parameters.serial.well_fill_alpha == 0.6
            assert parameters.serial.well_fill_beta == 0.8
            assert parameters.serial.well_fill_gamma == -0.1

        def test__3_species_in__sets_up_correct_class_instance(self):

            parameters = arctic_params.setup(s=True, s_trap_densities=(0.1, 0.3, 0.5), s_trap_lifetimes=(1.0, 10.0, 100.0),
                                            s_well_notch_depth=0.01, s_well_fill_alpha=0.6, s_well_fill_beta=0.8,
                                            s_well_fill_gamma=-0.1)

            assert type(parameters) == arctic_params.ArcticParams
            assert type(parameters.serial) == arctic_params.SerialThreeSpecies

            assert parameters.serial.trap_densities == (0.1, 0.3, 0.5)
            assert parameters.serial.trap_lifetimes == (1.0, 10.0, 100.0)
            assert parameters.serial.well_notch_depth == 0.01
            assert parameters.serial.well_fill_alpha == 0.6
            assert parameters.serial.well_fill_beta == 0.8
            assert parameters.serial.well_fill_gamma == -0.1

        def test__different_number_of_trap_densities_and_lifetimes__raises_error(self):

            with pytest.raises(AttributeError):
                parameters = arctic_params.setup(s=True, s_trap_densities=(0.2,), s_trap_lifetimes=(2.0, 4.0),
                                                s_well_notch_depth=0.02, s_well_fill_beta=0.4)


    class TestSetupBoth:
        
        def test__1_parallel_1_serial__sets_up_class_instance_correctly(self):

            parameters = arctic_params.setup(p=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                            p_well_notch_depth=0.01, p_well_fill_alpha=0.1, p_well_fill_beta=0.8,
                                            p_well_fill_gamma=0.5,
                                            s=True, s_trap_densities=(0.2,), s_trap_lifetimes=(2.0,),
                                            s_well_notch_depth=0.02, s_well_fill_alpha=0.6, s_well_fill_beta=0.8,
                                            s_well_fill_gamma=-0.1)

            assert type(parameters) == arctic_params.ArcticParams

            assert type(parameters.parallel) == arctic_params.ParallelOneSpecies

            assert parameters.parallel.trap_densities == (0.1,)
            assert parameters.parallel.trap_lifetimes == (1.0,)
            assert parameters.parallel.well_notch_depth == 0.01
            assert parameters.parallel.well_fill_alpha == 0.1
            assert parameters.parallel.well_fill_beta == 0.8
            assert parameters.parallel.well_fill_gamma == 0.5

            assert type(parameters.serial) == arctic_params.SerialOneSpecies

            assert parameters.serial.trap_densities == (0.2,)
            assert parameters.serial.trap_lifetimes == (2.0,)
            assert parameters.serial.well_notch_depth == 0.02
            assert parameters.serial.well_fill_alpha == 0.6
            assert parameters.serial.well_fill_beta == 0.8
            assert parameters.serial.well_fill_gamma == -0.1


        def test__2_parallel_1_serial__sets_up_class_instance_correctly(self):

            parameters = arctic_params.setup(p=True, p_trap_densities=(0.1, 0.3), p_trap_lifetimes=(1.0, 3.0),
                                            p_well_notch_depth=0.01, p_well_fill_alpha=0.1, p_well_fill_beta=0.8,
                                            p_well_fill_gamma=0.5,
                                            s=True, s_trap_densities=(0.2,), s_trap_lifetimes=(2.0,),
                                            s_well_notch_depth=0.02, s_well_fill_alpha=0.6, s_well_fill_beta=0.8,
                                            s_well_fill_gamma=-0.1)

            assert type(parameters) == arctic_params.ArcticParams

            assert type(parameters.parallel) == arctic_params.ParallelTwoSpecies

            assert parameters.parallel.trap_densities == (0.1, 0.3)
            assert parameters.parallel.trap_lifetimes == (1.0, 3.0)
            assert parameters.parallel.well_notch_depth == 0.01
            assert parameters.parallel.well_fill_alpha == 0.1
            assert parameters.parallel.well_fill_beta == 0.8
            assert parameters.parallel.well_fill_gamma == 0.5

            assert type(parameters.serial) == arctic_params.SerialOneSpecies

            assert parameters.serial.trap_densities == (0.2,)
            assert parameters.serial.trap_lifetimes == (2.0,)
            assert parameters.serial.well_notch_depth == 0.02
            assert parameters.serial.well_fill_alpha == 0.6
            assert parameters.serial.well_fill_beta == 0.8
            assert parameters.serial.well_fill_gamma == -0.1


        def test__3_parallel_3_serial__sets_up_class_instance_correctly(self):

            parameters = arctic_params.setup(p=True, p_trap_densities=(0.1, 0.3, 0.5), p_trap_lifetimes=(1.0, 3.0, 5.0),
                                            p_well_notch_depth=0.01, p_well_fill_alpha=0.1, p_well_fill_beta=0.8,
                                            p_well_fill_gamma=0.5,
                                            s=True, s_trap_densities=(0.2, 0.4, 0.6), s_trap_lifetimes=(2.0, 4.0, 6.0),
                                            s_well_notch_depth=0.02, s_well_fill_alpha=0.6, s_well_fill_beta=0.8,
                                            s_well_fill_gamma=-0.1)

            assert type(parameters) == arctic_params.ArcticParams

            assert type(parameters.parallel) == arctic_params.ParallelThreeSpecies

            assert parameters.parallel.trap_densities == (0.1, 0.3, 0.5)
            assert parameters.parallel.trap_lifetimes == (1.0, 3.0, 5.0)
            assert parameters.parallel.well_notch_depth == 0.01
            assert parameters.parallel.well_fill_alpha == 0.1
            assert parameters.parallel.well_fill_beta == 0.8
            assert parameters.parallel.well_fill_gamma == 0.5

            assert type(parameters.serial) == arctic_params.SerialThreeSpecies

            assert parameters.serial.trap_densities == (0.2, 0.4, 0.6)
            assert parameters.serial.trap_lifetimes == (2.0, 4.0, 6.0)
            assert parameters.serial.well_notch_depth == 0.02
            assert parameters.serial.well_fill_alpha == 0.6
            assert parameters.serial.well_fill_beta == 0.8
            assert parameters.serial.well_fill_gamma == -0.1


        def test__different_number_of_parallel_trap_densities_and_lifetimes__raises_error(self):

            with pytest.raises(AttributeError):
                parameters = arctic_params.setup(p=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0, 2.0),
                                                p_well_notch_depth=0.01, p_well_fill_beta=0.8,
                                                s=True, s_trap_densities=(0.2,), s_trap_lifetimes=(2.0,),
                                                s_well_notch_depth=0.02, s_well_fill_beta=0.4)

        def test__different_number_of_s_trap_densities_and_lifetimes__raises_error(self):

            with pytest.raises(AttributeError):
                parameters = arctic_params.setup(p=True, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,),
                                                p_well_notch_depth=0.01, p_well_fill_beta=0.8,
                                                s=True, s_trap_densities=(0.2,), s_trap_lifetimes=(2.0, 4.0),
                                                s_well_notch_depth=0.02, s_well_fill_beta=0.4)


class TestParallelParams:


    class TestConstructor:

        def test__1_species__sets_values_correctly(self):

            parallel_1_species = arctic_params.ParallelOneSpecies(trap_densities=(0.1,), trap_lifetimes=(1.0,),
            well_notch_depth=0.01, well_fill_alpha=0.1, well_fill_beta=0.8, well_fill_gamma=-0.1)

            parameters = arctic_params.ArcticParams(parallel=parallel_1_species)

            assert type(parameters) == arctic_params.ArcticParams
            assert type(parameters.parallel) == arctic_params.ParallelOneSpecies

            assert parameters.parallel.trap_densities[0] == (0.1)
            assert parameters.parallel.trap_lifetimes[0] == (1.0)
            assert parameters.parallel.well_notch_depth == 0.01
            assert parameters.parallel.well_fill_alpha == 0.1
            assert parameters.parallel.well_fill_beta == 0.8
            assert parameters.parallel.well_fill_gamma == -0.1

        def test__2_species__sets_values_correctly(self):

            parallel_2_species = arctic_params.ParallelTwoSpecies(trap_densities=(0.1, 0.3),
                                                                 trap_lifetimes=(1.0, 10.0),
                                                                 well_notch_depth=0.01, well_fill_alpha=0.1,
                                                                 well_fill_beta=0.8, well_fill_gamma=-0.1)

            parameters = arctic_params.ArcticParams(parallel=parallel_2_species)

            assert type(parameters) == arctic_params.ArcticParams
            assert type(parameters.parallel) == arctic_params.ParallelTwoSpecies

            assert parameters.parallel.trap_densities == (0.1, 0.3)
            assert parameters.parallel.trap_lifetimes == (1.0, 10.0)
            assert parameters.parallel.well_notch_depth == 0.01
            assert parameters.parallel.well_fill_alpha == 0.1
            assert parameters.parallel.well_fill_beta == 0.8
            assert parameters.parallel.well_fill_gamma == -0.1

        def test__3_species__sets_values_correctly(self):

            parallel_3_species = arctic_params.ParallelThreeSpecies(trap_densities=(0.1, 0.3, 0.5),
                                                                   trap_lifetimes=(1.0, 10.0, 100.0),
                                                                   well_notch_depth=0.01, well_fill_alpha=0.1,
                                                                   well_fill_beta=0.8, well_fill_gamma=-0.1)

            parameters = arctic_params.ArcticParams(parallel=parallel_3_species)

            assert type(parameters) == arctic_params.ArcticParams
            assert type(parameters.parallel) == arctic_params.ParallelThreeSpecies

            assert parameters.parallel.trap_densities == (0.1, 0.3, 0.5)
            assert parameters.parallel.trap_lifetimes == (1.0, 10.0, 100.0)
            assert parameters.parallel.well_notch_depth == 0.01
            assert parameters.parallel.well_fill_alpha == 0.1
            assert parameters.parallel.well_fill_beta == 0.8
            assert parameters.parallel.well_fill_gamma == -0.1


    class TestInfoFile:

        def test__1_species__output_info_file_follows_the_correct_format(self, info_path):

            parallel_1_species = arctic_params.ParallelOneSpecies(trap_densities=(0.1,), trap_lifetimes=(1.0,),
                            well_notch_depth=0.01, well_fill_alpha=0.2, well_fill_beta=0.8, well_fill_gamma=2.0)

            parameters = arctic_params.ArcticParams(parallel=parallel_1_species)

            parameters.output_info_file(path=info_path)

            parameters_file = open(info_path+'ArcticParams.info')

            parameters = parameters_file.readlines()

            assert parameters[0] == r'parallel_trap_densities = (0.1,)' + '\n'
            assert parameters[1] == r'parallel_trap_lifetimes = (1.0,)' + '\n'
            assert parameters[2] == r'parallel_well_notch_depth = 0.01' + '\n'
            assert parameters[3] == r'parallel_well_fill_alpha = 0.2' + '\n'
            assert parameters[4] == r'parallel_well_fill_beta = 0.8' + '\n'
            assert parameters[5] == r'parallel_well_fill_gamma = 2.0' + '\n'

        def test__3_species__output_info_file_follows_the_correct_format(self, info_path):

            parallel_3_species = arctic_params.ParallelThreeSpecies(trap_densities=(0.1, 0.3, 0.5),
                                                                   trap_lifetimes=(1.0, 10.0, 100.0),
                                                                   well_notch_depth=0.01, well_fill_alpha=0.2,
                                                                   well_fill_beta=0.8, well_fill_gamma=2.0)

            parameters = arctic_params.ArcticParams(parallel=parallel_3_species)

            parameters.output_info_file(path=info_path)

            parameters_file = open(info_path+'ArcticParams.info')

            parameters = parameters_file.readlines()

            assert parameters[0] == r'parallel_trap_densities = (0.1, 0.3, 0.5)' + '\n'
            assert parameters[1] == r'parallel_trap_lifetimes = (1.0, 10.0, 100.0)' + '\n'
            assert parameters[2] == r'parallel_well_notch_depth = 0.01' + '\n'
            assert parameters[3] == r'parallel_well_fill_alpha = 0.2' + '\n'
            assert parameters[4] == r'parallel_well_fill_beta = 0.8' + '\n'
            assert parameters[5] == r'parallel_well_fill_gamma = 2.0' + '\n'


    class TestUpdateFitsHeaderInfo:
        
        def test__1_species__sets_up_header_info_consistent_with_previous_vis_pf(self, hdr_path):

            parallel_1_species = arctic_params.ParallelOneSpecies(trap_densities=(0.1,), trap_lifetimes=(1.0,),
                                                                 well_notch_depth=0.01, well_fill_beta=0.8)

            parameters = arctic_params.ArcticParams(parallel=parallel_1_species)

            hdu = fits.PrimaryHDU(np.ones((1,1)), fits.Header())
            hdu.header = parameters.update_fits_header_info(ext_header=hdu.header)
            hdu.writeto(hdr_path + '/test.fits')

            hdu = fits.open(hdr_path+'/test.fits')
            ext_header = hdu[0].header

            assert ext_header['cte_pt1d'] == 0.1
            assert ext_header['cte_pt1t'] == 1.0
            assert ext_header['cte_pwln'] == 0.01
            assert ext_header['cte_pwlp'] == 0.8

        def test__3_species__sets_up_header_info_consistent_with_previous_vis_pf(self, hdr_path):

            parallel_3_species = arctic_params.ParallelThreeSpecies(trap_densities=(0.1, 0.3, 0.5),
                                                                   trap_lifetimes=(1.0, 10.0, 100.0),
                                                                   well_notch_depth=0.01, well_fill_beta=0.8)

            parameters = arctic_params.ArcticParams(parallel=parallel_3_species)

            hdu = fits.PrimaryHDU(np.ones((1,1)), fits.Header())
            hdu.header = parameters.update_fits_header_info(ext_header=hdu.header)
            hdu.writeto(hdr_path + '/test.fits')

            hdu = fits.open(hdr_path+'/test.fits')
            ext_header = hdu[0].header

            assert ext_header['cte_pt1d'] == 0.1
            assert ext_header['cte_pt1t'] == 1.0
            assert ext_header['cte_pt2d'] == 0.3
            assert ext_header['cte_pt2t'] == 10.0
            assert ext_header['cte_pt3d'] == 0.5
            assert ext_header['cte_pt3t'] == 100.0
            assert ext_header['cte_pwln'] == 0.01
            assert ext_header['cte_pwlp'] == 0.8


class TestSerialParams:


    class TestConstructor:

        def test__1_species__sets_values_correctly(self):

            serial_1_species = arctic_params.SerialOneSpecies(trap_densities=(0.2,), trap_lifetimes=(2.0,),
            well_notch_depth=0.02, well_fill_alpha=1.0, well_fill_beta=0.4, well_fill_gamma=0.5)

            parameters = arctic_params.ArcticParams(serial=serial_1_species)

            assert type(parameters) == arctic_params.ArcticParams
            assert type(parameters.serial) == arctic_params.SerialOneSpecies
    
            assert parameters.serial.trap_densities == (0.2,)
            assert parameters.serial.trap_lifetimes == (2.0,)
            assert parameters.serial.well_notch_depth == 0.02
            assert parameters.serial.well_fill_alpha == 1.0
            assert parameters.serial.well_fill_beta == 0.4
            assert parameters.serial.well_fill_gamma == 0.5
    
        def test__2_species__sets_values_correctly(self):

            serial_2_species = arctic_params.SerialTwoSpecies(trap_densities=(0.2, 0.6),
                                                             trap_lifetimes=(2.0, 20.0),
                                                             well_notch_depth=0.02, well_fill_alpha=1.0,
                                                             well_fill_beta=0.4, well_fill_gamma=0.5)

            parameters = arctic_params.ArcticParams(serial=serial_2_species)

            assert type(parameters) == arctic_params.ArcticParams
            assert type(parameters.serial) == arctic_params.SerialTwoSpecies
    
            assert parameters.serial.trap_densities == (0.2, 0.6)
            assert parameters.serial.trap_lifetimes == (2.0, 20.0)
            assert parameters.serial.well_notch_depth == 0.02
            assert parameters.serial.well_fill_alpha == 1.0
            assert parameters.serial.well_fill_beta == 0.4
            assert parameters.serial.well_fill_gamma == 0.5
    
        def test__3_species__sets_values_correctly(self):

            serial_3_species = arctic_params.SerialThreeSpecies(trap_densities=(0.2, 0.6, 1.0),
                                                               trap_lifetimes=(2.0, 20.0, 200.0),
                                                               well_notch_depth=0.02, well_fill_alpha=1.0,
                                                               well_fill_beta=0.4, well_fill_gamma=0.5)

            parameters = arctic_params.ArcticParams(serial=serial_3_species)

            assert type(parameters) == arctic_params.ArcticParams
            assert type(parameters.serial) == arctic_params.SerialThreeSpecies

            assert parameters.serial.trap_densities == (0.2, 0.6, 1.0)
            assert parameters.serial.trap_lifetimes == (2.0, 20.0, 200.0)
            assert parameters.serial.well_notch_depth == 0.02
            assert parameters.serial.well_fill_alpha == 1.0
            assert parameters.serial.well_fill_beta == 0.4
            assert parameters.serial.well_fill_gamma == 0.5


    class TestInfoFile:

        def test__1_species__output_info_file_follows_the_correct_format(self, info_path):

            serial_1_species = arctic_params.SerialOneSpecies(trap_densities=(0.2,), trap_lifetimes=(2.0,),
                             well_notch_depth=0.02, well_fill_alpha=0.1, well_fill_beta=0.4, well_fill_gamma=0.6)

            parameters = arctic_params.ArcticParams(serial=serial_1_species)

            parameters.output_info_file(path=info_path)

            parameters_file = open(info_path+'ArcticParams.info')

            parameters = parameters_file.readlines()

            assert parameters[0] == r'serial_trap_densities = (0.2,)' + '\n'
            assert parameters[1] == r'serial_trap_lifetimes = (2.0,)' + '\n'
            assert parameters[2] == r'serial_well_notch_depth = 0.02' + '\n'
            assert parameters[3] == r'serial_well_fill_alpha = 0.1' + '\n'
            assert parameters[4] == r'serial_well_fill_beta = 0.4' + '\n'
            assert parameters[5] == r'serial_well_fill_gamma = 0.6' + '\n'

        def test__3_species__output_info_file_follows_the_correct_format(self, info_path):

            serial_3_species = arctic_params.SerialThreeSpecies(trap_densities=(0.2, 0.6, 1.0),
                                                               trap_lifetimes=(2.0, 20.0, 200.0),
                                                               well_notch_depth=0.02, well_fill_alpha=0.1,
                                                               well_fill_beta=0.4, well_fill_gamma=0.6)

            parameters = arctic_params.ArcticParams(serial=serial_3_species)

            parameters.output_info_file(path=info_path)

            parameters_file = open(info_path+'ArcticParams.info')

            parameters = parameters_file.readlines()

            assert parameters[0] == r'serial_trap_densities = (0.2, 0.6, 1.0)' + '\n'
            assert parameters[1] == r'serial_trap_lifetimes = (2.0, 20.0, 200.0)' + '\n'
            assert parameters[2] == r'serial_well_notch_depth = 0.02' + '\n'
            assert parameters[3] == r'serial_well_fill_alpha = 0.1' + '\n'
            assert parameters[4] == r'serial_well_fill_beta = 0.4' + '\n'
            assert parameters[5] == r'serial_well_fill_gamma = 0.6' + '\n'


    class TestUpdateFitsHeaderInfo:

        def test__1_species__sets_up_header_info_consistent_with_previous_vis_pf(self, hdr_path):

            serial_1_species = arctic_params.SerialOneSpecies(trap_densities=(0.2,), trap_lifetimes=(2.0,),
                                                             well_notch_depth=0.02, well_fill_beta=0.4)

            parameters = arctic_params.ArcticParams(serial=serial_1_species)

            hdu = fits.PrimaryHDU(np.ones((1,1)), fits.Header())
            hdu.header = parameters.update_fits_header_info(ext_header=hdu.header)
            hdu.writeto(hdr_path + '/test.fits')

            hdu = fits.open(hdr_path+'/test.fits')
            ext_header = hdu[0].header

            assert ext_header['cte_st1d'] == 0.2
            assert ext_header['cte_st1t'] == 2.0
            assert ext_header['cte_swln'] == 0.02
            assert ext_header['cte_swlp'] == 0.4

        def test__3_species__sets_up_header_info_consistent_with_previous_vis_pf(self, hdr_path):

            serial_3_species = arctic_params.SerialThreeSpecies(trap_densities=(0.2, 0.6, 1.0),
                                                               trap_lifetimes=(2.0, 20.0, 200.0),
                                                               well_notch_depth=0.02, well_fill_beta=0.4)

            parameters = arctic_params.ArcticParams(serial=serial_3_species)

            hdu = fits.PrimaryHDU(np.ones((1,1)), fits.Header())
            hdu.header = parameters.update_fits_header_info(ext_header=hdu.header)
            hdu.writeto(hdr_path + '/test.fits')

            hdu = fits.open(hdr_path+'/test.fits')
            ext_header = hdu[0].header


            assert ext_header['cte_st1d'] == 0.2
            assert ext_header['cte_st1t'] == 2.0
            assert ext_header['cte_st2d'] == 0.6
            assert ext_header['cte_st2t'] == 20.0
            assert ext_header['cte_st3d'] == 1.0
            assert ext_header['cte_st3t'] == 200.0
            assert ext_header['cte_swln'] == 0.02
            assert ext_header['cte_swlp'] == 0.4


class TestParallelAndSerialParams:


    class TestConstructor:

        def test__1_parallel_1_serial__sets_values_correctly(self):

            parallel_1_species = arctic_params.ParallelOneSpecies(trap_densities=(0.1,), trap_lifetimes=(1.0,),
                                                                 well_notch_depth=0.01, well_fill_alpha=0.1,
                                                                 well_fill_beta=0.8, well_fill_gamma=-0.1)

            serial_1_species = arctic_params.SerialOneSpecies(trap_densities=(0.2,), trap_lifetimes=(2.0,),
                                                             well_notch_depth=0.02, well_fill_alpha=1.0,
                                                             well_fill_beta=0.4, well_fill_gamma=0.5)

            parameters = arctic_params.ArcticParams(parallel=parallel_1_species,
                                                   serial=serial_1_species)

            assert type(parameters) == arctic_params.ArcticParams

            assert type(parameters.parallel) == arctic_params.ParallelOneSpecies

            assert parameters.parallel.trap_densities == (0.1,)
            assert parameters.parallel.trap_lifetimes == (1.0,)
            assert parameters.parallel.well_notch_depth == 0.01
            assert parameters.parallel.well_fill_alpha == 0.1
            assert parameters.parallel.well_fill_beta == 0.8
            assert parameters.parallel.well_fill_gamma == -0.1

            assert type(parameters.serial) == arctic_params.SerialOneSpecies

            assert parameters.serial.trap_densities == (0.2,)
            assert parameters.serial.trap_lifetimes == (2.0,)
            assert parameters.serial.well_notch_depth == 0.02
            assert parameters.serial.well_fill_alpha == 1.0
            assert parameters.serial.well_fill_beta == 0.4
            assert parameters.serial.well_fill_gamma == 0.5

        def test__3_parallel_2_serial__sets_values_correctly(self):

            parallel_3_species = arctic_params.ParallelThreeSpecies(trap_densities=(0.1, 0.3, 0.5),
                                                                   trap_lifetimes=(1.0, 10.0, 100.0),
                                                                   well_notch_depth=0.01, well_fill_alpha=0.1,
                                                                   well_fill_beta=0.8, well_fill_gamma=-0.1)

            serial_2_species = arctic_params.SerialTwoSpecies(trap_densities=(0.2, 0.6),
                                                             trap_lifetimes=(2.0, 20.0),
                                                             well_notch_depth=0.02, well_fill_alpha=1.0,
                                                             well_fill_beta=0.4, well_fill_gamma=0.5)

            parameters = arctic_params.ArcticParams(parallel=parallel_3_species, serial=serial_2_species)

            assert type(parameters) == arctic_params.ArcticParams

            assert type(parameters.parallel) == arctic_params.ParallelThreeSpecies

            assert parameters.parallel.trap_densities == (0.1, 0.3, 0.5)
            assert parameters.parallel.trap_lifetimes == (1.0, 10.0, 100.0)
            assert parameters.parallel.well_notch_depth == 0.01
            assert parameters.parallel.well_fill_alpha == 0.1
            assert parameters.parallel.well_fill_beta == 0.8
            assert parameters.parallel.well_fill_gamma == -0.1

            assert type(parameters.serial) == arctic_params.SerialTwoSpecies

            assert parameters.serial.trap_densities == (0.2, 0.6)
            assert parameters.serial.trap_lifetimes == (2.0, 20.0)
            assert parameters.serial.well_notch_depth == 0.02
            assert parameters.serial.well_fill_alpha == 1.0
            assert parameters.serial.well_fill_beta == 0.4
            assert parameters.serial.well_fill_gamma == 0.5


    class TestInfoFile:

        def test__1_species__output_info_file_follows_the_correct_format(self, info_path):

            parallel_1_species = arctic_params.ParallelOneSpecies(trap_densities=(0.1,), trap_lifetimes=(1.0,),
                                                                 well_notch_depth=0.01, well_fill_alpha=0.2,
                                                                 well_fill_beta=0.8, well_fill_gamma=2.0)

            serial_1_species = arctic_params.SerialOneSpecies(trap_densities=(0.2,), trap_lifetimes=(2.0,),
                                                             well_notch_depth=0.02, well_fill_alpha=0.1,
                                                             well_fill_beta=0.4, well_fill_gamma=0.6)

            parameters = arctic_params.ArcticParams(parallel=parallel_1_species,
                                                   serial=serial_1_species)

            parameters.output_info_file(path=info_path)

            parameters_file = open(info_path+'ArcticParams.info')

            parameters = parameters_file.readlines()

            assert parameters[0] == r'parallel_trap_densities = (0.1,)' + '\n'
            assert parameters[1] == r'parallel_trap_lifetimes = (1.0,)' + '\n'
            assert parameters[2] == r'parallel_well_notch_depth = 0.01' + '\n'
            assert parameters[3] == r'parallel_well_fill_alpha = 0.2' + '\n'
            assert parameters[4] == r'parallel_well_fill_beta = 0.8' + '\n'
            assert parameters[5] == r'parallel_well_fill_gamma = 2.0' + '\n'
            assert parameters[6] == '\n'
            assert parameters[7] == r'serial_trap_densities = (0.2,)' + '\n'
            assert parameters[8] == r'serial_trap_lifetimes = (2.0,)' + '\n'
            assert parameters[9] == r'serial_well_notch_depth = 0.02' + '\n'
            assert parameters[10] == r'serial_well_fill_alpha = 0.1' + '\n'
            assert parameters[11] == r'serial_well_fill_beta = 0.4' + '\n'
            assert parameters[12] == r'serial_well_fill_gamma = 0.6' + '\n'

        def test__3_species__output_info_file_follows_the_correct_format(self, info_path):

            parallel_3_species = arctic_params.ParallelThreeSpecies(trap_densities=(0.1, 0.3, 0.5),
                                                                   trap_lifetimes=(1.0, 10.0, 100.0),
                                                                   well_notch_depth=0.01, well_fill_alpha=0.2,
                                                                   well_fill_beta=0.8, well_fill_gamma=2.0)

            serial_3_species = arctic_params.SerialThreeSpecies(trap_densities=(0.2, 0.6, 1.0),
                                                               trap_lifetimes=(2.0, 20.0, 200.0),
                                                               well_notch_depth=0.02, well_fill_alpha=0.1,
                                                               well_fill_beta=0.4, well_fill_gamma=0.6)

            parameters = arctic_params.ArcticParams(parallel=parallel_3_species,
                                                   serial=serial_3_species)

            parameters.output_info_file(path=info_path)

            parameters_file = open(info_path+'ArcticParams.info')

            parameters = parameters_file.readlines()

            assert parameters[0] == r'parallel_trap_densities = (0.1, 0.3, 0.5)' + '\n'
            assert parameters[1] == r'parallel_trap_lifetimes = (1.0, 10.0, 100.0)' + '\n'
            assert parameters[2] == r'parallel_well_notch_depth = 0.01' + '\n'
            assert parameters[3] == r'parallel_well_fill_alpha = 0.2' + '\n'
            assert parameters[4] == r'parallel_well_fill_beta = 0.8' + '\n'
            assert parameters[5] == r'parallel_well_fill_gamma = 2.0' + '\n'
            assert parameters[6] == '\n'
            assert parameters[7] == r'serial_trap_densities = (0.2, 0.6, 1.0)' + '\n'
            assert parameters[8] == r'serial_trap_lifetimes = (2.0, 20.0, 200.0)' + '\n'
            assert parameters[9] == r'serial_well_notch_depth = 0.02' + '\n'
            assert parameters[10] == r'serial_well_fill_alpha = 0.1' + '\n'
            assert parameters[11] == r'serial_well_fill_beta = 0.4' + '\n'
            assert parameters[12] == r'serial_well_fill_gamma = 0.6' + '\n'


    class TestUpdateFitsHeaderInfo:

        def test__1_species__sets_up_header_info_consistent_with_previous_vis_pf(self, hdr_path):

            parallel_1_species = arctic_params.ParallelOneSpecies(trap_densities=(0.1,), trap_lifetimes=(1.0,),
                                                                 well_notch_depth=0.01, well_fill_beta=0.8)

            serial_1_species = arctic_params.SerialOneSpecies(trap_densities=(0.2,), trap_lifetimes=(2.0,),
                                                             well_notch_depth=0.02, well_fill_beta=0.4)

            parameters = arctic_params.ArcticParams(parallel=parallel_1_species,
                                                   serial=serial_1_species)

            hdu = fits.PrimaryHDU(np.ones((1,1)), fits.Header())
            hdu.header = parameters.update_fits_header_info(ext_header=hdu.header)
            hdu.writeto(hdr_path + '/test.fits')

            hdu = fits.open(hdr_path+'/test.fits')
            ext_header = hdu[0].header

            assert ext_header['cte_pt1d'] == 0.1
            assert ext_header['cte_pt1t'] == 1.0
            assert ext_header['cte_pwln'] == 0.01
            assert ext_header['cte_pwlp'] == 0.8

            assert ext_header['cte_st1d'] == 0.2
            assert ext_header['cte_st1t'] == 2.0
            assert ext_header['cte_swln'] == 0.02
            assert ext_header['cte_swlp'] == 0.4

        def test__3_species__sets_up_header_info_consistent_with_previous_vis_pf(self, hdr_path):

            parallel_3_species = arctic_params.ParallelThreeSpecies(trap_densities=(0.1, 0.3, 0.5),
                                                                   trap_lifetimes=(1.0, 10.0, 100.0),
                                                                   well_notch_depth=0.01, well_fill_beta=0.8)

            serial_3_species = arctic_params.SerialThreeSpecies(trap_densities=(0.2, 0.6, 1.0),
                                                               trap_lifetimes=(2.0, 20.0, 200.0),
                                                               well_notch_depth=0.02, well_fill_beta=0.4)

            parameters = arctic_params.ArcticParams(parallel=parallel_3_species,
                                                   serial=serial_3_species)

            hdu = fits.PrimaryHDU(np.ones((1,1)), fits.Header())
            hdu.header = parameters.update_fits_header_info(ext_header=hdu.header)
            hdu.writeto(hdr_path + '/test.fits')

            hdu = fits.open(hdr_path+'/test.fits')
            ext_header = hdu[0].header

            assert ext_header['cte_pt1d'] == 0.1
            assert ext_header['cte_pt1t'] == 1.0
            assert ext_header['cte_pt2d'] == 0.3
            assert ext_header['cte_pt2t'] == 10.0
            assert ext_header['cte_pt3d'] == 0.5
            assert ext_header['cte_pt3t'] == 100.0
            assert ext_header['cte_pwln'] == 0.01
            assert ext_header['cte_pwlp'] == 0.8

            assert ext_header['cte_st1d'] == 0.2
            assert ext_header['cte_st1t'] == 2.0
            assert ext_header['cte_st2d'] == 0.6
            assert ext_header['cte_st2t'] == 20.0
            assert ext_header['cte_st3d'] == 1.0
            assert ext_header['cte_st3t'] == 200.0
            assert ext_header['cte_swln'] == 0.02
            assert ext_header['cte_swlp'] == 0.4


class TestParallelDensityVary:

    def test_1_species__density_01__1000_column_pixels__1_row_pixel_so_100_traps__posison_density_near_01(self):

        parallel_vary = arctic_params.ParallelDensityVary.poisson_densities(trap_densities=(0.1,),
    trap_lifetimes=(1.0,), well_notch_depth=0.01, well_fill_alpha=1.0, well_fill_beta=0.8, well_fill_gamma=0.0, shape=(1000, 1),
                                                                           seed=1)

        assert parallel_vary.trap_densities == [(0.098,)]

    def test__1_species__density_1__1000_column_pixels_so_1000_traps__1_row_pixel__posison_value_is_near_1(self):

        parallel_vary = arctic_params.ParallelDensityVary.poisson_densities(trap_densities=(1.0,),
          trap_lifetimes=(1.0,),  well_notch_depth=0.01, well_fill_alpha=1.0, well_fill_beta=0.8,
                                                                well_fill_gamma=0.0,shape=(1000, 1), seed=1)

        assert parallel_vary.trap_densities == [(0.992,)]

    def test__1_species__density_1___2_row_pixels__posison_value_is_near_1(self):

        parallel_vary = arctic_params.ParallelDensityVary.poisson_densities(trap_densities=(1.0,),
        trap_lifetimes=(1.0,),  well_notch_depth=0.01, well_fill_alpha=1.0, well_fill_beta=0.8,
                                                                           well_fill_gamma=0.0, shape=(1000, 2), seed=1)

        assert parallel_vary.trap_densities == [(0.992,), (0.962,)]

    def test__2_species__1_row_pixel__poisson_for_each_species_drawn(self):

        parallel_vary = arctic_params.ParallelDensityVary.poisson_densities(trap_densities=(1.0, 2.0),
    trap_lifetimes=(1.0, 2.0),  well_notch_depth=0.01, well_fill_alpha=1.0, well_fill_beta=0.8,
                                                                           well_fill_gamma=0.0, shape=(1000, 1), seed=1)

        assert parallel_vary.trap_densities == [(0.992, 1.946)]

    def test__2_species__2_row_pixel__poisson_for_each_species_drawn(self):

        parallel_vary = arctic_params.ParallelDensityVary.poisson_densities(trap_densities=(1.0, 2.0),
        trap_lifetimes=(1.0, 2.0),  well_notch_depth=0.01, well_fill_alpha=1.0, well_fill_beta=0.8,
                                                                           well_fill_gamma=0.0, shape=(1000, 2), seed=1)

        assert parallel_vary.trap_densities == [(0.992, 1.946), (0.968, 1.987)]

    def test__same_as_above_but_3_species_and_new_values(self):

        parallel_vary = arctic_params.ParallelDensityVary.poisson_densities(trap_densities=(1.0, 2.0, 0.1),
        trap_lifetimes=(1.0, 2.0, 3.0), well_fill_alpha=1.0, well_fill_beta=0.8, well_fill_gamma=0.0,
                                    well_notch_depth=0.01, shape=(1000, 3), seed=1)

        assert parallel_vary.trap_densities == [(0.992, 1.946, 0.09), (0.991, 1.99, 0.098), (0.961, 1.975, 0.113)]
