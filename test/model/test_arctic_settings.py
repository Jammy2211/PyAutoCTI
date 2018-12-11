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
File: tests/python/ArcticSettings_test.py

Created on: 02/13/18
Author: James Nightingale
"""

from __future__ import division, print_function
import sys
import os




import pytest
import os
import shutil
from astropy.io import fits
import numpy as np

from autocti.model import arctic_settings

@pytest.fixture(name='info_path')
def test_info():

    info_path = "{}/files/settings/info/".format(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(info_path):
        shutil.rmtree(info_path)

    os.mkdir(info_path)

    return info_path

@pytest.fixture(name='hdr_path')
def test_header_info():

    hdr_path = "{}/files/settings/header_info/".format(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(hdr_path):
        shutil.rmtree(hdr_path)

    os.mkdir(hdr_path)

    return hdr_path

class TestFactory:

    def test__sets_up_settings_parallel__with_correct_values(self):

        arctic_parallel = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=1, p_express=5, p_n_levels=2000,
                                           p_charge_injection_mode=True, p_readout_offset=0)

        assert arctic_parallel.neomode == 'NEO'

        assert arctic_parallel.parallel.well_depth == 84700
        assert arctic_parallel.parallel.niter == 1
        assert arctic_parallel.parallel.express == 5
        assert arctic_parallel.parallel.n_levels == 2000
        assert arctic_parallel.parallel.charge_injection_mode is True
        assert arctic_parallel.parallel.readout_offset == 0

        assert arctic_parallel.serial is None

    def test__sets_up_settings_serial__with_correct_values(self):

        arctic_serial = arctic_settings.setup(include_serial=True, s_well_depth= 84700, s_niter=1, s_express=5, s_n_levels=2000,
                                           s_charge_injection_mode=False, s_readout_offset=0)

        assert arctic_serial.neomode == 'NEO'

        assert arctic_serial.parallel is None

        assert arctic_serial.serial.well_depth == 84700
        assert arctic_serial.serial.niter == 1
        assert arctic_serial.serial.express == 5
        assert arctic_serial.serial.n_levels == 2000
        assert arctic_serial.serial.charge_injection_mode is False
        assert arctic_serial.serial.readout_offset == 0

    def test__sets_up_parameters_both_directions__with_correct_values(self):

        arctic_both = arctic_settings.setup(include_parallel=True, p_well_depth=84700, p_niter=1, p_express=5, p_n_levels=2000,
                            p_charge_injection_mode=True, p_readout_offset=0,
                                           include_serial=True, s_well_depth= 84700, s_niter=1, s_express=5, s_n_levels=2000,
                            s_charge_injection_mode=False, s_readout_offset=0)

        assert arctic_both.neomode == 'NEO'

        assert arctic_both.parallel.well_depth == 84700
        assert arctic_both.parallel.niter == 1
        assert arctic_both.parallel.express == 5
        assert arctic_both.parallel.n_levels == 2000
        assert arctic_both.parallel.charge_injection_mode is True
        assert arctic_both.parallel.readout_offset == 0

        assert arctic_both.serial.well_depth == 84700
        assert arctic_both.serial.niter == 1
        assert arctic_both.serial.express == 5
        assert arctic_both.serial.n_levels == 2000
        assert arctic_both.serial.charge_injection_mode is False
        assert arctic_both.serial.readout_offset == 0


class TestArcticSettings:


    class TestConstructor:

        def test__sets_up_parameters_with_correct_values(self):

            parallel_settings = arctic_settings.ParallelSettings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                            charge_injection_mode=True, readout_offset=0)

            serial_settings = arctic_settings.SerialSettings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                                            readout_offset=0)

            arctic_both = arctic_settings.ArcticSettings(neomode='NEO', parallel=parallel_settings,
                                                        serial=serial_settings)
    
            assert arctic_both.neomode == 'NEO'
    
            assert arctic_both.parallel.well_depth == 84700
            assert arctic_both.parallel.niter == 1
            assert arctic_both.parallel.express == 5
            assert arctic_both.parallel.n_levels == 2000
            assert arctic_both.parallel.charge_injection_mode is True
            assert arctic_both.parallel.readout_offset == 0
    
            assert arctic_both.serial.well_depth == 84700
            assert arctic_both.serial.niter == 1
            assert arctic_both.serial.express == 5
            assert arctic_both.serial.n_levels == 2000
            assert arctic_both.serial.charge_injection_mode is False
            assert arctic_both.serial.readout_offset == 0


    class TestInfoFile:
        
        def test__parallel_only__output_file_follows_correct_format(self, info_path):

            parallel_settings = arctic_settings.ParallelSettings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                                                charge_injection_mode=False, readout_offset=0)

            arctic_parallel = arctic_settings.ArcticSettings(neomode='NEO', parallel=parallel_settings)

            arctic_parallel.output_info_file(path=info_path)

            settings_file = open(info_path+'ArcticSettings.info')

            settings = settings_file.readlines()

            assert settings[0] == r'parallel_mode = True' + '\n'
            assert settings[1] == r'' + '\n'
            assert settings[2] == r'parallel_well_depth = 84700' + '\n'
            assert settings[3] == r'parallel_niter = 1' + '\n'
            assert settings[4] == r'parallel_express = 5' + '\n'
            assert settings[5] == r'parallel_n_levels = 2000' + '\n'
            assert settings[6] == r'parallel_charge_injection_mode = False' + '\n'
            assert settings[7] == r'parallel_readout_offset = 0' + '\n'

        def test__serial_only__output_file_follows_correct_format(self, info_path):

            serial_settings = arctic_settings.SerialSettings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                                            readout_offset=0)

            arctic_serial = arctic_settings.ArcticSettings(neomode='NEO', serial=serial_settings)

            arctic_serial.output_info_file(path=info_path)

            settings_file = open(info_path+'ArcticSettings.info')

            settings = settings_file.readlines()

            assert settings[0] == r'serial_mode = True' + '\n'
            assert settings[1] == r'' + '\n'
            assert settings[2] == r'serial_well_depth = 84700' + '\n'
            assert settings[3] == r'serial_niter = 1' + '\n'
            assert settings[4] == r'serial_express = 5' + '\n'
            assert settings[5] == r'serial_n_levels = 2000' + '\n'
            assert settings[6] == r'serial_charge_injection_mode = False' + '\n'
            assert settings[7] == r'serial_readout_offset = 0' + '\n'
            
        def test__parallel_and_serial__output_file_follows_correct_format(self, info_path):

            parallel_settings = arctic_settings.ParallelSettings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                                                charge_injection_mode=True, readout_offset=0)

            serial_settings = arctic_settings.SerialSettings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                                            readout_offset=0)

            arctic_both = arctic_settings.ArcticSettings(neomode='NEO', parallel=parallel_settings,
                                                        serial=serial_settings)

            arctic_both.output_info_file(path=info_path)

            settings_file = open(info_path+'ArcticSettings.info')

            settings = settings_file.readlines()

            assert settings[0] == r'parallel_mode = True' + '\n'
            assert settings[1] == r'' + '\n'
            assert settings[2] == r'parallel_well_depth = 84700' + '\n'
            assert settings[3] == r'parallel_niter = 1' + '\n'
            assert settings[4] == r'parallel_express = 5' + '\n'
            assert settings[5] == r'parallel_n_levels = 2000' + '\n'
            assert settings[6] == r'parallel_charge_injection_mode = True' + '\n'
            assert settings[7] == r'parallel_readout_offset = 0' + '\n'
            assert settings[8] == r'' + '\n'
            assert settings[9] == r'serial_mode = True' + '\n'
            assert settings[10] == r'' + '\n'
            assert settings[11] == r'serial_well_depth = 84700' + '\n'
            assert settings[12] == r'serial_niter = 1' + '\n'
            assert settings[13] == r'serial_express = 5' + '\n'
            assert settings[14] == r'serial_n_levels = 2000' + '\n'
            assert settings[15] == r'serial_charge_injection_mode = False' + '\n'
            assert settings[16] == r'serial_readout_offset = 0' + '\n'


    class TestFitsHeaderInfo:

        def test__parallel_only__sets_up_header_info_consistent_with_previous_vis_pf(self, hdr_path):

            parallel_settings = arctic_settings.ParallelSettings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                                                charge_injection_mode=False, readout_offset=0)

            arctic_parallel = arctic_settings.ArcticSettings(neomode='NEO', parallel=parallel_settings)

            hdu = fits.PrimaryHDU(np.ones((1,1)), fits.Header())
            hdu.header = arctic_parallel.update_fits_header_info(ext_header=hdu.header)
            hdu.writeto(hdr_path + '/test.fits')

            hdu = fits.open(hdr_path+'/test.fits')
            ext_header = hdu[0].header

            assert ext_header['cte_pite'] == 1
            assert ext_header['cte_pwld'] == 84700
            assert ext_header['cte_pnts'] == 2000
            
        def test__serial_only__sets_up_header_info_consistent_with_previous_vis_pf(self, hdr_path):

            serial_settings = arctic_settings.SerialSettings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                                                charge_injection_mode=False, readout_offset=0)

            arctic_serial = arctic_settings.ArcticSettings(neomode='NEO', serial=serial_settings)

            hdu = fits.PrimaryHDU(np.ones((1,1)), fits.Header())
            hdu.header = arctic_serial.update_fits_header_info(ext_header=hdu.header)
            hdu.writeto(hdr_path + '/test.fits')

            hdu = fits.open(hdr_path+'/test.fits')
            ext_header = hdu[0].header

            assert ext_header['cte_site'] == 1
            assert ext_header['cte_swld'] == 84700
            assert ext_header['cte_snts'] == 2000

        def test__parallel_and_serial__sets_up_header_info_consistent_with_previous_vis_pf(self, hdr_path):

            parallel_settings = arctic_settings.ParallelSettings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                                                charge_injection_mode=False, readout_offset=0)

            serial_settings = arctic_settings.SerialSettings(well_depth=84700, niter=1, express=5, n_levels=2000,
                                                                charge_injection_mode=False, readout_offset=0)

            arctic_both = arctic_settings.ArcticSettings(neomode='NEO', parallel=parallel_settings,
                                                        serial=serial_settings)

            hdu = fits.PrimaryHDU(np.ones((1,1)), fits.Header())
            hdu.header = arctic_both.update_fits_header_info(ext_header=hdu.header)
            hdu.writeto(hdr_path + '/test.fits')

            hdu = fits.open(hdr_path+'/test.fits')
            ext_header = hdu[0].header

            assert ext_header['cte_pite'] == 1
            assert ext_header['cte_pwld'] == 84700
            assert ext_header['cte_pnts'] == 2000

            assert ext_header['cte_site'] == 1
            assert ext_header['cte_swld'] == 84700
            assert ext_header['cte_snts'] == 2000