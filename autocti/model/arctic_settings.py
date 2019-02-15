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
File: python/SHE_ArCTIC/ArcticSettings.py

Created on: 02/13/18
Author: James Nightingale
"""


class ArcticSettings(object):

    def __init__(self, neomode='NEO', parallel=None, serial=None):
        """Sets up the CTI settings for parallel and serial clocking, specified using the \
        ArcticSettings.Settings and ArcticSettings.Settings classes.
        
        Params
        ----------
        neomode : str
            Which function arctic uses for charge clocking. Arctic has multiple clocking algorithms which are \
            numerically equivalent. The 'NEO' algorithm is currently (25/09/17) recommended.
            *Classic* - clocks charge using Massey et al 2014 algorithim without efficiency tricks.
            *NEO* - clocks charge using a trap accounting scheme to speed up code (~50x)
        parallel : ArcticSetup.Settings
            The arctic settings used for clocking in the parallel direction.
        serial : :ArcticSetup.Settings
            The arctic settings used for clocking in the serial direction.
        """
        self.neomode = neomode

        self.parallel = parallel
        self.serial = serial

    def update_fits_header_info(self, ext_header):
        """Output the Arctic Settings into the fits header of a fits image.

        Params
        -----------
        ext_header : astropy.io.hdulist
            The opened header of the astropy fits header.
        """

        if self.parallel is not None:
            ext_header.set('cte_pite', self.parallel.niter, 'Iterations Used In Correction (Parallel)')
            ext_header.set('cte_pwld', self.parallel.well_depth, 'CCD Well Depth (Parallel)')
            ext_header.set('cte_pnts', self.parallel.n_levels, 'Number of levels (Parallel)')

        if self.serial is not None:
            ext_header.set('cte_site', self.serial.niter, 'Iterations Used In Correction (Serial)')
            ext_header.set('cte_swld', self.serial.well_depth, 'CCD Well Depth (Serial)')
            ext_header.set('cte_snts', self.serial.n_levels, 'Number of levels (Serial)')

        return ext_header


class Settings(object):

    def __init__(self, well_depth, niter, express, n_levels, charge_injection_mode=False, readout_offset=0):
        """
        The CTI settings for parallel clocking.

        Returns
        --------
        well_depth : int
            The full well depth of the CCD.
        niter : int
            If CTI is being corrected, niter determines the number of times clocking is run to perform the \
            correction via forward modeling. For adding CTI only one run is required and niter is ignored.
        express : int
            The factor by which pixel-to-pixel transfers are combined for efficiency.
        n_levels : int
            Relic of old arctic code, not used anymore and will be removed in future.
        charge_injection_mode : bool
            If True, clocking is performed in charge injection line mode, where each pixel is clocked and therefore \
             trailed by traps over the entire CCD (as opposed to its distance from the CCD register).
        readout_offset : int
            Introduces an offset which increases the number of transfers each pixel takes in the parallel direction.
        """
        self.well_depth = well_depth
        self.niter = niter
        self.express = express
        self.n_levels = n_levels
        self.charge_injection_mode = charge_injection_mode
        self.readout_offset = readout_offset

    def update_fits_header_info(self, ext_header):
        """Update a fits header to include the parallel CTI settings.

        Params
        -----------
        ext_header : astropy.io.hdulist
            The opened header of the astropy fits header.
        """
        ext_header.set('cte_pite', self.niter, 'Iterations Used In Correction (Parallel)')
        ext_header.set('cte_pwld', self.well_depth, 'CCD Well Depth (Parallel)')
        ext_header.set('cte_pnts', self.n_levels, 'Number of levels (Parallel)')
        return ext_header
