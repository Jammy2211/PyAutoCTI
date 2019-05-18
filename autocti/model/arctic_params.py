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
File: python/SHE_ArCTIC/ArcticParams.pyeated on: 02/13/18
Author: James Nightingale
"""

import numpy as np


class ArcticParams(object):

    def __init__(self, parallel_ccd=None, serial_ccd=None, parallel_species=None, serial_species=None):
        """Sets up the arctic CTI model using parallel and serial parameters specified using a child of the
        ArcticParams.ParallelParams and ArcticParams.SerialParams abstract base classes.

        Parameters
        ----------
        parallel_ccd: CCD
            Class describing the state of the CCD in the parallel direction
        serial_ccd: CCD
            Class describing the state of the CCD in the serial direction
        parallel_species : [ArcticParams.ParallelParams]
           The parallel parameters for the arctic CTI model
        serial_species : [ArcticParams.SerialParams]
           The serial parameters for the arctic CTI model
        """
        self.parallel_ccd = parallel_ccd
        self.serial_ccd = serial_ccd
        self.parallel_species = parallel_species or []
        self.serial_species = serial_species or []

    @property
    def delta_ellipticity(self):
        return sum([species.delta_ellipticity for species in self.parallel_species]) + \
               sum([species.delta_ellipticity for species in self.serial_species])


class CCD(object):

    def __init__(self, well_notch_depth=1e-9, well_fill_alpha=1.0, well_fill_beta=0.58, well_fill_gamma=0.0):
        """Abstract base class of the cti model parameters. Parameters associated with the traps are set via a child \
        class.

        Parameters
        ----------
        well_notch_depth : float
            The CCD notch depth
        well_fill_alpha : float
            The volume-filling coefficient (alpha) of how an electron cloud fills the volume of a pixel.
        well_fill_beta : float
            The volume-filling power (beta) of how an electron cloud fills the volume of a pixel.
        well_fill_gamma : float
            The volume-filling constant (gamma) of how an electron cloud fills the volume of a pixel.
        """
        self.well_notch_depth = well_notch_depth
        self.well_fill_alpha = well_fill_alpha
        self.well_fill_beta = well_fill_beta
        self.well_fill_gamma = well_fill_gamma

    def __repr__(self):
        return '\n'.join(('Well Notch Depth: {}'.format(self.well_notch_depth),
                          'Well Fill Alpha: {}'.format(self.well_fill_alpha),
                          'Well Fill Beta: {}'.format(self.well_fill_beta),
                          'Well Fill Gamma: {}'.format(self.well_fill_gamma)))


class Species(object):

    def __init__(self, trap_density=0.13, trap_lifetime=0.25):
        """The CTI model parameters used for parallel clocking, using one species of trap.

        Parameters
        ----------
        trap_density : float
            The trap density of the species.
        trap_lifetime : float
            The trap lifetimes of the species.
        """
        self.trap_density = trap_density
        self.trap_lifetime = trap_lifetime

    @property
    def delta_ellipticity(self):

        a = 0.05333
        d_a = 0.03357
        d_p = 1.628
        d_w = 0.2951
        g_a = 0.09901
        g_p = 0.4553
        g_w = 0.4132

        return self.trap_density * \
               (a + d_a * (np.arctan((np.log(self.trap_lifetime) - d_p) / d_w)) +
               (g_a*np.exp(-((np.log(self.trap_lifetime) - g_p) ** 2.0) / (2 * g_w ** 2.0))))

    def update_fits_header_info(self, ext_header):
        """Output the CTI model parameters into the fits header of a fits image.

        Parameters
        -----------
        ext_header : astropy.io.hdulist
            The opened header of the astropy fits header.
        """

        def add_species(name, species_list):
            for i, species in species_list:
                ext_header.set('cte_pt{}d'.format(i), species.trap_density,
                               'Trap species {} density ({})'.format(i, name))
                ext_header.set('cte_pt{}t'.format(i), species.trap_lifetime,
                               'Trap species {} lifetime ({})'.format(i, name))

        add_species("Parallel", self.parallel_species)
        add_species("Serial", self.serial_species)

        if self.serial_ccd is not None:
            ext_header.set('cte_swln', self.serial_ccd.well_notch_depth, 'CCD Well notch depth (Serial)')
            ext_header.set('cte_swlp', self.serial_ccd.well_fill_beta, 'CCD Well filling power (Serial)')

        if self.parallel_ccd is not None:
            ext_header.set('cte_pwln', self.parallel_ccd.well_notch_depth, 'CCD Well notch depth (Parallel)')
            ext_header.set('cte_pwlp', self.parallel_ccd.well_fill_beta, 'CCD Well filling power (Parallel)')

        return ext_header

    def __repr__(self):
        return "\n".join(('Trap Density: {}'.format(self.trap_density),
                          'Trap Lifetime: {}'.format(self.trap_lifetime)))

    @classmethod
    def poisson_species(cls, species, shape, seed=0):
        """For a set of traps with a given set of densities (which are in traps per pixel), compute a new set of \
        trap densities by drawing new values for from a Poisson distribution.

        This requires us to first convert each trap density to the total number of traps in the column.

        This is used to model the random distribution of traps on a CCD, which changes the number of traps in each \
        column.

        Parameters
        -----------
        species
        shape : (int, int)
            The shape of the image, so that the correct number of trap densities are computed.
        seed : int
            The seed of the Poisson random number generator.
        """
        np.random.seed(seed)
        total_traps = tuple(map(lambda sp: sp.trap_density * shape[0], species))
        poisson_densities = [np.random.poisson(total_traps) / shape[0] for _ in range(shape[1])]
        poisson_species = []
        for densities in poisson_densities:
            for i, s in enumerate(species):
                poisson_species.append(Species(trap_density=densities[i], trap_lifetime=s.trap_lifetime))

        return poisson_species
