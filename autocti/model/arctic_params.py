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

from autocti.tools import infoio


def setup(include_serial=False, p_trap_densities=(0.1,), p_trap_lifetimes=(1.0,), p_well_notch_depth=0.0001,
          p_well_fill_alpha=1.0, p_well_fill_beta=0.8, p_well_fill_gamma=0.0,
          include_parallel=False, s_trap_densities=(0.1,), s_trap_lifetimes=(1.0,), s_well_notch_depth=0.0001,
          s_well_fill_alpha=1.0, s_well_fill_beta=0.8, s_well_fill_gamma=0.0):
    """Factory to set up a *ParallelParams* and / or *SerialParams* sub-class as an *ArcticParams*
    instance using any number of trap species in both directions.

    Parameters
    ----------
    include_parallel: Bool
        If True parallel parameters will be included in the ArcticParams object
    include_serial: Bool
        If True serial parameters will be included in the ArcticParams object
    p_trap_densities : (float,)
        The trap density of the parallel species
    p_trap_lifetimes : (float,)
        The trap lifetimes of the parallel species
    p_well_notch_depth : float
        The CCD notch depth for parallel clocking
    p_well_fill_alpha : float
        The volume-filling coefficient (alpha) of how an electron cloud fills the volume of a pixel for parallel
        clocking.
    p_well_fill_beta : float
        The volume-filling power (beta) of how an electron cloud fills the volume of a pixel for parallel clocking
    p_well_fill_gamma : float
        The volume-filling constant (gamma) of how an electron cloud fills the volume of a pixel for parallel clocking
    s_trap_densities : (float,)
        The trap density of the serial species
    s_trap_lifetimes : (float,)
        The trap lifetimes of the serial species
    s_well_notch_depth : float
        The CCD notch depth for serial clocking
    s_well_fill_alpha : float
        The volume-filling coefficient (alpha) of how an electron cloud fills the volume of a pixel for serial clocking.
    s_well_fill_beta : float
        The volume-filling power (beta) of how an electron cloud fills the volume of a pixel for serial clocking
    s_well_fill_gamma : float
        The volume-filling constant (gamma) of how an electron cloud fills the volume of a pixel for serial clocking
    """

    parallel_parameters = _setup_parallel(p_trap_densities, p_trap_lifetimes, p_well_notch_depth,
                                          p_well_fill_alpha, p_well_fill_beta,
                                          p_well_fill_gamma) if include_parallel else None

    serial_parameters = _setup_serial(s_trap_densities, s_trap_lifetimes, s_well_notch_depth,
                                      s_well_fill_alpha, s_well_fill_beta,
                                      s_well_fill_gamma) if include_serial else None

    return ArcticParams(parallel=parallel_parameters, serial=serial_parameters)


def _setup_parallel(trap_densities, trap_lifetimes, well_notch_depth, well_fill_alpha, well_fill_beta,
                    well_fill_gamma):
    """Setup the parallel parameters for the factory above"""

    infoio.check_all_tuples_and_equal_length(trap_densities, trap_lifetimes)

    parallel_no_species = len(trap_densities)

    if parallel_no_species == 1:
        return ParallelOneSpecies(trap_densities, trap_lifetimes, well_notch_depth,
                                  well_fill_alpha, well_fill_beta, well_fill_gamma)
    elif parallel_no_species == 2:
        return ParallelTwoSpecies(trap_densities, trap_lifetimes, well_notch_depth,
                                  well_fill_alpha, well_fill_beta, well_fill_gamma)
    elif parallel_no_species == 3:
        return ParallelThreeSpecies(trap_densities, trap_lifetimes, well_notch_depth,
                                    well_fill_alpha, well_fill_beta, well_fill_gamma)
    else:
        raise AttributeError('The number of parallel trap species must be > 0 and < 3')


def _setup_serial(trap_densities, trap_lifetimes, well_notch_depth, well_fill_alpha, well_fill_beta,
                  well_fill_gamma):
    """Setup the serial parameters for the factory above"""

    infoio.check_all_tuples_and_equal_length(trap_densities, trap_lifetimes)

    serial_no_species = len(trap_densities)

    if serial_no_species == 1:
        return SerialOneSpecies(trap_densities, trap_lifetimes, well_notch_depth,
                                well_fill_alpha, well_fill_beta, well_fill_gamma)
    elif serial_no_species == 2:
        return SerialTwoSpecies(trap_densities, trap_lifetimes, well_notch_depth,
                                well_fill_alpha, well_fill_beta, well_fill_gamma)
    elif serial_no_species == 3:
        return SerialThreeSpecies(trap_densities, trap_lifetimes, well_notch_depth,
                                  well_fill_alpha, well_fill_beta, well_fill_gamma)
    else:
        raise AttributeError('The number of serial trap species must be > 0 and < 3')


class ArcticParams(object):

    def __init__(self, parallel=None, serial=None):
        """Sets up the arctic CTI model using parallel and serial parameters specified using a child of the
        ArcticParams.ParallelParams and ArcticParams.SerialParams abstract base classes.

        Parameters
        ----------
        parallel : ArcticParams.ParallelParams
           The parallel parameters for the arctic CTI model
        serial : ArcticParams.SerialParams
           The serial parameters for the arctic CTI model
        """
        self.parallel = parallel
        self.serial = serial

    def output_info_file(self, path, filename='ArcticParams'):
        """Output information on the the parameters to a text file.

        Parameters
        ----------
        filename: str
            The output filename
        path : str
            The output directory path of the ci_data
        """
        infoio.output_class_info(self, path, filename)

    def generate_info(self):
        """Generate string containing information on the the arctic parameters."""

        info = ''

        if self.parallel is not None:
            info += self.parallel.generate_info()

        if self.serial is not None:
            info += self.serial.generate_info()

        return info

    def get_object_tag(self, tag=''):

        object_tag = ''

        if self.parallel is not None:
            object_tag += self.parallel.object_tag

        if self.serial is not None:

            if self.parallel is not None:
                object_tag += '_'

            object_tag += self.serial.object_tag

        return object_tag + tag

    def update_fits_header_info(self, ext_header):
        """Output the CTI model parameters into the fits header of a fits image.

        Parameters
        -----------
        ext_header : astropy.io.hdulist
            The opened header of the astropy fits header.
        """
        if self.parallel is not None:
            self.parallel.update_fits_header_info(ext_header)
        if self.serial is not None:
            self.serial.update_fits_header_info(ext_header)

        return ext_header


class Params(object):

    def __init__(self, trap_densities, trap_lifetimes, well_notch_depth, well_fill_alpha, well_fill_beta,
                 well_fill_gamma):
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

        self.trap_densities = trap_densities
        self.trap_lifetimes = trap_lifetimes
        self.well_notch_depth = well_notch_depth
        self.well_fill_alpha = well_fill_alpha
        self.well_fill_beta = well_fill_beta
        self.well_fill_gamma = well_fill_gamma

    def __repr__(self):
        string = "Number Species: {}".format(len(self.trap_lifetimes)) + '\n'
        string += 'Trap Densities: {}'.format(self.trap_densities) + '\n'
        string += 'Trap Lifetimes: {}'.format(self.trap_lifetimes) + '\n'
        string += 'Well Notch Depth: {}'.format(self.well_notch_depth) + '\n'
        string += 'Well Fill Alpha: {}'.format(self.well_fill_alpha) + '\n'
        string += 'Well Fill Beta: {}'.format(self.well_fill_beta) + '\n'
        string += 'Well Fill Gamma: {}'.format(self.well_fill_gamma) + '\n'
        return string


class ParallelParams(Params):

    def __init__(self, trap_densities, trap_lifetimes, well_notch_depth, well_fill_alpha, well_fill_beta,
                 well_fill_gamma):
        """Abstract base class of the cti model parameters for parallel clocking. Params associated with the traps \
         are set via a child class.

        Parameters
        ----------
        trap_densities : tuple
            The trap density of the species
        trap_lifetimes : tuple
            The trap lifetimes of the species            
        well_notch_depth : float
            The CCD notch depth
        well_fill_alpha : float
            The volume-filling coefficient (alpha) of how an electron cloud fills the volume of a pixel.
        well_fill_beta : float
            The volume-filling power (beta) of how an electron cloud fills the volume of a pixel.
        well_fill_gamma : float
            The volume-filling constant (gamma) of how an electron cloud fills the volume of a pixel.
        """

        super(ParallelParams, self).__init__(trap_densities, trap_lifetimes, well_notch_depth,
                                             well_fill_alpha, well_fill_beta, well_fill_gamma)

    def update_fits_header_info(self, ext_header):
        """Update a fits header to include the parallel CTI model parameters.

        Params
        -----------
        ext_header : astropy.io.hdulist
            The opened header of the astropy fits header.
        """
        ext_header.set('cte_pt1d', self.trap_densities[0], 'First trap species density (Parallel)')
        ext_header.set('cte_pt1t', self.trap_lifetimes[0], 'First trap species lifetime (Parallel)')

        no_species = len(self.trap_densities)

        if no_species > 1:
            ext_header.set('cte_pt2d', self.trap_densities[1], 'Second trap species density (Parallel)')
            ext_header.set('cte_pt2t', self.trap_lifetimes[1], 'Second trap species lifetime (Parallel)')

        if no_species > 2:
            ext_header.set('cte_pt3d', self.trap_densities[2], 'Third trap species density (Parallel)')
            ext_header.set('cte_pt3t', self.trap_lifetimes[2], 'Third trap species lifetime (Parallel)')

        ext_header.set('cte_pwln', self.well_notch_depth, 'CCD Well notch depth (Parallel)')
        ext_header.set('cte_pwlp', self.well_fill_beta, 'CCD Well filling power (Parallel)')

        return ext_header

    def generate_info(self):
        """Generate string containing information on the parallel parameters."""
        info = ''
        info += infoio.generate_class_info(cls=self, prefix='parallel_', include_types=[int, float, tuple])
        info += '\n'
        return info


class ParallelOneSpecies(ParallelParams):

    def __init__(self, trap_densities=(0.13,), trap_lifetimes=(0.25,), well_notch_depth=1e-9, well_fill_alpha=1.0,
                 well_fill_beta=0.58, well_fill_gamma=0.0):
        """The CTI model parameters used for parallel clocking, using one species of trap.

        Parameters
        ----------
        trap_densities : (float,)
            The trap density of the species.
        trap_lifetimes : (float,)
            The trap lifetimes of the species.
        well_notch_depth : float
            The CCD notch depth.
        well_fill_alpha : float
            The volume-filling coefficient (alpha) of how an electron cloud fills the volume of a pixel.
        well_fill_beta : float
            The volume-filling power (beta) of how an electron cloud fills the volume of a pixel.
        well_fill_gamma : float
            The volume-filling constant (gamma) of how an electron cloud fills the volume of a pixel.

        Examples
        --------
        parallel_two_species = ParallelTwoSpecies(trap_densities=(0.1,), trap_lifetimes=(1.0,),
                                             trap_lifetimes=10.0, well_notch_depth=0.01, well_fill_beta=0.8)

        """

        super(ParallelOneSpecies, self).__init__(trap_densities, trap_lifetimes, well_notch_depth, well_fill_alpha,
                                                 well_fill_beta, well_fill_gamma)

    @property
    def object_tag(self):
        return 'p1_d' + str(self.trap_densities) + '_t' + str(self.trap_lifetimes) + '_a' + str(
            self.well_fill_alpha) + '_b' + \
               str(self.well_fill_beta) + '_g' + str(self.well_fill_gamma)


class ParallelTwoSpecies(ParallelParams):

    def __init__(self, trap_densities=(0.13, 0.25), trap_lifetimes=(0.25, 4.4), well_notch_depth=1e-9,
                 well_fill_alpha=1.0, well_fill_beta=0.58, well_fill_gamma=0.0):
        """The CTI model parameters used for parallel clocking, using two species of traps.

        Parameters
        ----------
        trap_densities : (float, float)
            The trap density of the two trap species
        trap_lifetimes : (float, float)
            The trap lifetimes of the two trap species
        well_notch_depth : float
            The CCD notch depth
        well_fill_beta : float
            The volume-filling power (beta) of an electron cloud
        well_fill_alpha : float
            The volume-filling coefficient (alpha) of how an electron cloud fills the volume of a pixel.
        well_fill_beta : float
            The volume-filling power (beta) of how an electron cloud fills the volume of a pixel.
        well_fill_gamma : float
            The volume-filling constant (gamma) of how an electron cloud fills the volume of a pixel.

        Examples
        --------
        parallel_two_species = ParallelTwoSpecies(trap_densities=(0.1, 0.2), trap_lifetimes=(1.0, 2.0),
                                             trap_lifetimes=10.0, well_notch_depth=0.01, well_fill_beta=0.8)

        """

        super(ParallelTwoSpecies, self).__init__(trap_densities, trap_lifetimes, well_notch_depth, well_fill_alpha,
                                                 well_fill_beta, well_fill_gamma)

    @property
    def object_tag(self):
        return 'p2_d' + str(self.trap_densities) + '_t' + str(self.trap_lifetimes) + '_a' + str(
            self.well_fill_alpha) + '_b' + \
               str(self.well_fill_beta) + '_g' + str(self.well_fill_gamma)


class ParallelThreeSpecies(ParallelParams):

    def __init__(self, trap_densities=(0.13, 0.25, 0.01), trap_lifetimes=(0.25, 4.4, 10.0), well_notch_depth=1e-9,
                 well_fill_alpha=1.0, well_fill_beta=0.58, well_fill_gamma=0.0):
        """The CTI model parameters used for parallel clocking, using three species of traps.

        Parameters
        ----------
        trap_densities : (float, float, float)
            The trap density of the three trap species
        trap_lifetimes : (float, float, float)
            The trap lifetimes of the three trap species
        well_notch_depth : float
            The CCD notch depth
        well_fill_alpha : float
            The volume-filling coefficient (alpha) of how an electron cloud fills the volume of a pixel.
        well_fill_beta : float
            The volume-filling power (beta) of how an electron cloud fills the volume of a pixel.
        well_fill_gamma : float
            The volume-filling constant (gamma) of how an electron cloud fills the volume of a pixel.

        Examples
        --------
        parallel_three_species = ParallelThreeSpecies(trap_densities=(0.1, 0.2, 0.3), trap_lifetimes=(1.0, 2.0, 3.0),
                                                      well_notch_depth=0.01, well_fill_beta=0.8)

        """

        super(ParallelThreeSpecies, self).__init__(trap_densities, trap_lifetimes, well_notch_depth,
                                                   well_fill_alpha, well_fill_beta, well_fill_gamma)

    @property
    def object_tag(self):
        return 'p3_d' + str(self.trap_densities) + '_t' + str(self.trap_lifetimes) + '_a' + str(
            self.well_fill_alpha) + '_b' + \
               str(self.well_fill_beta) + '_g' + str(self.well_fill_gamma)


class ParallelDensityVary(ParallelParams):

    def __init__(self, trap_densities, trap_lifetimes, well_notch_depth=1e-9, well_fill_alpha=1.0, well_fill_beta=0.58,
                 well_fill_gamma=0.0):
        """The CTI model parameters used for parallel clocking, where the density of traps in each column of the list \
        varies as it would on a CCD.

        Parameters
        ----------
        trap_densities : [(float, float, float)]
            Each set of densities for each trap species.
        trap_lifetimes : (float, float, float)
            The trap lifetimes of the three trap species.
        well_notch_depth : float
            The CCD notch depth.
        well_fill_alpha : float
            The volume-filling coefficient (alpha) of how an electron cloud fills the volume of a pixel.
        well_fill_beta : float
            The volume-filling power (beta) of how an electron cloud fills the volume of a pixel.
        well_fill_gamma : float
            The volume-filling constant (gamma) of how an electron cloud fills the volume of a pixel.
        """

        super(ParallelDensityVary, self).__init__(trap_densities, trap_lifetimes, well_notch_depth,
                                                  well_fill_alpha, well_fill_beta, well_fill_gamma)

    @classmethod
    def poisson_densities(cls, trap_densities, trap_lifetimes, well_notch_depth, well_fill_alpha, well_fill_beta,
                          well_fill_gamma, shape, seed=0):
        """For a set of traps with a given set of densities (which are in traps per pixel), compute a new set of \
        trap densities by drawing new values for from a Poisson distribution.

        This requires us to first convert each trap density to the total number of traps in the column.

        This is used to model the random distribution of traps on a CCD, which changes the number of traps in each \
        column.
        
        Parameters
        -----------
        trap_densities : tuple
            The average trap density of each species.
        trap_lifetimes : (float, float, float)
            The trap lifetimes of the three trap species.
        well_notch_depth : float
            The CCD notch depth.
        well_fill_alpha : float
            The volume-filling coefficient (alpha) of how an electron cloud fills the volume of a pixel.
        well_fill_beta : float
            The volume-filling power (beta) of how an electron cloud fills the volume of a pixel.
        well_fill_gamma : float
            The volume-filling constant (gamma) of how an electron cloud fills the volume of a pixel.
        shape : (int, int)
            The shape of the image, so that the correct number of trap densities are computed.
        seed : int
            The seed of the Poisson random number generator.
        """
        np.random.seed(seed)
        total_traps = tuple(map(lambda density: density * shape[0], trap_densities))
        poisson_densities = [tuple(np.random.poisson(total_traps) / shape[0]) for row in range(shape[1])]
        return ParallelDensityVary(poisson_densities, trap_lifetimes, well_notch_depth, well_fill_alpha,
                                   well_fill_beta, well_fill_gamma)


class SerialParams(Params):

    def __init__(self, trap_densities, trap_lifetimes, well_notch_depth, well_fill_alpha, well_fill_beta,
                 well_fill_gamma):
        """Abstract base class of the cti model parameters for serial clocking. Parameters associated with the traps \
         are set via a child class.

        Params
        ----------
        trap_densities : tuple
            The trap density(ies) of the species
        trap_lifetimes : tuple
            The trap lifetime(s) of the species
        well_notch_depth : float
            The CCD notch depth
        well_fill_alpha : float
            The volume-filling coefficient (alpha) of how an electron cloud fills the volume of a pixel.
        well_fill_beta : float
            The volume-filling power (beta) of how an electron cloud fills the volume of a pixel.
        well_fill_gamma : float
            The volume-filling constant (gamma) of how an electron cloud fills the volume of a pixel.
        """

        super(SerialParams, self).__init__(trap_densities, trap_lifetimes, well_notch_depth, well_fill_alpha,
                                           well_fill_beta, well_fill_gamma)

    def generate_info(self):
        """Generate string containing information on the parallel parameters."""

        info = ''
        info += infoio.generate_class_info(cls=self, prefix='serial_', include_types=[int, float, tuple])
        info += '\n'
        return info

    def update_fits_header_info(self, ext_header):
        """Update a fits header to include the serial CTI model parameters.

        Params
        -----------
        ext_header : astropy.io.hdulist
            The opened header of the astropy fits header.
        """

        no_species = len(self.trap_densities)

        ext_header.set('cte_st1d', self.trap_densities[0], 'First trap species density (Serial)')
        ext_header.set('cte_st1t', self.trap_lifetimes[0], 'First trap species lifetime (Serial)')

        if no_species > 1:
            ext_header.set('cte_st2d', self.trap_densities[1], 'Second trap species density (Serial)')
            ext_header.set('cte_st2t', self.trap_lifetimes[1], 'Second trap species lifetime (Serial)')

        if no_species > 2:
            ext_header.set('cte_st3d', self.trap_densities[2], 'Third trap species density (Serial)')
            ext_header.set('cte_st3t', self.trap_lifetimes[2], 'Third trap species lifetime (Serial)')

        ext_header.set('cte_swln', self.well_notch_depth, 'CCD Well notch depth (Serial)')
        ext_header.set('cte_swlp', self.well_fill_beta, 'CCD Well filling power (Serial)')

        return ext_header


class SerialOneSpecies(SerialParams):

    def __init__(self, trap_densities=(0.01,), trap_lifetimes=(0.8,), well_notch_depth=1e-9, well_fill_alpha=1.0,
                 well_fill_beta=0.58, well_fill_gamma=0.0):
        """The CTI model parameters used for serial clocking, using one species of trap.

        Params
        ----------
        trap_densities : (float,)
            The trap density of the species
        trap_lifetimes : (float,)
            The trap lifetimes of the species
        well_notch_depth : float
            The CCD notch depth
        well_fill_alpha : float
            The volume-filling coefficient (alpha) of how an electron cloud fills the volume of a pixel.
        well_fill_beta : float
            The volume-filling power (beta) of how an electron cloud fills the volume of a pixel.
        well_fill_gamma : float
            The volume-filling constant (gamma) of how an electron cloud fills the volume of a pixel.

        Examples
        --------
        serial_one_species = SerialOneSpecies(trap_densities=(0.1,), trap_lifetimes=1.0, well_notch_depth=0.01,
                                              well_fill_beta=0.8)

        """

        super(SerialOneSpecies, self).__init__(trap_densities, trap_lifetimes, well_notch_depth, well_fill_alpha,
                                               well_fill_beta, well_fill_gamma)

    @property
    def object_tag(self):
        return 's1_d' + str(self.trap_densities) + '_t' + str(self.trap_lifetimes) + '_a' + str(self.well_fill_alpha) + \
               '_b' + str(self.well_fill_beta) + '_g' + str(self.well_fill_gamma)


class SerialTwoSpecies(SerialParams):

    def __init__(self, trap_densities=(0.01, 0.03), trap_lifetimes=(0.8, 3.5), well_notch_depth=1e-9,
                 well_fill_alpha=1.0, well_fill_beta=0.65, well_fill_gamma=0.0):
        """The CTI model parameters used for serial clocking, using two species of traps.

        Params
        ----------
        trap_densities : (float, float)
            The trap density of the two trap species
        trap_lifetimes : (float, float)
            The trap lifetimes of the two trap species
        well_notch_depth : float
            The CCD notch depth
        well_fill_alpha : float
            The volume-filling coefficient (alpha) of how an electron cloud fills the volume of a pixel.
        well_fill_beta : float
            The volume-filling power (beta) of how an electron cloud fills the volume of a pixel.
        well_fill_gamma : float
            The volume-filling constant (gamma) of how an electron cloud fills the volume of a pixel.

        Examples
        --------
        serial_two_species = SerialTwoSpecies(trap_densities=(0.1, 0.2), trap_lifetimes=(1.0, 2.0),
                                              well_notch_depth=0.01, well_fill_beta=0.8)

        """

        super(SerialTwoSpecies, self).__init__(trap_densities, trap_lifetimes, well_notch_depth, well_fill_alpha,
                                               well_fill_beta, well_fill_gamma)

    @property
    def object_tag(self):
        return 'p3_d' + str(self.trap_densities) + '_t' + str(self.trap_lifetimes) + '_a' + str(
            self.well_fill_alpha) + '_b' + \
               str(self.well_fill_beta) + '_g' + str(self.well_fill_gamma)


class SerialThreeSpecies(SerialParams):

    def __init__(self, trap_densities=(0.01, 0.03, 0.9), trap_lifetimes=(0.8, 3.5, 20.0), well_notch_depth=1e-9,
                 well_fill_alpha=1.0, well_fill_beta=0.58, well_fill_gamma=0.0):
        """The CTI model parameters used for serial clocking, using three species of traps.

        Params
        ----------
        trap_densities : (float, float, float)
            The trap density of the three trap species
        trap_lifetimes : (float, float, float)
            The trap lifetimes of the three trap species
        well_notch_depth : float
            The CCD notch depth
        well_fill_alpha : float
            The volume-filling coefficient (alpha) of how an electron cloud fills the volume of a pixel.
        well_fill_beta : float
            The volume-filling power (beta) of how an electron cloud fills the volume of a pixel.
        well_fill_gamma : float
            The volume-filling constant (gamma) of how an electron cloud fills the volume of a pixel.

        Examples
        --------
        serial_three_species = SerialThreeSpecies(trap_densities=(0.1, 0.2, 0.3),, trap_lifetimes=(1.0, 2.0, 3.0),
                                                  well_notch_depth=0.01, well_fill_beta=0.8)

        """

        super(SerialThreeSpecies, self).__init__(trap_densities, trap_lifetimes, well_notch_depth, well_fill_alpha,
                                                 well_fill_beta, well_fill_gamma)

    @property
    def object_tag(self):
        return 'p3_d' + str(self.trap_densities) + '_t' + str(self.trap_lifetimes) + '_a' + str(
            self.well_fill_alpha) + '_b' + \
               str(self.well_fill_beta) + '_g' + str(self.well_fill_gamma)
