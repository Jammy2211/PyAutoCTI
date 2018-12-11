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
File: python/SHE_ArCTIC/PyArCTIC.py

Created on: 02/13/18
Author: James Nightingale
"""

import numpy as np

from autocti.model import arctic_params


def add_parallel_cti_to_image(image, params, settings):
    """ Add parallel cti to an image using arctic, given the CTI settings and CTI model parameters.

    Parameters
    ----------
    image : ndarray
        The two-dimensional image passed to arctic for CTI addition, assuming the direction of parallel clocking is \
        upwards relative to a ndarray (e.g towards image[0, :]).
    params : ArcticParams
        The CTI parameters (trap density, lifetimes etc.). This may include both sets of model parameters in the \
        parallel and serial direction (e.g. ParallelParams and SerialParams).
    settings : ArcticSettings
        The settings that control arctic (e.g. ccd well_depth express option). This may include both settings in the \
        serial and serial direction (e.g. ParallelSettings and SerialSettings).

    Returns
    ----------
    Two-dimensional image after clocking and therefore with parallel CTI added.

    Notes
    ----------

    This routine adds parallel cti to an image assuming that clocking is upwards relative to a numpy array \
    (e.g towards image[0, :]). Thus, you must be certain the image is oriented in the appropriate direction.

    For Euclid quadrants, see the VIS_CTI_Image.CTIImage class, which handles all rotations and arctic calls
    automatically.

    Examples
    --------
    settings = ArcticSettings(neomode='NEO',parallel_settings=ParallelSettings(well_depth=84700, niter=1,
                                                                        express=5, n_levels=2000, readout_offset=0))
                                                                        
    model = ArcticParams(parallel_parameters=ParallelOneSpecies(trap_densities=(0.1,), trap_lifetimes=(1.0,)
                                                                well_notch_depth=0.01, well_fill_beta=0.8))
                                                                       
    image_post_clocking = PyArctic.add_parallel_cti_to_image(image=image_data, settings=settings, model=model)
    """

    unclock = False

    return call_arctic(image, unclock, params.parallel, settings.parallel)


def add_serial_cti_to_image(image, params, settings):
    """
    Add serial cti to an image using arctic, given the CTI settings and CTI model parameters.

    Parameters
    ----------
    image : ndarray
        The two-dimensional image passed to arctic for CTI addition, assuming the direction of serial clocking is \
         upwards relative to a ndarray (e.g towards image[0, :]).
    params : ArcticParams
        The CTI parameters (trap density, lifetimes etc.). This may include both sets of model parameters in the \
        serial and serial direction (e.g. ParallelParams and SerialParams)
    settings : ArcticSettings
        The settings that control arctic (e.g. ccd well_depth express option). This may include both settings in the \
        serial and serial direction (e.g. ParallelSettings and SerialSettings).

    Returns
    ----------
    Two-dimensional image, after clocking and therefore with serial CTI added.

    Notes
    ----------

    This routine adds serial cti to an image assuming that clocking is upwards relative to a numpy array \
    (e.g towards image[0, :]). Thus, you must be certain the image is oriented in the appropriate direction.

    For Euclid quadrants, see the VIS_CTI.CTITools.CTIImage class, which handles all rotations and arctic calls
    automatically.

    Examples
    --------
    settings = ArcticSettings(neomode='NEO',serial_settings=SerialSettings(well_depth=84700, niter=1,
                                                                        express=5, n_levels=2000, readout_offset=0))

    model = ArcticParams(serial_parameters=SerialOneSpecies(trap_densities=(0.1,), trap_lifetimes=(1.0,)
                                                            well_notch_depth=0.01, well_fill_beta=0.8))

    image_post_clocking = PyArctic.add_serial_cti_to_image(image=image_data, settings=settings, model=model)

    """

    unclock = False

    return call_arctic(image, unclock, params.serial, settings.serial)


def correct_parallel_cti_from_image(image, params, settings):
    """Correct parallel cti from an image, given the CTI settings and CTI model parameters.

    Parameters
    ----------
    image : ndarray
        The two-dimensional image passed to arctic for CTI correction, assuming the direction of parallel clocking is \
        upwards relative to a ndarray (e.g towards image[0, :]).
    params : ArcticParams
        The CTI parameters (trap density, lifetimes etc.). This may include both sets of model parameters in the \
        serial and serial direction (e.g. ParallelParams and SerialParams).
    settings : ArcticSettings
        The settings that control arctic (e.g. ccd well_depth express option). This may include both settings in the \
        serial and serial direction (e.g. ParallelSettings and SerialSettings).

    Returns
    ----------
    Two-dimensional image, after clocking and therefore with parallel CTI corrected.

    Notes
    ----------

    This routine corrects parallel cti to an image assuming that clocking is upwards relative to a numpy array \
    (e.g towards image[0, :]). Thus, you must be certain the image is oriented in the appropriate direction.

    For Euclid quadrants, see the VIS_CTI_Image.CTIImage class, which handles all rotations and arctic calls
    automatically.

    Examples
    --------
    settings = ArcticSettings(neomode='NEO',parallel_settings=ParallelSettings(well_depth=84700, niter=1,
                                                                            express=5, n_levels=2000, readout_offset=0))

    model = ArcticParams(parallel_parameters=ParallelOneSpecies(trap_densities=(0.1,), trap_lifetimes=(1.0,)
                                                                well_notch_depth=0.01, well_fill_beta=0.8))

    image_with_cti_corrected = PyArctic.correct_parallel_cti_from_image(image=image_data, settings=settings,
                                                                        model=model)
    """

    unclock = True

    return call_arctic(image, unclock, params.parallel, settings.parallel)


def correct_serial_cti_from_image(image, params, settings):
    """Correct serial cti from an image, given the CTI settings and CTI model parameters.

    Parameters
    ----------
    image : ndarray
        The two-dimensional image passed to arctic for CTI correction, assuming the direction of serial clocking is \
        upwards relative to a ndarray (e.g towards image[0, :]).
    settings : ArcticSettings
        The settings that control arctic (e.g. ccd well_depth express option). This may include both settings in the \
        serial and serial direction (e.g. ParallelSettings and SerialSettings).
    params : ArcticParams
        The CTI parameters (trap density, lifetimes etc.). This may include both sets of model parameters in the \
        serial and serial direction (e.g. ParallelParams and SerialParams).

    Returns
    ----------
    Two-dimensional image, after clocking and therefore with serial CTI corrected.

    Notes
    ----------

    This routine corrects serial cti to an image assuming that clocking is upwards relative to a numpy array \
    (e.g towards image[0, :]). Thus, you must be certain the image is oriented in the appropriate direction.

    For Euclid quadrants, see the VIS_CTI_Image.CTIImageclass, which handles all rotations and arctic calls
    automatically.

    Examples
    --------
    settings = ArcticSettings(neomode='NEO',serial_settings=SerialSettings(well_depth=84700, niter=1,
                                                                        express=5, n_levels=2000, readout_offset=0))
    model = ArcticParams(serial_parameters=SerialOneSpecies(trap_densities=(0.1,), trap_lifetimes=(1.0,)

                                                                       well_notch_depth=0.01, well_fill_beta=0.8))

    image_with_cti_corrected = PyArctic.correct_serial_cti_from_image(image=image_data, settings=settings, model=model)
    """

    unclock = True

    return call_arctic(image, unclock, params.serial, settings.serial)


def call_arctic(image_pre_clocking, unclock, params, settings):
    """
    Perform image clocking via an arctic call (via swig wrapping), either adding or correcting cti to an image in \
    either the parallel or serial direction

    Parameters
    ----------
    image_pre_clocking : ndarray
        The two-dimensional image passed to arctic for CTI correction, assuming the direction of clocking is \
        upwards relative to a ndarray (e.g towards image[0, :]).
    unclock : bool
        Determines whether arctic is correcting CTI (unclock=True) or adding CTI (unclock=False).
    settings : ArcticSettings
        The settings that control arctic (e.g. ccd well_depth express option). This is the settings in one specific \
        direction of clocking (e.g. ArcticSettings.ParallelSettings or ArcticSettings.SerialSettings)
    params : ArcticParams
        The CTI parameters (trap density, lifetimes etc.). These parameters are in one specific \
        direction of clocking (e.g. arctic_params.ParallelParams or arctic_params.SerialParams)

    Returns
    ----------
    image : ndarray
        Two-dimensional image which has had CTI added / corrected via arctic

    Examples
    --------
    settings = ArcticSettings(neomode='NEO',serial_settings=SerialSettings(well_depth=84700, niter=1,
                                                                        express=5, n_levels=2000, readout_offset=0))

    model = ArcticParams(serial_parameters=SerialOneSpecies(trap_densities=(0.1,), trap_lifetimes=(1.0,)
                                                                       well_notch_depth=0.01, well_fill_beta=0.8))

    image = call_arctic(image, unclock=True, settings.serial, model.serial)

    """

    # noinspection PyUnresolvedReferences,PyPep8Naming
    import pySHE_ArCTIC as arctic

    settings.unclock = unclock  # Include unclock in parameters to pass to different routines easily
    clock_routine = arctic.cte_image_neo()  # Setup instance of arctic charge clocking routine

    clock_params = clock_routine.parameters  # Initiate the parameters which the arctic clocking routine uses

    set_arctic_settings(clock_params=clock_params, settings=settings)

    if not isinstance(params, arctic_params.ParallelDensityVary):

        return clock_image(clock_routine=clock_routine, clock_params=clock_params,
                           image=image_pre_clocking, params=params)

    elif isinstance(params, arctic_params.ParallelDensityVary):

        return clock_image_variable_density(clock_routine=clock_routine, clock_params=clock_params,
                                            image=image_pre_clocking, params=params)


def clock_image(clock_routine, clock_params, image, params):
    """Clock the image using arctic."""

    set_arctic_image_dimensions(clock_routine=clock_routine, clock_params=clock_params,
                                dimensions=image.shape)

    set_arctic_params(clock_params=clock_params, no_species=len(params.trap_lifetimes),
                      trap_densities=params.trap_densities, trap_lifetimes=params.trap_lifetimes,
                      well_notch_depth=params.well_notch_depth, well_fill_alpha=params.well_fill_alpha,
                      well_fill_beta=params.well_fill_beta, well_fill_gamma=params.well_fill_gamma)

    image = image.astype(np.float64)  # Have to convert type to avoid c++ memory issues
    clock_routine.clock_charge(image)
    return image


def clock_image_variable_density(clock_routine, clock_params, image, params):
    """Clock the image via arctic, inputing one column at a time. This is done so that the Poisson density feature \
    can drawn a different density of traps for each column.

    Note that for serial CTI, the image will have already been rotated to the corrct orientation and that using this \
    feature on columns will give the correct CTI ci_pattern."""

    # The post clocking image is stored in a new array.
    image_post_clocking = np.zeros(image.shape)

    # Setup the density / column length for computing Poisson density values
    column_pre_clocking = np.zeros(shape=(image.shape[0], 1))

    # Setup the arctic image such that it knows to expect one column from every call
    set_arctic_image_dimensions(clock_routine=clock_routine, clock_params=clock_params,
                                dimensions=(image.shape[0], 1))

    for column_no in range(image.shape[1]):
        set_arctic_params(clock_params=clock_params, no_species=len(params.trap_lifetimes),
                          trap_densities=params.trap_densities[column_no],
                          trap_lifetimes=params.trap_lifetimes, well_notch_depth=params.well_notch_depth,
                          well_fill_alpha=params.well_fill_alpha, well_fill_beta=params.well_fill_beta,
                          well_fill_gamma=params.well_fill_gamma)

        column_pre_clocking[:, 0] = image[:, column_no]
        column_post_clocking = column_pre_clocking.astype(np.float64)  # Have to convert type to avoid c++ memory issues
        clock_routine.clock_charge(column_post_clocking)
        image_post_clocking[:, column_no] = column_post_clocking[:, 0]

    return image_post_clocking


def set_arctic_settings(clock_params, settings):
    """Set the settings for the arctic clocking routine"""

    clock_params.unclock = settings.unclock

    clock_params.well_depth = settings.well_depth
    clock_params.n_iterations = settings.niter
    clock_params.express = settings.express
    clock_params.n_levels = settings.n_levels
    clock_params.charge_injection = settings.charge_injection_mode
    clock_params.readout_offset = settings.readout_offset


def set_arctic_params(clock_params, no_species, trap_densities, trap_lifetimes, well_notch_depth,
                      well_fill_alpha, well_fill_beta, well_fill_gamma):
    """Set the clock_params for the arctic clocking routine."""
    if no_species == 1:

        clock_params.set_traps([trap_densities[0]], [trap_lifetimes[0]])

    elif no_species == 2:

        clock_params.set_traps([trap_densities[0], trap_densities[1]],
                               [trap_lifetimes[0], trap_lifetimes[1]])

    elif no_species == 3:

        clock_params.set_traps([trap_densities[0], trap_densities[1],
                                trap_densities[2]],
                               [trap_lifetimes[0], trap_lifetimes[1],
                                trap_lifetimes[2]])

    else:

        raise ArcticException('Cannot input cti clock_params with more than 3 species of traps')

    clock_params.well_notch_depth = well_notch_depth
    clock_params.well_fill_alpha = well_fill_alpha
    clock_params.well_fill_beta = well_fill_beta
    clock_params.well_fill_gamma = well_fill_gamma


def set_arctic_image_dimensions(clock_routine, clock_params, dimensions):
    clock_params.start_x = 0
    clock_params.start_y = 0

    clock_params.end_x = dimensions[1]
    clock_params.end_y = dimensions[0]

    clock_routine.setup(dimensions[1], dimensions[0])


class ArcticException(Exception):
    pass
