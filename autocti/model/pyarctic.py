import numpy as np

from autocti.model import arctic_params


def call_arctic(
    image, species, ccd, settings, correct_cti=False, use_poisson_densities=False
):

    if not use_poisson_densities:
        return call_arctic_constant_density(
            image=image,
            species=species,
            ccd=ccd,
            settings=settings,
            correct_cti=correct_cti,
        )
    elif use_poisson_densities:
        return call_arctic_parallel_poisson_density(
            image=image,
            species=species,
            ccd=ccd,
            settings=settings,
            correct_cti=correct_cti,
        )


def call_arctic_constant_density(image, species, ccd, settings, correct_cti):
    """
    Perform image clocking via an arctic call (via swig wrapping), either adding or correcting cti to an image in \
    either the parallel or serial direction

    Parameters
    ----------
    image : ndarray
        The two-dimensional image passed to arctic for CTI correction, assuming the direction of clocking is \
        upwards relative to a ndarray (e.g towards image[0, :]).
    correct_cti : bool
        Determines whether arctic is correcting CTI (unclock=True) or adding CTI (unclock=False).
    settings : ArcticSettings
        The settings that control arctic (e.g. ccd well_depth express option). This is the settings in one specific \
        direction of clocking (e.g. ArcticSettings.Settings or ArcticSettings.Settings)
    species: [arctic_params.Species]
    ccd: arctic_params.CCDVolume

    Returns
    ----------
    image : ndarray
        Two-dimensional image which has had CTI added / corrected via arctic

    Examples
    --------
    settings = ArcticSettings(neomode='NEO',serial_settings=Settings(well_depth=84700, niter=1,
                                                                        express=5, n_levels=2000, readout_offset=0))

    model = ArcticParams(serial_parameters=SerialOneSpecies(trap_densities=(0.1,), trap_lifetimes=(1.0,)
                                                                       well_notch_depth=0.01, well_fill_beta=0.8))

    image = call_arctic(image, unclock=True, settings.serial, model.serial)

    """

    # noinspection PyUnresolvedReferences,PyPep8Naming
    import pySHE_ArCTIC as arctic

    clock_routine = (
        arctic.cte_image_neo()
    )  # Setup instance of arctic charge clocking routine

    clock_params = (
        clock_routine.parameters
    )  # Initiate the parameters which the arctic clocking routine uses
    clock_params.unclock = (
        correct_cti
    )  # Include unclock in parameters to pass to different
    # routines easily

    set_arctic_settings(clock_params=clock_params, settings=settings)
    set_arctic_params(clock_params=clock_params, species=species, ccd=ccd)

    return clock_image(
        clock_routine=clock_routine, clock_params=clock_params, image=image
    )


def clock_image(clock_routine, clock_params, image):
    """Clock the image using arctic."""

    set_arctic_image_dimensions(
        clock_routine=clock_routine, clock_params=clock_params, dimensions=image.shape
    )

    image = image.astype(np.float64)  # Have to convert type to avoid c++ memory issues
    clock_routine.clock_charge(image)
    return image


def call_arctic_parallel_poisson_density(image, species, ccd, settings, correct_cti):
    # noinspection PyUnresolvedReferences,PyPep8Naming
    import pySHE_ArCTIC as arctic

    clock_routine = (
        arctic.cte_image_neo()
    )  # Setup instance of arctic charge clocking routine

    clock_params = (
        clock_routine.parameters
    )  # Initiate the parameters which the arctic clocking routine uses
    clock_params.unclock = (
        correct_cti
    )  # Include unclock in parameters to pass to different

    set_arctic_settings(clock_params=clock_params, settings=settings)

    return clock_image_variable_density(
        clock_routine=clock_routine,
        clock_params=clock_params,
        image=image,
        species=species,
        ccd=ccd,
    )


def clock_image_variable_density(clock_routine, clock_params, image, species, ccd):
    """Clock the image via arctic, inputting one column at a time. This is done so that the Poisson density feature \
    can drawn a different density of traps for each column.

    Note that for serial CTI, the image will have already been rotated to the corrct orientation and that using this \
    feature on columns will give the correct CTI ci_pattern."""

    # The post clocking image is stored in a new array.
    image_post_clocking = np.zeros(image.shape)

    # Setup the density / column length for computing Poisson density values
    column_pre_clocking = np.zeros(shape=(image.shape[0], 1))

    # Setup the arctic image such that it knows to expect one column from every call
    set_arctic_image_dimensions(
        clock_routine=clock_routine,
        clock_params=clock_params,
        dimensions=(image.shape[0], 1),
    )

    species_per_column = int(len(species) / image.shape[1])

    for column_no in range(image.shape[1]):

        set_arctic_params(
            clock_params=clock_params,
            species=species[
                column_no * species_per_column : (column_no + 1) * species_per_column
            ],
            ccd=ccd,
        )

        column_pre_clocking[:, 0] = image[:, column_no]
        column_post_clocking = column_pre_clocking.astype(
            np.float64
        )  # Have to convert type to avoid c++ memory issues
        clock_routine.clock_charge(column_post_clocking)
        image_post_clocking[:, column_no] = column_post_clocking[:, 0]

    return image_post_clocking


def set_arctic_settings(clock_params, settings):
    """Set the settings for the arctic clocking routine"""

    clock_params.well_depth = settings.well_depth
    clock_params.n_iterations = settings.niter
    clock_params.express = settings.express
    clock_params.n_levels = settings.n_levels
    clock_params.charge_injection = settings.charge_injection_mode
    clock_params.readout_offset = settings.readout_offset


def set_arctic_params(clock_params, species, ccd):
    """Set the clock_params for the arctic clocking routine."""
    clock_params.set_traps(
        [s.trap_density for s in species], [s.trap_lifetime for s in species]
    )

    clock_params.well_notch_depth = ccd.well_notch_depth
    clock_params.well_fill_alpha = ccd.well_fill_alpha
    clock_params.well_fill_beta = ccd.well_fill_beta
    clock_params.well_fill_gamma = ccd.well_fill_gamma


def set_arctic_image_dimensions(clock_routine, clock_params, dimensions):
    clock_params.start_x = 0
    clock_params.start_y = 0

    clock_params.end_x = dimensions[1]
    clock_params.end_y = dimensions[0]

    clock_routine.setup(dimensions[1], dimensions[0])
