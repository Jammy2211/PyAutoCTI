from autocti.plot import plotters
from autocti.plot import plotters, ci_line_plots


@plotters.set_include_and_sub_plotter
@plotters.set_subplot_filename
def subplot_ci_imaging(ci_imaging, include=None, sub_plotter=None):
    """Plot the imaging data_type as a sub-plotters of all its quantites (e.g. the dataset, noise_map, PSF, Signal-to_noise map, \
     etc).

    Set *autolens.data_type.array.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    ci_imaging : data_type.ImagingData
        The imaging data_type, which includes the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    ignore_config : bool
        If *False*, the config file general.ini is used to determine whether the subpot is plotted. If *True*, the \
        config file is ignored.
    """

    number_subplots = 4

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    image(ci_imaging=ci_imaging, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    noise_map(ci_imaging=ci_imaging, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=3)

    ci_pre_cti(ci_imaging=ci_imaging, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=4)

    signal_to_noise_map(ci_imaging=ci_imaging, include=include, plotter=sub_plotter)

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()


def individual(
    ci_imaging,
    plot_image=False,
    plot_noise_map=False,
    plot_signal_to_noise_map=False,
    plot_ci_pre_cti=False,
    plot_cosmic_ray_map=False,
    include=None,
    plotter=None,
):
    """Plot each attribute of the imaging data_type as individual figures one by one (e.g. the dataset, noise_map, PSF, \
     Signal-to_noise map, etc).

    Set *autolens.data_type.array.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    ci_imaging : data_type.ImagingData
        The imaging data_type, which includes the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    """

    if plot_image:

        image(ci_imaging=ci_imaging, include=include, plotter=plotter)

    if plot_noise_map:

        noise_map(ci_imaging=ci_imaging, include=include, plotter=plotter)

    if plot_signal_to_noise_map:

        signal_to_noise_map(ci_imaging=ci_imaging, include=include, plotter=plotter)

    if plot_ci_pre_cti:

        ci_pre_cti(ci_imaging=ci_imaging, include=include, plotter=plotter)

    if plot_cosmic_ray_map:

        cosmic_ray_map(ci_imaging=ci_imaging, include=include, plotter=plotter)


@plotters.set_include_and_plotter
@plotters.set_labels
def image(ci_imaging, include=None, plotter=None):
    """Plot the observed data_type of the imaging data_type.

    Set *autolens.data_type.array.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
    include_origin : True
        If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    """

    plotter.plot_frame(
        frame=ci_imaging.image,
        include_origin=include.origin,
        include_parallel_overscan=include.parallel_overscan,
        include_serial_prescan=include.serial_prescan,
        include_serial_overscan=include.serial_overscan,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def noise_map(ci_imaging, include=None, plotter=None):
    """Plot the noise_map of the imaging data_type.

    Set *autolens.data_type.array.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
    include_origin : True
        If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
    """
    plotter.plot_frame(
        frame=ci_imaging.noise_map,
        include_origin=include.origin,
        include_parallel_overscan=include.parallel_overscan,
        include_serial_prescan=include.serial_prescan,
        include_serial_overscan=include.serial_overscan,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def signal_to_noise_map(ci_imaging, include=None, plotter=None):
    """Plot the signal-to-noise_map of the imaging data_type.

    Set *autolens.data_type.array.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data_type.ImagingData
        The imaging data_type, which includes the observed image, noise_map, PSF, signal-to-noise_map, etc.
    include_origin : True
        If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
    """
    plotter.plot_frame(
        frame=ci_imaging.signal_to_noise_map,
        include_origin=include.origin,
        include_parallel_overscan=include.parallel_overscan,
        include_serial_prescan=include.serial_prescan,
        include_serial_overscan=include.serial_overscan,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def ci_pre_cti(ci_imaging, include=None, plotter=None):
    """Plot the observed ci_pre_cti of the ccd simulator.

    Set *autocti.simulator.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    ci_pre_cti : CIFrame
        The ci_pre_cti of the dataset.
    """
    plotter.plot_frame(
        frame=ci_imaging.ci_pre_cti,
        include_origin=include.origin,
        include_parallel_overscan=include.parallel_overscan,
        include_serial_prescan=include.serial_prescan,
        include_serial_overscan=include.serial_overscan,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def cosmic_ray_map(ci_imaging, include=None, plotter=None):
    """Plot the observed ci_pre_cti of the ccd simulator.

    Set *autocti.simulator.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    ci_pre_cti : CIFrame
        The ci_pre_cti of the dataset.
    """
    plotter.plot_frame(
        frame=ci_imaging.cosmic_ray_map,
        include_origin=include.origin,
        include_parallel_overscan=include.parallel_overscan,
        include_serial_prescan=include.serial_prescan,
        include_serial_overscan=include.serial_overscan,
    )


@plotters.set_include_and_sub_plotter
@plotters.set_subplot_filename
def subplot_ci_lines(ci_imaging, line_region, include=None, sub_plotter=None):
    """Plot the ci simulator as a sub-plotters of all its quantites (e.g. the dataset, noise_map, PSF, Signal-to_noise map, \
     etc).

    Set *autolens.simulator.arrays.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    ci_imaging : simulator.CCDData
        The ci simulator, which includes the observed dataset, noise_map, PSF, signal-to-noise_map, etc.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or simulator.arrays.grid_lines.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    ignore_config : bool
        If *False*, the config file general.ini is used to determine whether the subpot is plotted. If *True*, the \
        config file is ignored.
    """

    number_subplots = 4

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    image_line(
        ci_imaging=ci_imaging,
        line_region=line_region,
        include=include,
        plotter=sub_plotter,
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    noise_map_line(
        ci_imaging=ci_imaging,
        line_region=line_region,
        include=include,
        plotter=sub_plotter,
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=3)

    ci_pre_cti_line(
        ci_imaging=ci_imaging,
        line_region=line_region,
        include=include,
        plotter=sub_plotter,
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=4)

    signal_to_noise_map_line(
        ci_imaging=ci_imaging,
        line_region=line_region,
        include=include,
        plotter=sub_plotter,
    )

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()


def individual_ci_lines(
    ci_imaging,
    line_region,
    plot_image=False,
    plot_noise_map=False,
    plot_ci_pre_cti=False,
    plot_signal_to_noise_map=False,
    include=None,
    plotter=None,
):
    """Plot each attribute of the ci simulator as individual figures one by one (e.g. the dataset, noise_map, PSF, \
     Signal-to_noise map, etc).

    Set *autolens.simulator.arrays.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    ci_imaging : simulator.CCDData
        The ci simulator, which includes the observed dataset, noise_map, PSF, signal-to-noise_map, etc.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    """

    if plot_image:
        image_line(
            ci_imaging=ci_imaging,
            line_region=line_region,
            include=include,
            plotter=plotter,
        )

    if plot_noise_map:
        noise_map_line(
            ci_imaging=ci_imaging,
            line_region=line_region,
            include=include,
            plotter=plotter,
        )

    if plot_ci_pre_cti:
        ci_pre_cti_line(
            ci_imaging=ci_imaging,
            line_region=line_region,
            include=include,
            plotter=plotter,
        )

    if plot_signal_to_noise_map:
        signal_to_noise_map_line(
            ci_imaging=ci_imaging,
            line_region=line_region,
            include=include,
            plotter=plotter,
        )


@plotters.set_include_and_plotter
@plotters.set_labels
def image_line(ci_imaging, line_region, include=None, plotter=None):
    """Plot the observed image of the ccd simulator.

    Set *autocti.simulator.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    ci_imaging : CIFrame
        The image of the dataset.
    """
    ci_line_plots.plot_line_from_ci_frame(
        ci_frame=ci_imaging.image,
        line_region=line_region,
        include=include,
        plotter=plotter,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def noise_map_line(ci_imaging, line_region, include=None, plotter=None):
    """Plot the observed noise_map of the ccd simulator.

    Set *autocti.simulator.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    ci_imaging : CIFrame
        The noise_map of the dataset.
    """
    ci_line_plots.plot_line_from_ci_frame(
        ci_frame=ci_imaging.noise_map,
        line_region=line_region,
        include=include,
        plotter=plotter,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def ci_pre_cti_line(ci_imaging, line_region, include=None, plotter=None):
    """Plot the observed ci_pre_cti of the ccd simulator.

    Set *autocti.simulator.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    ci_imaging : CIFrame
        The ci_pre_cti of the dataset.
    """
    ci_line_plots.plot_line_from_ci_frame(
        ci_frame=ci_imaging.ci_pre_cti,
        line_region=line_region,
        include=include,
        plotter=plotter,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def signal_to_noise_map_line(ci_imaging, line_region, include=None, plotter=None):
    """Plot the observed signal_to_noise_map of the ccd simulator.

    Set *autocti.simulator.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    ci_imaging : CIFrame
        The signal_to_noise_map of the dataset.
    """
    ci_line_plots.plot_line_from_ci_frame(
        ci_frame=ci_imaging.signal_to_noise_map,
        line_region=line_region,
        include=include,
        plotter=plotter,
    )
