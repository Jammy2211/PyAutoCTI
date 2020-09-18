from autoarray.plot import plotters
from autocti.plot import ci_line_plots


@plotters.set_include_and_sub_plotter
@plotters.set_labels
def subplot_ci_fit(fit, include=None, sub_plotter=None):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    xyticksize
    ysize
    xsize
    titlesize
    cb_tick_labels
    cb_tick_values
    cb_pad
    cb_fraction
    cb_ticksize
    linscale
    linthresh
    norm_max
    norm_min
    norm
    cmap
    aspect
    figsize
    fit : autocti.lens.fitting.Fitter
        Class containing fit between the model datas_ and observed lens datas_ (including residual_map, chi_squared_map etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
    output_filename : str
        The name of the file that is output, if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    number_subplots = 9

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    image(fit=fit, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    noise_map(fit=fit, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=3)

    signal_to_noise_map(fit=fit, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=4)

    ci_pre_cti(fit=fit, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=5)

    ci_post_cti(fit=fit, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=7)

    residual_map(fit=fit, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=8)

    chi_squared_map(fit=fit, include=include, plotter=sub_plotter)

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()


@plotters.set_include_and_sub_plotter
@plotters.set_labels
def subplot_residual_maps(fits, include=None, sub_plotter=None):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customied.

    """

    number_subplots = len(fits)

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    for index, fit in enumerate(fits):

        sub_plotter.setup_subplot(
            number_subplots=number_subplots, subplot_index=index + 1
        )

        residual_map(fit=fit, include=include, plotter=sub_plotter)

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()


@plotters.set_include_and_sub_plotter
@plotters.set_labels
def subplot_normalized_residual_maps(fits, include=None, sub_plotter=None):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customied.

    """

    number_subplots = len(fits)

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    for index, fit in enumerate(fits):

        sub_plotter.setup_subplot(
            number_subplots=number_subplots, subplot_index=index + 1
        )

        normalized_residual_map(fit=fit, include=include, plotter=sub_plotter)

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()


@plotters.set_include_and_sub_plotter
@plotters.set_labels
def subplot_chi_squared_maps(fits, include=None, sub_plotter=None):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    """

    number_subplots = len(fits)

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    for index, fit in enumerate(fits):

        sub_plotter.setup_subplot(
            number_subplots=number_subplots, subplot_index=index + 1
        )

        chi_squared_map(fit=fit, include=include, plotter=sub_plotter)

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()


def individuals(
    fit,
    plot_image=False,
    plot_noise_map=False,
    plot_signal_to_noise_map=False,
    plot_ci_pre_cti=False,
    plot_ci_post_cti=False,
    plot_residual_map=False,
    plot_normalized_residual_map=False,
    plot_chi_squared_map=False,
    include=None,
    plotter=None,
):

    if plot_image:
        image(fit=fit, include=include, plotter=plotter)

    if plot_noise_map:
        noise_map(fit=fit, include=include, plotter=plotter)

    if plot_signal_to_noise_map:
        signal_to_noise_map(fit=fit, include=include, plotter=plotter)

    if plot_ci_pre_cti:
        ci_pre_cti(fit=fit, include=include, plotter=plotter)

    if plot_ci_post_cti:
        ci_post_cti(fit=fit, include=include, plotter=plotter)

    if plot_residual_map:
        residual_map(fit=fit, include=include, plotter=plotter)

    if plot_normalized_residual_map:
        normalized_residual_map(fit=fit, include=include, plotter=plotter)

    if plot_chi_squared_map:
        chi_squared_map(fit=fit, include=include, plotter=plotter)


@plotters.set_include_and_sub_plotter
@plotters.set_labels
def subplot_fit_lines(fit, line_region, include=None, sub_plotter=None):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    """

    number_subplots = 9

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    image_line(fit=fit, line_region=line_region, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    noise_map_line(
        fit=fit, line_region=line_region, include=include, plotter=sub_plotter
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=3)

    signal_to_noise_map_line(
        fit=fit, line_region=line_region, include=include, plotter=sub_plotter
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=4)

    ci_pre_cti_line(
        fit=fit, line_region=line_region, include=include, plotter=sub_plotter
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=5)

    ci_post_cti_line(
        fit=fit, line_region=line_region, include=include, plotter=sub_plotter
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=7)

    residual_map_line(
        fit=fit, line_region=line_region, include=include, plotter=sub_plotter
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=8)

    chi_squared_map_line(
        fit=fit, line_region=line_region, include=include, plotter=sub_plotter
    )

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()


@plotters.set_include_and_sub_plotter
@plotters.set_labels
def subplot_residual_map_lines(fits, line_region, include=None, sub_plotter=None):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.
    """

    number_subplots = len(fits)

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    for index, fit in enumerate(fits):
        sub_plotter.setup_subplot(
            number_subplots=number_subplots, subplot_index=index + 1
        )

        residual_map_line(
            fit=fit, line_region=line_region, include=include, plotter=sub_plotter
        )

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()


@plotters.set_include_and_sub_plotter
@plotters.set_labels
def subplot_normalized_residual_map_lines(
    fits, line_region, include=None, sub_plotter=None
):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.
    """

    number_subplots = len(fits)

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    for index, fit in enumerate(fits):
        sub_plotter.setup_subplot(
            number_subplots=number_subplots, subplot_index=index + 1
        )

        normalized_residual_map_line(
            fit=fit, line_region=line_region, include=include, plotter=sub_plotter
        )

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()


@plotters.set_include_and_sub_plotter
@plotters.set_labels
def subplot_chi_squared_map_lines(fits, line_region, include=None, sub_plotter=None):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.
    """

    number_subplots = len(fits)

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    for index, fit in enumerate(fits):

        sub_plotter.setup_subplot(
            number_subplots=number_subplots, subplot_index=index + 1
        )

        chi_squared_map_line(
            fit=fit, line_region=line_region, include=include, plotter=sub_plotter
        )

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()


def individuals_lines(
    fit,
    line_region,
    plot_image=False,
    plot_noise_map=False,
    plot_signal_to_noise_map=False,
    plot_ci_pre_cti=False,
    plot_ci_post_cti=False,
    plot_residual_map=False,
    plot_normalized_residual_map=False,
    plot_chi_squared_map=False,
    include=None,
    plotter=None,
):

    if plot_image:
        image_line(fit=fit, line_region=line_region, include=include, plotter=plotter)

    if plot_noise_map:
        noise_map_line(
            fit=fit, line_region=line_region, include=include, plotter=plotter
        )

    if plot_signal_to_noise_map:
        signal_to_noise_map_line(
            fit=fit, line_region=line_region, include=include, plotter=plotter
        )

    if plot_ci_pre_cti:
        ci_pre_cti_line(
            fit=fit, line_region=line_region, include=include, plotter=plotter
        )

    if plot_ci_post_cti:
        ci_post_cti_line(
            fit=fit, line_region=line_region, include=include, plotter=plotter
        )

    if plot_residual_map:
        residual_map_line(
            fit=fit, line_region=line_region, include=include, plotter=plotter
        )

    if plot_normalized_residual_map:
        normalized_residual_map_line(
            fit=fit, line_region=line_region, include=include, plotter=plotter
        )

    if plot_chi_squared_map:
        chi_squared_map_line(
            fit=fit, line_region=line_region, include=include, plotter=plotter
        )


@plotters.set_include_and_plotter
@plotters.set_labels
def image(fit, include=None, plotter=None):
    """Plot the observed image of the ccd simulator.

    Set *autocti.simulator.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : CIFrame
        The image of the dataset.
    """

    plotter.plot_frame(
        frame=fit.image,
        include_origin=include.origin,
        include_parallel_overscan=include.parallel_overscan,
        include_serial_prescan=include.serial_prescan,
        include_serial_overscan=include.serial_overscan,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def noise_map(fit, include=None, plotter=None):
    """Plot the observed noise_map of the ccd simulator.

    Set *autocti.simulator.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    noise_map : CIFrame
        The noise_map of the dataset.
    """

    plotter.plot_frame(
        frame=fit.noise_map,
        include_origin=include.origin,
        include_parallel_overscan=include.parallel_overscan,
        include_serial_prescan=include.serial_prescan,
        include_serial_overscan=include.serial_overscan,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def signal_to_noise_map(fit, include=None, plotter=None):
    """Plot the observed signal_to_noise_map of the ccd simulator.

    Set *autocti.simulator.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    signal_to_noise_map : CIFrame
        The signal_to_noise_map of the dataset.
    """

    plotter.plot_frame(
        frame=fit.signal_to_noise_map,
        include_origin=include.origin,
        include_parallel_overscan=include.parallel_overscan,
        include_serial_prescan=include.serial_prescan,
        include_serial_overscan=include.serial_overscan,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def ci_pre_cti(fit, include=None, plotter=None):
    """Plot the observed ci_pre_cti of the ccd simulator.

    Set *autocti.simulator.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    ci_pre_cti : CIFrame
        The ci_pre_cti of the dataset.
    """

    plotter.plot_frame(
        frame=fit.ci_pre_cti,
        include_origin=include.origin,
        include_parallel_overscan=include.parallel_overscan,
        include_serial_prescan=include.serial_prescan,
        include_serial_overscan=include.serial_overscan,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def ci_post_cti(fit, include=None, plotter=None):
    """Plot the observed ci_post_cti of the ccd simulator.

    Set *autocti.simulator.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    ci_post_cti : CIFrame
        The ci_post_cti of the dataset.
    """

    plotter.plot_frame(
        frame=fit.ci_post_cti,
        include_origin=include.origin,
        include_parallel_overscan=include.parallel_overscan,
        include_serial_prescan=include.serial_prescan,
        include_serial_overscan=include.serial_overscan,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def residual_map(fit, include=None, plotter=None):
    """Plot the observed residual_map of the ccd simulator.

    Set *autocti.simulator.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    residual_map : CIFrame
        The residual_map of the dataset.
    """

    plotter.plot_frame(
        frame=fit.residual_map,
        include_origin=include.origin,
        include_parallel_overscan=include.parallel_overscan,
        include_serial_prescan=include.serial_prescan,
        include_serial_overscan=include.serial_overscan,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def normalized_residual_map(fit, include=None, plotter=None):
    """Plot the observed normalized_residual_map of the ccd simulator.

    Set *autocti.simulator.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    normalized_residual_map : CIFrame
        The normalized_residual_map of the dataset.
    """

    plotter.plot_frame(
        frame=fit.normalized_residual_map,
        include_origin=include.origin,
        include_parallel_overscan=include.parallel_overscan,
        include_serial_prescan=include.serial_prescan,
        include_serial_overscan=include.serial_overscan,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def chi_squared_map(fit, include=None, plotter=None):
    """Plot the observed chi_squared_map of the ccd simulator.

    Set *autocti.simulator.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    chi_squared_map : CIFrame
        The chi_squared_map of the dataset.
    """

    plotter.plot_frame(
        frame=fit.chi_squared_map,
        include_origin=include.origin,
        include_parallel_overscan=include.parallel_overscan,
        include_serial_prescan=include.serial_prescan,
        include_serial_overscan=include.serial_overscan,
    )


@plotters.set_include_and_sub_plotter
@plotters.set_labels
def noise_scaling_maps(fit, include=None, sub_plotter=None):
    """Plot the observed chi_squared_map of the ccd simulator.

    Set *autocti.simulator.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    chi_squared_map : CIFrame
        The chi_squared_map of the dataset.
    """

    number_subplots = len(fit.noise_scaling_maps)

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    for index in range(len(fit.noise_scaling_maps)):

        sub_plotter.setup_subplot(
            number_subplots=number_subplots, subplot_index=index + 1
        )

        sub_plotter.plot_frame(
            frame=fit.noise_scaling_maps[index],
            include_origin=include.origin,
            include_parallel_overscan=include.parallel_overscan,
            include_serial_prescan=include.serial_prescan,
            include_serial_overscan=include.serial_overscan,
        )

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()


@plotters.set_include_and_plotter
@plotters.set_labels
def image_line(fit, line_region, include=None, plotter=None):
    """Plot the observed image of the ccd simulator.

    Set *autocti.simulator.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : CIFrame
        The image of the dataset.
    """
    ci_line_plots.plot_line_from_ci_frame(
        ci_frame=fit.image, line_region=line_region, include=include, plotter=plotter
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def noise_map_line(fit, line_region, include=None, plotter=None):
    """Plot the observed noise_map of the ccd simulator.

    Set *autocti.simulator.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    noise_map : CIFrame
        The noise_map of the dataset.
    """
    ci_line_plots.plot_line_from_ci_frame(
        ci_frame=fit.noise_map,
        line_region=line_region,
        include=include,
        plotter=plotter,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def signal_to_noise_map_line(fit, line_region, include=None, plotter=None):
    """Plot the observed signal_to_noise_map of the ccd simulator.

    Set *autocti.simulator.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    signal_to_noise_map : CIFrame
        The signal_to_noise_map of the dataset.
    """
    ci_line_plots.plot_line_from_ci_frame(
        ci_frame=fit.signal_to_noise_map,
        line_region=line_region,
        include=include,
        plotter=plotter,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def ci_pre_cti_line(fit, line_region, include=None, plotter=None):
    """Plot the observed ci_pre_cti of the ccd simulator.

    Set *autocti.simulator.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    ci_pre_cti : CIFrame
        The ci_pre_cti of the dataset.
    """
    ci_line_plots.plot_line_from_ci_frame(
        ci_frame=fit.ci_pre_cti,
        line_region=line_region,
        include=include,
        plotter=plotter,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def ci_post_cti_line(fit, line_region, include=None, plotter=None):
    """Plot the observed ci_post_cti of the ccd simulator.

    Set *autocti.simulator.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    ci_post_cti : CIFrame
        The ci_post_cti of the dataset.
    """
    ci_line_plots.plot_line_from_ci_frame(
        ci_frame=fit.ci_post_cti,
        line_region=line_region,
        include=include,
        plotter=plotter,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def residual_map_line(fit, line_region, include=None, plotter=None):
    """Plot the observed residual_map of the ccd simulator.

    Set *autocti.simulator.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    residual_map : CIFrame
        The residual_map of the dataset.
    """
    ci_line_plots.plot_line_from_ci_frame(
        ci_frame=fit.residual_map,
        line_region=line_region,
        include=include,
        plotter=plotter,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def normalized_residual_map_line(fit, line_region, include=None, plotter=None):
    """Plot the observed normalized_residual_map of the ccd simulator.

    Set *autocti.simulator.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    normalized_residual_map : CIFrame
        The normalized_residual_map of the dataset.
    """
    ci_line_plots.plot_line_from_ci_frame(
        ci_frame=fit.normalized_residual_map,
        line_region=line_region,
        include=include,
        plotter=plotter,
    )


@plotters.set_include_and_plotter
@plotters.set_labels
def chi_squared_map_line(fit, line_region, include=None, plotter=None):
    """Plot the observed chi_squared_map of the ccd simulator.

    Set *autocti.simulator.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    chi_squared_map : CIFrame
        The chi_squared_map of the dataset.
    """
    ci_line_plots.plot_line_from_ci_frame(
        ci_frame=fit.chi_squared_map,
        line_region=line_region,
        include=include,
        plotter=plotter,
    )
