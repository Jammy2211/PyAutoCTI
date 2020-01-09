from matplotlib import pyplot as plt
from autoarray.plotters import array_plotters
from autocti.plotters import ci_line_plotters
from autocti.plotters import ci_plotter_util


def plot_image(
    fit,
    mask=None,
    as_subplot=False,
    figsize=(7, 7),
    aspect="equal",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Fit Image",
    titlesize=16,
    xsize=16,
    ysize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="fit_image",
):
    """Plot the observed image of the ccd simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : CIFrame
        The image of the dataset.
    """

    array_plotters.plot_array(
        array=fit.image,
        mask=mask,
        as_subplot=as_subplot,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        title=title,
        titlesize=titlesize,
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_noise_map(
    fit,
    mask=None,
    as_subplot=False,
    figsize=(7, 7),
    aspect="equal",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Fit Noise-Map",
    titlesize=16,
    xsize=16,
    ysize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="fit_noise_map",
):
    """Plot the observed noise_map of the ccd simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    noise_map : CIFrame
        The noise_map of the dataset.
    """

    array_plotters.plot_array(
        array=fit.noise_map,
        mask=mask,
        as_subplot=as_subplot,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        title=title,
        titlesize=titlesize,
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_signal_to_noise_map(
    fit,
    mask=None,
    as_subplot=False,
    figsize=(7, 7),
    aspect="equal",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Fit Signal-to-Noise Map",
    titlesize=16,
    xsize=16,
    ysize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="fit_signal_to_noise_map",
):
    """Plot the observed signal_to_noise_map of the ccd simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    signal_to_noise_map : CIFrame
        The signal_to_noise_map of the dataset.
    """

    array_plotters.plot_array(
        array=fit.signal_to_noise_map,
        mask=mask,
        as_subplot=as_subplot,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        title=title,
        titlesize=titlesize,
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_ci_pre_cti(
    fit,
    mask=None,
    as_subplot=False,
    figsize=(7, 7),
    aspect="equal",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Fit Pre-CTI CI Image",
    titlesize=16,
    xsize=16,
    ysize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="fit_ci_pre_cti",
):
    """Plot the observed ci_pre_cti of the ccd simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    ci_pre_cti : CIFrame
        The ci_pre_cti of the dataset.
    """

    array_plotters.plot_array(
        array=fit.ci_pre_cti,
        mask=mask,
        as_subplot=as_subplot,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        title=title,
        titlesize=titlesize,
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_ci_post_cti(
    fit,
    mask=None,
    as_subplot=False,
    figsize=(7, 7),
    aspect="equal",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Fit Post-CTI CI Image",
    titlesize=16,
    xsize=16,
    ysize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="fit_ci_post_cti",
):
    """Plot the observed ci_post_cti of the ccd simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    ci_post_cti : CIFrame
        The ci_post_cti of the dataset.
    """

    array_plotters.plot_array(
        array=fit.ci_post_cti,
        mask=mask,
        as_subplot=as_subplot,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        title=title,
        titlesize=titlesize,
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_residual_map(
    fit,
    mask=None,
    as_subplot=False,
    figsize=(7, 7),
    aspect="equal",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Fit Residual Map",
    titlesize=16,
    xsize=16,
    ysize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="fit_residual_map",
):
    """Plot the observed residual_map of the ccd simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    residual_map : CIFrame
        The residual_map of the dataset.
    """

    array_plotters.plot_array(
        array=fit.residual_map,
        mask=mask,
        as_subplot=as_subplot,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        title=title,
        titlesize=titlesize,
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_chi_squared_map(
    fit,
    mask=None,
    as_subplot=False,
    figsize=(7, 7),
    aspect="equal",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Fit Chi-Squared Map",
    titlesize=16,
    xsize=16,
    ysize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="fit_chi_squared_map",
):
    """Plot the observed chi_squared_map of the ccd simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    chi_squared_map : CIFrame
        The chi_squared_map of the dataset.
    """

    array_plotters.plot_array(
        array=fit.chi_squared_map,
        mask=mask,
        as_subplot=as_subplot,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        title=title,
        titlesize=titlesize,
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_noise_scaling_maps(
    fit_hyper,
    mask=None,
    figsize=(7, 7),
    aspect="equal",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Fit Chi-Squared Map",
    titlesize=16,
    xsize=16,
    ysize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="fit_noise_scaling_maps",
):
    """Plot the observed chi_squared_map of the ccd simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    chi_squared_map : CIFrame
        The chi_squared_map of the dataset.
    """

    rows, columns, figsize_tool = ci_plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=len(fit_hyper.noise_scaling_maps)
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    for index in range(len(fit_hyper.noise_scaling_maps)):

        plt.subplot(rows, columns, index + 1)

        array_plotters.plot_array(
            array=fit_hyper.noise_scaling_maps[index],
            mask=mask,
            as_subplot=True,
            figsize=figsize,
            aspect=aspect,
            cmap=cmap,
            norm=norm,
            norm_min=norm_min,
            norm_max=norm_max,
            linthresh=linthresh,
            linscale=linscale,
            cb_ticksize=cb_ticksize,
            cb_fraction=cb_fraction,
            cb_pad=cb_pad,
            cb_tick_values=cb_tick_values,
            cb_tick_labels=cb_tick_labels,
            title=title,
            titlesize=titlesize,
            xsize=xsize,
            ysize=ysize,
            xyticksize=xyticksize,
            output_path=output_path,
            output_format=output_format,
            output_filename=output_filename,
        )

    ci_plotter_util.output.to_figure(structure=None, is_sub_plotter=False)(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )

    plt.close()


def plot_image_line(
    fit,
    line_region,
    mask=None,
    as_subplot=False,
    figsize=(7, 7),
    title="Image",
    titlesize=16,
    xsize=16,
    ysize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="fit_image_line",
):
    """Plot the observed image of the ccd simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : CIFrame
        The image of the dataset.
    """
    ci_line_plotters.plot_line_from_ci_frame(
        array=fit.image,
        line_region=line_region,
        ci_frame=fit.ci_data_masked.ci_frame,
        mask=mask,
        as_subplot=as_subplot,
        figsize=figsize,
        title=title,
        titlesize=titlesize,
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_noise_map_line(
    fit,
    line_region,
    mask=None,
    as_subplot=False,
    figsize=(7, 7),
    title="Noise Map",
    titlesize=16,
    xsize=16,
    ysize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="fit_noise_map_line",
):
    """Plot the observed noise_map of the ccd simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    noise_map : CIFrame
        The noise_map of the dataset.
    """
    ci_line_plotters.plot_line_from_ci_frame(
        array=fit.noise_map,
        line_region=line_region,
        ci_frame=fit.ci_data_masked.ci_frame,
        mask=mask,
        as_subplot=as_subplot,
        figsize=figsize,
        title=title,
        titlesize=titlesize,
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_signal_to_noise_map_line(
    fit,
    line_region,
    mask=None,
    as_subplot=False,
    figsize=(7, 7),
    title="Signal-To-Noise Map",
    titlesize=16,
    xsize=16,
    ysize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="fit_signal_to_noise_map_line",
):
    """Plot the observed signal_to_noise_map of the ccd simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    signal_to_noise_map : CIFrame
        The signal_to_noise_map of the dataset.
    """
    ci_line_plotters.plot_line_from_ci_frame(
        array=fit.signal_to_noise_map,
        line_region=line_region,
        ci_frame=fit.ci_data_masked.ci_frame,
        mask=mask,
        as_subplot=as_subplot,
        figsize=figsize,
        title=title,
        titlesize=titlesize,
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_ci_pre_cti_line(
    fit,
    line_region,
    mask=None,
    as_subplot=False,
    figsize=(7, 7),
    title="CI Pre-CTI Image",
    titlesize=16,
    xsize=16,
    ysize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="fit_ci_pre_cti_line",
):
    """Plot the observed ci_pre_cti of the ccd simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    ci_pre_cti : CIFrame
        The ci_pre_cti of the dataset.
    """
    ci_line_plotters.plot_line_from_ci_frame(
        array=fit.ci_pre_cti,
        line_region=line_region,
        ci_frame=fit.ci_data_masked.ci_frame,
        mask=mask,
        as_subplot=as_subplot,
        figsize=figsize,
        title=title,
        titlesize=titlesize,
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_ci_post_cti_line(
    fit,
    line_region,
    mask=None,
    as_subplot=False,
    figsize=(7, 7),
    title="CI Post-CTI Image",
    titlesize=16,
    xsize=16,
    ysize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="fit_ci_post_cti_line",
):
    """Plot the observed ci_post_cti of the ccd simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    ci_post_cti : CIFrame
        The ci_post_cti of the dataset.
    """
    ci_line_plotters.plot_line_from_ci_frame(
        array=fit.ci_post_cti,
        line_region=line_region,
        ci_frame=fit.ci_data_masked.ci_frame,
        mask=mask,
        as_subplot=as_subplot,
        figsize=figsize,
        title=title,
        titlesize=titlesize,
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_residual_map_line(
    fit,
    line_region,
    mask=None,
    as_subplot=False,
    figsize=(7, 7),
    title="Residual Map",
    titlesize=16,
    xsize=16,
    ysize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="fit_residual_map_line",
):
    """Plot the observed residual_map of the ccd simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    residual_map : CIFrame
        The residual_map of the dataset.
    """
    ci_line_plotters.plot_line_from_ci_frame(
        array=fit.residual_map,
        line_region=line_region,
        ci_frame=fit.ci_data_masked.ci_frame,
        mask=mask,
        as_subplot=as_subplot,
        figsize=figsize,
        title=title,
        titlesize=titlesize,
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_chi_squared_map_line(
    fit,
    line_region,
    mask=None,
    as_subplot=False,
    figsize=(7, 7),
    title="Chi-Squared Map",
    titlesize=16,
    xsize=16,
    ysize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="fit_chi_squared_map_line",
):
    """Plot the observed chi_squared_map of the ccd simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    chi_squared_map : CIFrame
        The chi_squared_map of the dataset.
    """
    ci_line_plotters.plot_line_from_ci_frame(
        array=fit.chi_squared_map,
        line_region=line_region,
        ci_frame=fit.ci_data_masked.ci_frame,
        mask=mask,
        as_subplot=as_subplot,
        figsize=figsize,
        title=title,
        titlesize=titlesize,
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )
