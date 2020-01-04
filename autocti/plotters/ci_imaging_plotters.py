from autoarray import conf
import matplotlib

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
from matplotlib import pyplot as plt

from autocti.plotters import ci_line_plotters, ci_plotter_util
from autoarray.plotters import array_plotters
from autoarray.plots import imaging_plotters
from autoarray.util import plotter_util


def subplot(
    imaging,
    use_scaled_units=True,
    unit_conversion_factor=None,
    unit_label="scaled",
    figsize=None,
    aspect="square",
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
    titlesize=10,
    xlabelsize=10,
    ylabelsize=10,
    xyticksize=10,
    mask_pointsize=10,
    position_pointsize=30,
    grid_pointsize=1,
    output_path=None,
    output_filename="imaging",
    output_format="show",
):
    """Plot the imaging data_type as a sub-plotters of all its quantites (e.g. the dataset, noise_map-map, PSF, Signal-to_noise-map, \
     etc).

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    imaging : data_type.ImagingData
        The imaging data_type, which includes the observed data_type, noise_map-map, PSF, signal-to-noise_map-map, etc.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data_type.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    ignore_config : bool
        If *False*, the config file general.ini is used to determine whether the subpot is plotted. If *True*, the \
        config file is ignored.
    """

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=4
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    imaging_plotters.image(
        imaging=imaging,
        as_subplot=True,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
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
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        position_pointsize=position_pointsize,
        grid_pointsize=grid_pointsize,
        output_path=output_path,
        output_format=output_format,
    )

    plt.subplot(rows, columns, 2)

    imaging_plotters.noise_map(
        imaging=imaging,
        as_subplot=True,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
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
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        output_path=output_path,
        output_format=output_format,
    )

    plt.subplot(rows, columns, 3)

    ci_pre_cti(
        imaging=imaging,
        as_subplot=True,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
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
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
    )

    plt.subplot(rows, columns, 4)

    imaging_plotters.signal_to_noise_map(
        imaging=imaging,
        as_subplot=True,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
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
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        output_path=output_path,
        output_format=output_format,
    )

    plotter_util.output_subplot_array(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )

    plt.close()


def individual(
    imaging,
    unit_label="scaled",
    unit_conversion_factor=None,
    plot_image=False,
    plot_noise_map=False,
    plot_signal_to_noise_map=False,
    plot_ci_pre_cti=False,
    plot_cosmic_ray_image=False,
    output_path=None,
    output_format="png",
):
    """Plot each attribute of the imaging data_type as individual figures one by one (e.g. the dataset, noise_map-map, PSF, \
     Signal-to_noise-map, etc).

    Set *autolens.data_type.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    imaging : data_type.ImagingData
        The imaging data_type, which includes the observed data_type, noise_map-map, PSF, signal-to-noise_map-map, etc.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    """

    imaging_plotters.individual(
        imaging=imaging,
        plot_image=plot_image,
        plot_noise_map=plot_noise_map,
        plot_signal_to_noise_map=plot_signal_to_noise_map,
        output_path=output_path,
        output_format=output_format,
    )

    if plot_ci_pre_cti:

        ci_pre_cti(
            imaging=imaging,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            output_path=output_path,
            output_format=output_format,
        )

    # if plot_cosmic_ray_image:
    #
    #     cosmic_ray_image(
    #         imaging=imaging,
    #         unit_label=unit_label,
    #         unit_conversion_factor=unit_conversion_factor,
    #         output_path=output_path,
    #         output_format=output_format,
    #     )


def ci_pre_cti(
    imaging,
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
    title="",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="ci_pre_cti",
):
    """Plot the observed ci_pre_cti of the ccd simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    ci_pre_cti : CIFrame
        The ci_pre_cti of the dataset.
    """
    array_plotters.plot_array(
        array=imaging.ci_pre_cti,
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
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_ci_line_subplot(
    ci_data,
    line_region,
    mask=None,
    figsize=None,
    title="Image",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="ci_data_line",
):
    """Plot the ci simulator as a sub-plotters of all its quantites (e.g. the dataset, noise_map-map, PSF, Signal-to_noise-map, \
     etc).

    Set *autolens.simulator.arrays.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    ci_data : simulator.CCDData
        The ci simulator, which includes the observed simulator, noise_map-map, PSF, signal-to-noise_map-map, etc.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or simulator.arrays.grid_lines.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    ignore_config : bool
        If *False*, the config file general.ini is used to determine whether the subpot is plotted. If *True*, the \
        config file is ignored.
    """

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=4
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    image_line(
        ci_data=ci_data,
        line_region=line_region,
        mask=mask,
        as_subplot=True,
        figsize=figsize,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )

    plt.subplot(rows, columns, 2)

    noise_map_line(
        ci_data=ci_data,
        line_region=line_region,
        mask=mask,
        as_subplot=True,
        figsize=figsize,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )

    plt.subplot(rows, columns, 3)

    ci_pre_cti_line(
        ci_data=ci_data,
        line_region=line_region,
        mask=mask,
        as_subplot=True,
        figsize=figsize,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )

    plt.subplot(rows, columns, 4)

    signal_to_noise_map_line(
        ci_data=ci_data,
        line_region=line_region,
        mask=mask,
        as_subplot=True,
        figsize=figsize,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )

    ci_plotter_util.output_subplot_array(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )

    plt.close()


def plot_ci_data_line_individual(
    ci_data,
    line_region,
    mask=None,
    plot_image=False,
    plot_noise_map=False,
    plot_ci_pre_cti=False,
    plot_signal_to_noise_map=False,
    output_path=None,
    output_format="png",
):
    """Plot each attribute of the ci simulator as individual figures one by one (e.g. the dataset, noise_map-map, PSF, \
     Signal-to_noise-map, etc).

    Set *autolens.simulator.arrays.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    ci_data : simulator.CCDData
        The ci simulator, which includes the observed simulator, noise_map-map, PSF, signal-to-noise_map-map, etc.
    origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    """

    if plot_image:
        image_line(
            ci_data=ci_data,
            line_region=line_region,
            mask=mask,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_noise_map:
        noise_map_line(
            ci_data=ci_data,
            line_region=line_region,
            mask=mask,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_ci_pre_cti:
        ci_pre_cti_line(
            ci_data=ci_data,
            line_region=line_region,
            mask=mask,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_signal_to_noise_map:
        signal_to_noise_map_line(
            ci_data=ci_data,
            line_region=line_region,
            mask=mask,
            output_path=output_path,
            output_format=output_format,
        )


def image_line(
    imaging,
    line_region,
    as_subplot=False,
    figsize=(7, 7),
    title="Image",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="image_line",
):
    """Plot the observed image of the ccd simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    imaging : CIFrame
        The image of the dataset.
    """
    ci_line_plotters.plot_line_from_ci_frame(
        ci_frame=imaging,
        line_region=line_region,
        as_subplot=as_subplot,
        figsize=figsize,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def noise_map_line(
    imaging,
    line_region,
    as_subplot=False,
    figsize=(7, 7),
    title="Noise-Map",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="noise_map_line",
):
    """Plot the observed noise_map of the ccd simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    imaging : CIFrame
        The noise_map of the dataset.
    """
    ci_line_plotters.plot_line_from_ci_frame(
        ci_frame=imaging,
        line_region=line_region,
        as_subplot=as_subplot,
        figsize=figsize,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def ci_pre_cti_line(
    imaging,
    line_region,
    as_subplot=False,
    figsize=(7, 7),
    title="CI Pre-CTI Image",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="ci_pre_cti_line",
):
    """Plot the observed ci_pre_cti of the ccd simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    imaging : CIFrame
        The ci_pre_cti of the dataset.
    """
    ci_line_plotters.plot_line_from_ci_frame(
        ci_frame=imaging,
        line_region=line_region,
        as_subplot=as_subplot,
        figsize=figsize,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def signal_to_noise_map_line(
    imaging,
    line_region,
    as_subplot=False,
    figsize=(7, 7),
    title="Signal-To-Noise Map",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="signal_to_noise_map_line",
):
    """Plot the observed signal_to_noise_map of the ccd simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    imaging : CIFrame
        The signal_to_noise_map of the dataset.
    """
    ci_line_plotters.plot_line_from_ci_frame(
        ci_frame=imaging,
        line_region=line_region,
        as_subplot=as_subplot,
        figsize=figsize,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )
