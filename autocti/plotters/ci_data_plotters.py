from matplotlib import pyplot as plt

from autocti.plotters import ci_plotter_util, ci_imaging_plotters
from autoarray.util import array_util


def plot_ci_data_for_phase(
    ci_datas_extracted,
    extract_array_from_mask,
    should_plot_as_subplot,
    should_plot_image,
    should_plot_noise_map,
    should_plot_ci_pre_cti,
    should_plot_signal_to_noise_map,
    should_plot_parallel_front_edge_line,
    should_plot_parallel_trails_line,
    should_plot_serial_front_edge_line,
    should_plot_serial_trails_line,
    visualize_path,
):

    plot_ci_data_arrays_for_phase(
        ci_datas_extracted=ci_datas_extracted,
        should_plot_as_subplot=should_plot_as_subplot,
        should_plot_image=should_plot_image,
        should_plot_noise_map=should_plot_noise_map,
        should_plot_ci_pre_cti=should_plot_ci_pre_cti,
        should_plot_signal_to_noise_map=should_plot_signal_to_noise_map,
        visualize_path=visualize_path,
    )

    plot_ci_data_lines_for_phase(
        ci_datas_extracted=ci_datas_extracted,
        should_plot_as_subplot=should_plot_as_subplot,
        should_plot_image=should_plot_image,
        should_plot_noise_map=should_plot_noise_map,
        should_plot_ci_pre_cti=should_plot_ci_pre_cti,
        should_plot_signal_to_noise_map=should_plot_signal_to_noise_map,
        should_plot_parallel_front_edge_line=should_plot_parallel_front_edge_line,
        should_plot_parallel_trails_line=should_plot_parallel_trails_line,
        should_plot_serial_front_edge_line=should_plot_serial_front_edge_line,
        should_plot_serial_trails_line=should_plot_serial_trails_line,
        visualize_path=visualize_path,
    )


def plot_ci_data_arrays_for_phase(
    ci_datas_extracted,
    extract_array_from_mask,
    should_plot_as_subplot,
    should_plot_image,
    should_plot_noise_map,
    should_plot_ci_pre_cti,
    should_plot_signal_to_noise_map,
    visualize_path,
):

    for data_index in range(len(ci_datas_extracted)):

        normalization = ci_datas_extracted[data_index].ci_pattern.normalization
        output_path = (
            visualize_path
            + "/"
            + "ci_image_"
            + str(int(normalization))
            + "/structures/"
        )
        array_util.make_path_if_does_not_exist(path=output_path + "fits/")

        if should_plot_as_subplot:

            plot_ci_subplot(
                ci_data=ci_datas_extracted[data_index],
                mask=ci_datas_extracted[data_index].mask,
                output_path=output_path,
                output_format="png",
            )

        plot_ci_data_individual(
            ci_data=ci_datas_extracted[data_index],
            mask=ci_datas_extracted[data_index].mask,
            should_plot_image=should_plot_image,
            should_plot_noise_map=should_plot_noise_map,
            should_plot_ci_pre_cti=should_plot_ci_pre_cti,
            should_plot_signal_to_noise_map=should_plot_signal_to_noise_map,
            output_path=output_path,
            output_format="png",
        )


def plot_ci_data_lines_for_phase(
    ci_datas_extracted,
    should_plot_as_subplot,
    should_plot_image,
    should_plot_noise_map,
    should_plot_ci_pre_cti,
    should_plot_signal_to_noise_map,
    should_plot_parallel_front_edge_line,
    should_plot_parallel_trails_line,
    should_plot_serial_front_edge_line,
    should_plot_serial_trails_line,
    visualize_path,
):

    line_regions = ci_plotter_util.line_regions_from_should_plots(
        should_plot_parallel_front_edge_line=should_plot_parallel_front_edge_line,
        should_plot_parallel_trails_line=should_plot_parallel_trails_line,
        should_plot_serial_front_edge_line=should_plot_serial_front_edge_line,
        should_plot_serial_trails_line=should_plot_serial_trails_line,
    )

    for data_index in range(len(ci_datas_extracted)):

        for line_region in line_regions:

            normalization = ci_datas_extracted[data_index].ci_pattern.normalization
            output_path = (
                visualize_path
                + "/"
                + "ci_image_"
                + str(int(normalization))
                + "/"
                + line_region
                + "/"
            )
            array_util.make_path_if_does_not_exist(path=output_path + "fits/")

            if should_plot_as_subplot:

                plot_ci_line_subplot(
                    ci_data=ci_datas_extracted[data_index],
                    line_region=line_region,
                    output_path=output_path,
                    output_format="png",
                )

            plot_ci_data_line_individual(
                ci_data=ci_datas_extracted[data_index],
                mask=ci_datas_extracted[data_index].mask,
                line_region=line_region,
                should_plot_image=should_plot_image,
                should_plot_noise_map=should_plot_noise_map,
                should_plot_ci_pre_cti=should_plot_ci_pre_cti,
                should_plot_signal_to_noise_map=should_plot_signal_to_noise_map,
                output_path=output_path,
                output_format="png",
            )

            plot_ci_data_line_individual(
                ci_data=ci_datas_extracted[data_index],
                mask=ci_datas_extracted[data_index].mask,
                line_region=line_region,
                should_plot_image=should_plot_image,
                should_plot_noise_map=should_plot_noise_map,
                should_plot_ci_pre_cti=should_plot_ci_pre_cti,
                should_plot_signal_to_noise_map=should_plot_signal_to_noise_map,
                output_path=output_path,
                output_format="fits",
            )


def plot_ci_subplot(
    ci_data,
    mask=None,
    figsize=None,
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
    titlesize=10,
    xlabelsize=10,
    ylabelsize=10,
    xyticksize=10,
    output_path=None,
    output_filename="ci_data",
    output_format="show",
):
    """Plot the ci simulator as a sub-plotters of all its quantites (e.g. the dataset, noise_map-map, PSF, Signal-to_noise-map, \
     etc).

    Set *autolens.simulator.arrays.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    ci_data : simulator.CCDData
        The ci simulator, which includes the observed simulator, noise_map-map, PSF, signal-to-noise_map-map, etc.
    include_origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or simulator.arrays.grid_lines.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    ignore_config : bool
        If *False*, the config file general.ini is used to determine whether the subpot is plotted. If *True*, the \
        config file is ignored.
    """

    rows, columns, figsize_tool = ci_plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=4
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    plot_image(
        ci_data=ci_data,
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
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
    )

    plt.subplot(rows, columns, 2)

    plot_noise_map(
        ci_data=ci_data,
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
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
    )

    plt.subplot(rows, columns, 3)

    plot_ci_pre_cti(
        ci_data=ci_data,
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
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
    )

    plt.subplot(rows, columns, 4)

    plot_signal_to_noise_map(
        ci_data=ci_data,
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
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
    )

    ci_plotter_util.output_subplot_array(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )

    plt.close()


def plot_ci_data_individual(
    ci_data,
    mask=None,
    should_plot_image=False,
    should_plot_noise_map=False,
    should_plot_ci_pre_cti=False,
    should_plot_signal_to_noise_map=False,
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
    include_origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    """

    if should_plot_image:
        plot_image(
            ci_data=ci_data,
            mask=mask,
            output_path=output_path,
            output_format=output_format,
        )

    if should_plot_noise_map:
        plot_noise_map(
            ci_data=ci_data,
            mask=mask,
            output_path=output_path,
            output_format=output_format,
        )

    if should_plot_ci_pre_cti:
        plot_ci_pre_cti(
            ci_data=ci_data,
            output_path=output_path,
            output_format=output_format,
        )

    if should_plot_signal_to_noise_map:
        plot_signal_to_noise_map(
            ci_data=ci_data,
            mask=mask,
            output_path=output_path,
            output_format=output_format,
        )


def plot_image(
    ci_data,
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
    title="Charge Injection Image",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="ci_image",
):
    """Plot the observed image of the ci simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : CIFrame
        The image of the dataset.
    """
    ci_imaging_plotters.plot_image(
        image=ci_data.image,
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


def plot_noise_map(
    ci_data,
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
    title="Charge Injection Noise-Map",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="ci_noise_map",
):
    """Plot the observed noise_map of the ci simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    noise_map : CIFrame
        The noise map of the dataset.
    """
    ci_imaging_plotters.plot_noise_map(
        noise_map=ci_data.noise_map,
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


def plot_ci_pre_cti(
    ci_data,
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
    title="Charge Injection Pre-CTI Image",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="ci_pre_cti",
):
    """Plot the observed ci_pre_cti of the ci simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    ci_pre_cti : CIFrame
        The ci_pre_cti of the dataset.
    """
    ci_imaging_plotters.ci_pre_cti(
        ci_pre_cti=ci_data.ci_pre_cti,
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


def plot_signal_to_noise_map(
    ci_data,
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
    title="Charge Injection Signal-to-Noise Map",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="ci_signal_to_noise_map",
):
    """Plot the observed signal_to_noise_map of the ci simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    signal_to_noise_map : CIFrame
        The signal-to-noise map of the dataset.
    """
    ci_imaging_plotters.plot_signal_to_noise_map(
        signal_to_noise_map=ci_data.signal_to_noise_map,
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
    include_origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or simulator.arrays.grid_lines.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
        over the immage.
    ignore_config : bool
        If *False*, the config file general.ini is used to determine whether the subpot is plotted. If *True*, the \
        config file is ignored.
    """

    rows, columns, figsize_tool = ci_plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=4
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    plot_image_line(
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

    plot_noise_map_line(
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

    plot_ci_pre_cti_line(
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

    plot_signal_to_noise_map_line(
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
    should_plot_image=False,
    should_plot_noise_map=False,
    should_plot_ci_pre_cti=False,
    should_plot_signal_to_noise_map=False,
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
    include_origin : True
        If true, the origin of the dataset's coordinate system is plotted as a 'x'.
    """

    if should_plot_image:
        plot_image_line(
            ci_data=ci_data,
            line_region=line_region,
            mask=mask,
            output_path=output_path,
            output_format=output_format,
        )

    if should_plot_noise_map:
        plot_noise_map_line(
            ci_data=ci_data,
            line_region=line_region,
            mask=mask,
            output_path=output_path,
            output_format=output_format,
        )

    if should_plot_ci_pre_cti:
        plot_ci_pre_cti_line(
            ci_data=ci_data,
            line_region=line_region,
            mask=mask,
            output_path=output_path,
            output_format=output_format,
        )

    if should_plot_signal_to_noise_map:
        plot_signal_to_noise_map_line(
            ci_data=ci_data,
            line_region=line_region,
            mask=mask,
            output_path=output_path,
            output_format=output_format,
        )


def plot_image_line(
    ci_data,
    line_region,
    mask=None,
    as_subplot=False,
    figsize=(7, 7),
    title="CI Image",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="ci_image_line",
):
    """Plot the observed image of the ci simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : CIFrame
        The image of the dataset.
    """

    ci_imaging_plotters.plot_image_line(
        image=ci_data.image,
        line_region=line_region,
        ci_frame=ci_data.ci_frame,
        mask=mask,
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


def plot_noise_map_line(
    ci_data,
    line_region,
    mask=None,
    as_subplot=False,
    figsize=(7, 7),
    title="CI Noise Map",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="ci_noise_map_line",
):
    """Plot the observed noise_map of the ci simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    noise_map : CIFrame
        The noise map of the dataset.
    """
    ci_imaging_plotters.plot_noise_map_line(
        noise_map=ci_data.noise_map,
        line_region=line_region,
        ci_frame=ci_data.ci_frame,
        mask=mask,
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


def plot_ci_pre_cti_line(
    ci_data,
    line_region,
    mask=None,
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
    """Plot the observed ci_pre_cti of the ci simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    ci_pre_cti : CIFrame
        The ci_pre_cti of the dataset.
    """
    ci_imaging_plotters.plot_ci_pre_cti_line(
        ci_pre_cti=ci_data.ci_pre_cti,
        line_region=line_region,
        ci_frame=ci_data.ci_frame,
        mask=mask,
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


def plot_signal_to_noise_map_line(
    ci_data,
    line_region,
    mask=None,
    as_subplot=False,
    figsize=(7, 7),
    title="CI Signal-To-Noise Map",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="ci_signal_to_noise_map_line",
):
    """Plot the observed signal_to_noise_map of the ci simulator.

    Set *autocti.simulator.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    signal_to_noise_map : CIFrame
        The signal-to-noise map of the dataset.
    """
    ci_imaging_plotters.plot_signal_to_noise_map_line(
        signal_to_noise_map=ci_data.signal_to_noise_map,
        line_region=line_region,
        ci_frame=ci_data.ci_frame,
        mask=mask,
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
