from matplotlib import pyplot as plt

from autocti.plot import cti_plotters
from autoarray.util import array_util


def plot_ci_fit_for_phase(
    fits,
    during_analysis,
    extract_array_from_mask,
    plot_all_at_end_png,
    plot_all_at_end_fits,
    plot_as_subplot,
    plot_residual_maps_subplot,
    plot_chi_squared_maps_subplot,
    plot_image,
    plot_noise_map,
    plot_signal_to_noise_map,
    plot_ci_pre_cti,
    plot_ci_post_cti,
    plot_residual_map,
    plot_chi_squared_map,
    plot_noise_scaling_maps,
    plot_parallel_front_edge_line,
    plot_parallel_trails_line,
    plot_serial_front_edge_line,
    plot_serial_trails_line,
    visualize_path=None,
):

    plot_ci_fit_arrays_for_phase(
        fits=fits,
        during_analysis=during_analysis,
        plot_all_at_end_png=plot_all_at_end_png,
        plot_all_at_end_fits=plot_all_at_end_fits,
        plot_as_subplot=plot_as_subplot,
        plot_residual_maps_subplot=plot_residual_maps_subplot,
        plot_chi_squared_maps_subplot=plot_chi_squared_maps_subplot,
        plot_image=plot_image,
        plot_noise_map=plot_noise_map,
        plot_signal_to_noise_map=plot_signal_to_noise_map,
        plot_ci_pre_cti=plot_ci_pre_cti,
        plot_ci_post_cti=plot_ci_post_cti,
        plot_noise_scaling_maps=plot_noise_scaling_maps,
        plot_residual_map=plot_residual_map,
        plot_chi_squared_map=plot_chi_squared_map,
        visualize_path=visualize_path,
    )

    plot_ci_fit_lines_for_phase(
        fits=fits,
        during_analysis=during_analysis,
        plot_all_at_end_png=plot_all_at_end_png,
        plot_all_at_end_fits=plot_all_at_end_fits,
        plot_as_subplot=plot_as_subplot,
        plot_residual_maps_subplot=plot_residual_maps_subplot,
        plot_chi_squared_maps_subplot=plot_chi_squared_maps_subplot,
        plot_image=plot_image,
        plot_noise_map=plot_noise_map,
        plot_signal_to_noise_map=plot_signal_to_noise_map,
        plot_ci_pre_cti=plot_ci_pre_cti,
        plot_ci_post_cti=plot_ci_post_cti,
        plot_residual_map=plot_residual_map,
        plot_chi_squared_map=plot_chi_squared_map,
        plot_parallel_front_edge_line=plot_parallel_front_edge_line,
        plot_parallel_trails_line=plot_parallel_trails_line,
        plot_serial_front_edge_line=plot_serial_front_edge_line,
        plot_serial_trails_line=plot_serial_trails_line,
        visualize_path=visualize_path,
    )


def plot_ci_fit_arrays_for_phase(
    fits,
    during_analysis,
    extract_array_from_mask,
    plot_all_at_end_png,
    plot_all_at_end_fits,
    plot_as_subplot,
    plot_residual_maps_subplot,
    plot_chi_squared_maps_subplot,
    plot_image,
    plot_noise_map,
    plot_signal_to_noise_map,
    plot_ci_pre_cti,
    plot_ci_post_cti,
    plot_residual_map,
    plot_chi_squared_map,
    plot_noise_scaling_maps,
    visualize_path=None,
):

    for fit in fits:

        normalization = fit.ci_data_masked.ci_pattern.normalization
        output_path = (
            visualize_path
            + "/"
            + "ci_image_"
            + str(int(normalization))
            + "/structures/"
        )
        array_util.make_path_if_does_not_exist(path=output_path + "fits/")

        if plot_as_subplot:

            plot_fit_subplot(fit=fit, output_path=output_path, format="png")

        plot_fit_individuals(
            fit=fit,
            plot_image=plot_image,
            plot_noise_map=plot_noise_map,
            plot_signal_to_noise_map=plot_signal_to_noise_map,
            plot_ci_pre_cti=plot_ci_pre_cti,
            plot_ci_post_cti=plot_ci_post_cti,
            plot_residual_map=plot_residual_map,
            plot_chi_squared_map=plot_chi_squared_map,
            plot_noise_scaling_maps=plot_noise_scaling_maps,
            output_path=output_path,
            format="png",
        )

        if not during_analysis:

            if plot_all_at_end_png:

                plot_fit_individuals(
                    fit=fit,
                    plot_image=True,
                    plot_noise_map=True,
                    plot_signal_to_noise_map=True,
                    plot_ci_pre_cti=True,
                    plot_ci_post_cti=True,
                    plot_residual_map=True,
                    plot_chi_squared_map=True,
                    plot_noise_scaling_maps=True,
                    output_path=output_path,
                    format="png",
                )

            if plot_all_at_end_fits:

                plot_fit_individuals(
                    fit=fit,
                    plot_image=True,
                    plot_noise_map=True,
                    plot_signal_to_noise_map=True,
                    plot_ci_pre_cti=True,
                    plot_ci_post_cti=True,
                    plot_residual_map=True,
                    plot_chi_squared_map=True,
                    plot_noise_scaling_maps=True,
                    output_path="{}/fits/".format(output_path),
                    output_format="fits",
                )

        output_path = visualize_path + "/"

        if plot_residual_maps_subplot:

            plot_fit_residual_maps_subplot(
                fits=fits, output_path=output_path, format="png"
            )

        if plot_chi_squared_maps_subplot:

            plot_fit_chi_squared_maps_subplot(
                fits=fits, output_path=output_path, format="png"
            )


def plot_ci_fit_lines_for_phase(
    fits,
    during_analysis,
    plot_all_at_end_png,
    plot_all_at_end_fits,
    plot_as_subplot,
    plot_residual_maps_subplot,
    plot_chi_squared_maps_subplot,
    plot_image,
    plot_noise_map,
    plot_signal_to_noise_map,
    plot_ci_pre_cti,
    plot_ci_post_cti,
    plot_residual_map,
    plot_chi_squared_map,
    plot_parallel_front_edge_line,
    plot_parallel_trails_line,
    plot_serial_front_edge_line,
    plot_serial_trails_line,
    visualize_path=None,
):

    line_regions = ci_plotter_util.line_regions_from_plots(
        plot_parallel_front_edge_line=plot_parallel_front_edge_line,
        plot_parallel_trails_line=plot_parallel_trails_line,
        plot_serial_front_edge_line=plot_serial_front_edge_line,
        plot_serial_trails_line=plot_serial_trails_line,
    )

    for line_region in line_regions:

        for fit in fits:

            normalization = fit.ci_data_masked.ci_pattern.normalization
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

            if plot_as_subplot:

                plot_fit_line_subplot(
                    fit=fit,
                    line_region=line_region,
                    output_path=output_path,
                    format="png",
                )

            plot_fit_line_individuals(
                fit=fit,
                line_region=line_region,
                plot_image=plot_image,
                plot_noise_map=plot_noise_map,
                plot_signal_to_noise_map=plot_signal_to_noise_map,
                plot_ci_pre_cti=plot_ci_pre_cti,
                plot_ci_post_cti=plot_ci_post_cti,
                plot_residual_map=plot_residual_map,
                plot_chi_squared_map=plot_chi_squared_map,
                output_path=output_path,
                format="png",
            )

            if not during_analysis:

                if plot_all_at_end_png:

                    plot_fit_line_individuals(
                        fit=fit,
                        line_region=line_region,
                        plot_image=True,
                        plot_noise_map=True,
                        plot_signal_to_noise_map=True,
                        plot_ci_pre_cti=True,
                        plot_ci_post_cti=True,
                        plot_residual_map=True,
                        plot_chi_squared_map=True,
                        output_path=output_path,
                        format="png",
                    )

                if plot_all_at_end_fits:

                    plot_fit_line_individuals(
                        fit=fit,
                        line_region=line_region,
                        plot_image=True,
                        plot_noise_map=True,
                        plot_signal_to_noise_map=True,
                        plot_ci_pre_cti=True,
                        plot_ci_post_cti=True,
                        plot_residual_map=True,
                        plot_chi_squared_map=True,
                        output_path=output_path,
                        output_format="fits",
                    )

        output_path = visualize_path + "/" + line_region + "_"

        if plot_residual_maps_subplot:

            plot_fit_residual_maps_lines_subplot(
                fits=fits,
                line_region=line_region,
                output_path=output_path,
                format="png",
            )

        if plot_chi_squared_maps_subplot:

            plot_fit_chi_squared_maps_lines_subplot(
                fits=fits,
                line_region=line_region,
                output_path=output_path,
                format="png",
            )


def plot_fit_subplot(
    fit,
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
    xsize=10,
    ysize=10,
    xyticksize=10,
    output_path=None,
    output_filename="ci_fit",
    output_format="show",
):
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
    extract_array_from_mask
    fit : autolens.lens.fitting.Fitter
        Class containing fit between the model datas_ and observed lens datas_ (including residual_map, chi_squared_map etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
    output_filename : str
        The name of the file that is output, if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    rows, columns, figsize_tool = ci_plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=9
    )

    if figsize is None:
        figsize = figsize_tool

        sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    fit_plots.plot_image(
        fit=fit,
        mask=fit.mask,
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
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= 2)

    fit_plots.plot_noise_map(
        fit=fit,
        mask=fit.mask,
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
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= 3)

    fit_plots.plot_signal_to_noise_map(
        fit=fit,
        mask=fit.mask,
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
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= 4)

    fit_plots.plot_ci_pre_cti(
        fit=fit,
        mask=fit.mask,
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
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= 5)

    fit_plots.plot_ci_post_cti(
        fit=fit,
        mask=fit.mask,
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
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= 7)

    fit_plots.plot_residual_map(
        fit=fit,
        mask=fit.mask,
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
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= 8)

    fit_plots.plot_chi_squared_map(
        fit=fit,
        mask=fit.mask,
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
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )

    ci_plotter_util.output.to_figure(structure=None)(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )

    plt.close()


def plot_fit_residual_maps_subplot(
    fits,
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
    xsize=10,
    ysize=10,
    xyticksize=10,
    output_path=None,
    output_filename="ci_fits_residual_maps",
    output_format="show",
):
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
    extract_array_from_mask
    fit : autolens.lens.fitting.Fitter
        Class containing fit between the model datas_ and observed lens datas_ (including residual_map, chi_squared_map etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
    output_filename : str
        The name of the file that is output, if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    rows, columns, figsize_tool = ci_plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=len(fits)
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    for index, fit in enumerate(fits):

        sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= index + 1)

        fit_plots.plot_residual_map(
            fit=fit,
            mask=fit.mask,
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
            xsize=xsize,
            ysize=ysize,
            xyticksize=xyticksize,
            output_path=output_path,
            output_format=output_format,
            output_filename=output_filename,
        )

    ci_plotter_util.output.to_figure(structure=None)(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )

    plt.close()


def plot_fit_chi_squared_maps_subplot(
    fits,
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
    xsize=10,
    ysize=10,
    xyticksize=10,
    output_path=None,
    output_filename="ci_fits_chi_squared_maps",
    output_format="show",
):
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
    extract_array_from_mask
    fit : autolens.lens.fitting.Fitter
        Class containing fit between the model datas_ and observed lens datas_ (including chi_squared_map, chi_squared_map etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
    output_filename : str
        The name of the file that is output, if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    rows, columns, figsize_tool = ci_plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=len(fits)
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    for index, fit in enumerate(fits):

        sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= index + 1)

        fit_plots.plot_chi_squared_map(
            fit=fit,
            mask=fit.mask,
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
            xsize=xsize,
            ysize=ysize,
            xyticksize=xyticksize,
            output_path=output_path,
            output_format=output_format,
            output_filename=output_filename,
        )

    ci_plotter_util.output.to_figure(structure=None)(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )

    plt.close()


def plot_fit_individuals(
    fit,
    plot_image=False,
    plot_noise_map=False,
    plot_signal_to_noise_map=False,
    plot_ci_pre_cti=False,
    plot_ci_post_cti=False,
    plot_residual_map=False,
    plot_chi_squared_map=False,
    plot_noise_scaling_maps=False,
    output_path=None,
    output_format="show",
):

    if plot_image:
        fit_plots.plot_image(
            fit=fit, mask=fit.mask, output_path=output_path, output_format=output_format
        )

    if plot_noise_map:
        fit_plots.plot_noise_map(
            fit=fit, mask=fit.mask, output_path=output_path, output_format=output_format
        )

    if plot_signal_to_noise_map:
        fit_plots.plot_signal_to_noise_map(
            fit=fit, mask=fit.mask, output_path=output_path, output_format=output_format
        )

    if plot_ci_pre_cti:
        fit_plots.plot_ci_pre_cti(
            fit=fit, mask=fit.mask, output_path=output_path, output_format=output_format
        )

    if plot_ci_post_cti:
        fit_plots.plot_ci_post_cti(
            fit=fit, mask=fit.mask, output_path=output_path, output_format=output_format
        )

    if plot_residual_map:
        fit_plots.plot_residual_map(
            fit=fit, mask=fit.mask, output_path=output_path, output_format=output_format
        )

    if plot_chi_squared_map:
        fit_plots.plot_chi_squared_map(
            fit=fit, mask=fit.mask, output_path=output_path, output_format=output_format
        )

    if plot_noise_scaling_maps and hasattr(fit, "noise_scaling_maps"):

        fit_plots.plot_noise_scaling_maps(
            fit_hyper=fit,
            mask=fit.mask,
            output_path=output_path,
            output_format=output_format,
        )


def plot_fit_line_subplot(
    fit,
    line_region,
    figsize=None,
    titlesize=16,
    xsize=16,
    ysize=16,
    xyticksize=16,
    output_path=None,
    output_filename="ci_fit_line",
    output_format="show",
):
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
    extract_array_from_mask
    fit : autolens.lens.fitting.Fitter
        Class containing fit between the model datas_ and observed lens datas_ (including residual_map, chi_squared_map etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
    output_filename : str
        The name of the file that is output, if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    rows, columns, figsize_tool = ci_plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=9
    )

    if figsize is None:
        figsize = figsize_tool

        sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    fit_plots.plot_image_line(
        fit=fit,
        line_region=line_region,
        mask=fit.mask,
        as_subplot=True,
        figsize=figsize,
        titlesize=titlesize,
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= 2)

    fit_plots.plot_noise_map_line(
        fit=fit,
        line_region=line_region,
        mask=fit.mask,
        as_subplot=True,
        figsize=figsize,
        titlesize=titlesize,
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= 3)

    fit_plots.plot_signal_to_noise_map_line(
        fit=fit,
        line_region=line_region,
        mask=fit.mask,
        as_subplot=True,
        figsize=figsize,
        titlesize=titlesize,
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= 4)

    fit_plots.plot_ci_pre_cti_line(
        fit=fit,
        line_region=line_region,
        mask=fit.mask,
        as_subplot=True,
        figsize=figsize,
        titlesize=titlesize,
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= 5)

    fit_plots.plot_ci_post_cti_line(
        fit=fit,
        line_region=line_region,
        mask=fit.mask,
        as_subplot=True,
        figsize=figsize,
        titlesize=titlesize,
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= 7)

    fit_plots.plot_residual_map_line(
        fit=fit,
        line_region=line_region,
        mask=fit.mask,
        as_subplot=True,
        figsize=figsize,
        titlesize=titlesize,
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= 8)

    fit_plots.plot_chi_squared_map_line(
        fit=fit,
        line_region=line_region,
        mask=fit.mask,
        as_subplot=True,
        figsize=figsize,
        titlesize=titlesize,
        xsize=xsize,
        ysize=ysize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )

    ci_plotter_util.output.to_figure(structure=None)(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )

    plt.close()


def plot_fit_residual_maps_lines_subplot(
    fits,
    line_region,
    figsize=None,
    titlesize=16,
    xsize=16,
    ysize=16,
    xyticksize=16,
    output_path=None,
    output_filename="ci_fits_residual_maps_lines",
    output_format="show",
):
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
    extract_array_from_mask
    fit : autolens.lens.fitting.Fitter
        Class containing fit between the model datas_ and observed lens datas_ (including residual_map, residual_map etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
    output_filename : str
        The name of the file that is output, if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    rows, columns, figsize_tool = ci_plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=len(fits)
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    for index, fit in enumerate(fits):
        sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= index + 1)

        fit_plots.plot_residual_map_line(
            fit=fit,
            mask=fit.mask,
            line_region=line_region,
            as_subplot=True,
            titlesize=titlesize,
            xsize=xsize,
            ysize=ysize,
            xyticksize=xyticksize,
            output_path=output_path,
            output_format=output_format,
            output_filename=output_filename,
        )

    ci_plotter_util.output.to_figure(structure=None)(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )

    plt.close()


def plot_fit_chi_squared_maps_lines_subplot(
    fits,
    line_region,
    figsize=None,
    titlesize=16,
    xsize=16,
    ysize=16,
    xyticksize=16,
    output_path=None,
    output_filename="ci_fits_chi_squared_maps_lines",
    output_format="show",
):
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
    extract_array_from_mask
    fit : autolens.lens.fitting.Fitter
        Class containing fit between the model datas_ and observed lens datas_ (including chi_squared_map, chi_squared_map etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
    output_filename : str
        The name of the file that is output, if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    rows, columns, figsize_tool = ci_plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=len(fits)
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    for index, fit in enumerate(fits):

        sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= index + 1)

        fit_plots.plot_chi_squared_map_line(
            fit=fit,
            mask=fit.mask,
            line_region=line_region,
            as_subplot=True,
            titlesize=titlesize,
            xsize=xsize,
            ysize=ysize,
            xyticksize=xyticksize,
            output_path=output_path,
            output_format=output_format,
            output_filename=output_filename,
        )

    ci_plotter_util.output.to_figure(structure=None)(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )

    plt.close()


def plot_fit_line_individuals(
    fit,
    line_region,
    plot_image=False,
    plot_noise_map=False,
    plot_signal_to_noise_map=False,
    plot_ci_pre_cti=False,
    plot_ci_post_cti=False,
    plot_residual_map=False,
    plot_chi_squared_map=False,
    output_path=None,
    output_format="show",
):

    if plot_image:
        fit_plots.plot_image_line(
            fit=fit,
            line_region=line_region,
            mask=fit.mask,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_noise_map:
        fit_plots.plot_noise_map_line(
            fit=fit,
            line_region=line_region,
            mask=fit.mask,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_signal_to_noise_map:
        fit_plots.plot_signal_to_noise_map_line(
            fit=fit,
            line_region=line_region,
            mask=fit.mask,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_ci_pre_cti:
        fit_plots.plot_ci_pre_cti_line(
            fit=fit,
            line_region=line_region,
            mask=fit.mask,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_ci_post_cti:
        fit_plots.plot_ci_post_cti_line(
            fit=fit,
            line_region=line_region,
            mask=fit.mask,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_residual_map:
        fit_plots.plot_residual_map_line(
            fit=fit,
            line_region=line_region,
            mask=fit.mask,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_chi_squared_map:
        fit_plots.plot_chi_squared_map_line(
            fit=fit,
            line_region=line_region,
            mask=fit.mask,
            output_path=output_path,
            output_format=output_format,
        )
