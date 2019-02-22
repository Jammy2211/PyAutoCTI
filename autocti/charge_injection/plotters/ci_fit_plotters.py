from matplotlib import pyplot as plt

from autocti.charge_injection.plotters import fit_plotters
from autocti.data.plotters import plotter_util


def plot_fit_subplot(
        fit, extract_array_from_mask=False,
        figsize=None, aspect='equal',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
        output_path=None, output_filename='ci_fit', output_format='show'):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    xyticksize
    ylabelsize
    xlabelsize
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

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(number_subplots=9)

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    fit_plotters.plot_image(
        fit=fit, mask=fit.mask, extract_array_from_mask=extract_array_from_mask, as_subplot=True,
        figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

    plt.subplot(rows, columns, 2)

    fit_plotters.plot_noise_map(
        fit=fit, mask=fit.mask, extract_array_from_mask=extract_array_from_mask, as_subplot=True,
        figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

    plt.subplot(rows, columns, 3)

    fit_plotters.plot_signal_to_noise_map(
        fit=fit, mask=fit.mask, extract_array_from_mask=extract_array_from_mask, as_subplot=True,
        figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

    plt.subplot(rows, columns, 4)

    fit_plotters.plot_ci_pre_cti(
        fit=fit, mask=fit.mask, extract_array_from_mask=extract_array_from_mask, as_subplot=True,
        figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

    plt.subplot(rows, columns, 5)

    fit_plotters.plot_ci_post_cti(
        fit=fit, mask=fit.mask, extract_array_from_mask=extract_array_from_mask, as_subplot=True,
        figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

    plt.subplot(rows, columns, 7)

    fit_plotters.plot_residual_map(
        fit=fit, mask=fit.mask, extract_array_from_mask=extract_array_from_mask, as_subplot=True,
        figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

    plt.subplot(rows, columns, 8)

    fit_plotters.plot_chi_squared_map(
        fit=fit, mask=fit.mask, extract_array_from_mask=extract_array_from_mask, as_subplot=True,
        figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

    plotter_util.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                      output_format=output_format)

    plt.close()


def plot_fit_individuals(
        fit, extract_array_from_mask=False,
        should_plot_image=False,
        should_plot_noise_map=False,
        should_plot_signal_to_noise_map=False,
        should_plot_ci_pre_cti=False,
        should_plot_ci_post_cti=False,
        should_plot_residual_map=False,
        should_plot_chi_squared_map=False,
        output_path=None, output_format='show'):
    if should_plot_image:
        fit_plotters.plot_image(
            fit=fit, mask=fit.mask, extract_array_from_mask=extract_array_from_mask,
            output_path=output_path, output_format=output_format)

    if should_plot_noise_map:
        fit_plotters.plot_noise_map(
            fit=fit, mask=fit.mask, extract_array_from_mask=extract_array_from_mask,
            output_path=output_path, output_format=output_format)

    if should_plot_signal_to_noise_map:
        fit_plotters.plot_signal_to_noise_map(
            fit=fit, mask=fit.mask, extract_array_from_mask=extract_array_from_mask,
            output_path=output_path, output_format=output_format)

    if should_plot_ci_pre_cti:
        fit_plotters.plot_ci_pre_cti(
            fit=fit, mask=fit.mask, extract_array_from_mask=extract_array_from_mask,
            output_path=output_path, output_format=output_format)

    if should_plot_ci_post_cti:
        fit_plotters.plot_ci_post_cti(
            fit=fit, mask=fit.mask, extract_array_from_mask=extract_array_from_mask,
            output_path=output_path, output_format=output_format)

    if should_plot_residual_map:
        fit_plotters.plot_residual_map(
            fit=fit, mask=fit.mask, extract_array_from_mask=extract_array_from_mask,
            output_path=output_path, output_format=output_format)

    if should_plot_chi_squared_map:
        fit_plotters.plot_chi_squared_map(
            fit=fit, mask=fit.mask, extract_array_from_mask=extract_array_from_mask,
            output_path=output_path, output_format=output_format)
