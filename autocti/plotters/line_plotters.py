import matplotlib.pyplot as plt

from autocti import exc
from autocti.plotters import plotter_util


def line_from_line_region_and_arrays(line_region, array, mask, ci_frame):
    """

    Parameters
    -----------
    ci_frame : charge_injection.ci_frame.ChInj
    """
    if line_region is "parallel_front_edge":
        return ci_frame.parallel_front_edge_line_binned_over_columns_from_frame(
            array=array, mask=mask
        )
    elif line_region is "parallel_trails":
        return ci_frame.parallel_trails_line_binned_over_columns_from_frame(
            array=array, mask=mask
        )
    elif line_region is "serial_front_edge":
        return ci_frame.serial_front_edge_line_binned_over_rows_from_frame(
            array=array, mask=mask
        )
    elif line_region is "serial_trails":
        return ci_frame.serial_trails_line_binned_over_rows_from_frame(
            array=array, mask=mask
        )
    else:
        raise exc.PlottingException(
            "The line region specified for the plotting of a line was invalid"
        )


def plot_line_from_array_and_ci_frame(
    array,
    line_region,
    ci_frame,
    mask=None,
    as_subplot=False,
    figsize=(7, 7),
    title="Stack",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="line",
):

    line = line_from_line_region_and_arrays(
        line_region=line_region, array=array, mask=mask, ci_frame=ci_frame
    )

    plot_line(
        line=line,
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


def plot_line(
    line,
    as_subplot=False,
    figsize=(7, 7),
    title="Stack",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="line",
):
    """Plot an arrays of hyper as a figure.

    Parameters
    -----------
    line : ndarray or hyper.arrays.scaled_array.ScaledArray
        The 2D arrays of hyper which is plotted.
    mask : ndarray of simulate.mask.Mask
        The masks applied to the hyper, the edge of which is plotted as a set of points over the plotted arrays.
    extract_line_from_mask : bool
        The plotter arrays is extracted using the mask, such that masked values are plotted as zeros. This ensures \
        bright features outside the mask do not impact the color map of the plotters.
    as_subplot : bool
        Whether the arrays is plotted as part of a subplot, in which case the grid figure is not opened / closed.
    figsize : (int, int)
        The size of the figure in (rows, columns).
    aspect : str
        The aspect ratio of the hyper, specifically whether it is forced to be square ('equal') or adapts its size to \
        the figure size ('auto').
    cb_ticksize : int
        The size of the tick labels on the colorbar.
    cb_fraction : float
        The fraction of the figure that the colorbar takes up, which resizes the colorbar relative to the figure.
    cb_pad : float
        Pads the color bar in the figure, which resizes the colorbar relative to the figure.
    xlabelsize : int
        The fontsize of the x axes label.
    ylabelsize : int
        The fontsize of the y axes label.
    xyticksize : int
        The font size of the x and y ticks on the figure axes.
    output_path : str
        The path on the hard-disk where the figure is output.
    output_filename : str
        The filename of the figure that is output.
    output_format : str
        The format the figue is output:
        'show' - display on computer screen.
        'png' - output to hard-disk as a png.
        'fits' - output to hard-disk as a fits file.'
    """

    if line is None:
        return

    plot_figure(line=line, as_subplot=as_subplot, figsize=figsize)

    plotter_util.set_title(title=title, titlesize=titlesize)
    set_xy_labels_and_ticksize(
        xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize
    )
    plotter_util.output_figure(
        line,
        as_subplot=as_subplot,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
    plotter_util.close_figure(as_subplot=as_subplot)


def plot_figure(line, as_subplot, figsize):
    """Open a matplotlib figure and plotters the arrays of hyper on it.

    Parameters
    -----------
    line : ndarray or hyper.arrays.scaled_array.ScaledArray
        The 2D arrays of hyper which is plotted.
    as_subplot : bool
        Whether the arrays is plotted as part of a subplot, in which case the grid figure is not opened / closed.
    units : str
        The units of the y / x axis of the plots, in arc-seconds ('arcsec') or kiloparsecs ('kpc').
    kpc_per_arcsec : float or None
        The conversion factor between arc-seconds and kiloparsecs, required to plotters the units in kpc.
    figsize : (int, int)
        The size of the figure in (rows, columns).
    aspect : str
        The aspect ratio of the hyper, specifically whether it is forced to be square ('equal') or adapts its size to \
        the figure size ('auto').
    cmap : str
        The colormap the arrays is plotted using, which may be chosen from the standard matplotlib colormaps.
    norm : str
        The normalization of the colormap used to plotters the hyper, specifically whether it is linear ('linear'), log \
        ('log') or a symmetric log normalization ('symmetric_log').
    norm_min : float or None
        The minimum arrays value the colormap map spans (all values below this value are plotted the same color).
    norm_max : float or None
        The maximum arrays value the colormap map spans (all values above this value are plotted the same color).
    linthresh : float
        For the 'symmetric_log' colormap normalization ,this specifies the range of values within which the colormap \
        is linear.
    linscale : float
        For the 'symmetric_log' colormap normalization, this allowws the linear range set by linthresh to be stretched \
        relative to the logarithmic range.
    xticks_manual :  [] or None
        If input, the xticks do not use the arrays's default xticks but instead overwrite them as these values.
    yticks_manual :  [] or None
        If input, the yticks do not use the arrays's default yticks but instead overwrite them as these values.
    """
    plotter_util.setup_figure(figsize=figsize, as_subplot=as_subplot)
    plt.plot(line)


def set_xy_labels_and_ticksize(xlabelsize, ylabelsize, xyticksize):
    """Set the x and y labels of the figure, and set the fontsize of those labels.

    The x and y labels are always the distance scales, thus the labels are either arc-seconds or kpc and depend on the \
    units the figure is plotted in.

    Parameters
    -----------
    units : str
        The units of the y / x axis of the plots, in arc-seconds ('arcsec') or kiloparsecs ('kpc').
    kpc_per_arcsec : float
        The conversion factor between arc-seconds and kiloparsecs, required to plotters the units in kpc.
    xlabelsize : int
        The fontsize of the x axes label.
    ylabelsize : int
        The fontsize of the y axes label.
    xyticksize : int
        The font size of the x and y ticks on the figure axes.
    """
    plt.xlabel("x (pixels)", fontsize=xlabelsize)
    plt.ylabel("y (pixels)", fontsize=ylabelsize)
    plt.tick_params(labelsize=xyticksize)
