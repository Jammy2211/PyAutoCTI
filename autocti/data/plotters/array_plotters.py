import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import itertools

from autocti import exc
from autocti.data.plotters import plotter_util


def plot_array(array, mask=None, extract_array_from_mask=False, as_subplot=False,
               figsize=(7, 7), aspect='equal',
               cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
               cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
               title='Array', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
               output_path=None, output_format='show', output_filename='array'):
    """Plot an array of hyper as a figure.

    Parameters
    -----------
    array : ndarray or hyper.array.scaled_array.ScaledArray
        The 2D array of hyper which is plotted.
    mask : ndarray of data.mask.Mask
        The masks applied to the hyper, the edge of which is plotted as a set of points over the plotted array.
    extract_array_from_mask : bool
        The plotter array is extracted using the mask, such that masked values are plotted as zeros. This ensures \
        bright features outside the mask do not impact the color map of the plot.
    as_subplot : bool
        Whether the array is plotted as part of a subplot, in which case the grid figure is not opened / closed.
    figsize : (int, int)
        The size of the figure in (rows, columns).
    aspect : str
        The aspect ratio of the hyper, specifically whether it is forced to be square ('equal') or adapts its size to \
        the figure size ('auto').
    cmap : str
        The colormap the array is plotted using, which may be chosen from the standard matplotlib colormaps.
    norm : str
        The normalization of the colormap used to plot the hyper, specifically whether it is linear ('linear'), log \
        ('log') or a symmetric log normalization ('symmetric_log').
    norm_min : float or None
        The minimum array value the colormap map spans (all values below this value are plotted the same color).
    norm_max : float or None
        The maximum array value the colormap map spans (all values above this value are plotted the same color).
    linthresh : float
        For the 'symmetric_log' colormap normalization ,this specifies the range of values within which the colormap \
        is linear.
    linscale : float
        For the 'symmetric_log' colormap normalization, this allowws the linear range set by linthresh to be stretched \
        relative to the logarithmic range.
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

    if array is None:
        return

    if extract_array_from_mask and mask is not None:
        array = np.add(array, 0.0, out=np.zeros_like(array), where=np.asarray(mask) == 0)

    plot_figure(array=array, as_subplot=as_subplot,
                figsize=figsize, aspect=aspect, cmap=cmap, norm=norm,
                norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale)

    plotter_util.set_title(title=title, titlesize=titlesize)
    set_xy_labels_and_ticksize(xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize)

    set_colorbar(cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                 cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels)
    plotter_util.output_figure(array, as_subplot=as_subplot, output_path=output_path, output_filename=output_filename,
                               output_format=output_format)
    plotter_util.close_figure(as_subplot=as_subplot)


def plot_figure(array, as_subplot, figsize, aspect, cmap, norm, norm_min, norm_max,
                linthresh, linscale):
    """Open a matplotlib figure and plot the array of hyper on it.

    Parameters
    -----------
    array : ndarray or hyper.array.scaled_array.ScaledArray
        The 2D array of hyper which is plotted.
    as_subplot : bool
        Whether the array is plotted as part of a subplot, in which case the grid figure is not opened / closed.
    units : str
        The units of the y / x axis of the plots, in arc-seconds ('arcsec') or kiloparsecs ('kpc').
    kpc_per_arcsec : float or None
        The conversion factor between arc-seconds and kiloparsecs, required to plot the units in kpc.
    figsize : (int, int)
        The size of the figure in (rows, columns).
    aspect : str
        The aspect ratio of the hyper, specifically whether it is forced to be square ('equal') or adapts its size to \
        the figure size ('auto').
    cmap : str
        The colormap the array is plotted using, which may be chosen from the standard matplotlib colormaps.
    norm : str
        The normalization of the colormap used to plot the hyper, specifically whether it is linear ('linear'), log \
        ('log') or a symmetric log normalization ('symmetric_log').
    norm_min : float or None
        The minimum array value the colormap map spans (all values below this value are plotted the same color).
    norm_max : float or None
        The maximum array value the colormap map spans (all values above this value are plotted the same color).
    linthresh : float
        For the 'symmetric_log' colormap normalization ,this specifies the range of values within which the colormap \
        is linear.
    linscale : float
        For the 'symmetric_log' colormap normalization, this allowws the linear range set by linthresh to be stretched \
        relative to the logarithmic range.
    xticks_manual :  [] or None
        If input, the xticks do not use the array's default xticks but instead overwrite them as these values.
    yticks_manual :  [] or None
        If input, the yticks do not use the array's default yticks but instead overwrite them as these values.
    """

    plotter_util.setup_figure(figsize=figsize, as_subplot=as_subplot)

    norm_min, norm_max = get_normalization_min_max(array=array, norm_min=norm_min, norm_max=norm_max)
    norm_scale = get_normalization_scale(norm=norm, norm_min=norm_min, norm_max=norm_max,
                                         linthresh=linthresh, linscale=linscale)

    extent = get_extent(array=array)

    plt.imshow(array, aspect=aspect, cmap=cmap, norm=norm_scale, extent=extent)


def get_extent(array):
    """Get the extent of the dimensions of the array in the units of the figure (e.g. arc-seconds or kpc).

    This is used to set the extent of the array and thus the y / x axis limits.

    Parameters
    -----------
    array : ndarray or hyper.array.scaled_array.ScaledArray
        The 2D array of hyper which is plotted.
    units : str
        The units of the y / x axis of the plots, in arc-seconds ('arcsec') or kiloparsecs ('kpc').
    kpc_per_arcsec : float
        The conversion factor between arc-seconds and kiloparsecs, required to plot the units in kpc.
    xticks_manual :  [] or None
        If input, the xticks do not use the array's default xticks but instead overwrite them as these values.
    yticks_manual :  [] or None
        If input, the yticks do not use the array's default yticks but instead overwrite them as these values.
    """
    return np.asarray([0, array.shape[1], 0, array.shape[0]])


def get_normalization_min_max(array, norm_min, norm_max):
    """Get the minimum and maximum of the normalization of the array, which sets the lower and upper limits of the \
    colormap.

    If norm_min / norm_max are not supplied, the minimum / maximum values of the array of hyper are used.

    Parameters
    -----------
    array : ndarray or hyper.array.scaled_array.ScaledArray
        The 2D array of hyper which is plotted.
    norm_min : float or None
        The minimum array value the colormap map spans (all values below this value are plotted the same color).
    norm_max : float or None
        The maximum array value the colormap map spans (all values above this value are plotted the same color).
    """
    if norm_min is None:
        norm_min = array.min()
    if norm_max is None:
        norm_max = array.max()

    return norm_min, norm_max


def get_normalization_scale(norm, norm_min, norm_max, linthresh, linscale):
    """Get the normalization scale of the colormap. This will be scaled based on the input min / max normalization \
    values.

    For a 'symmetric_log' colormap, linthesh and linscale also change the colormap.

    If norm_min / norm_max are not supplied, the minimum / maximum values of the array of hyper are used.

    Parameters
    -----------
    array : ndarray or hyper.array.scaled_array.ScaledArray
        The 2D array of hyper which is plotted.
    norm_min : float or None
        The minimum array value the colormap map spans (all values below this value are plotted the same color).
    norm_max : float or None
        The maximum array value the colormap map spans (all values above this value are plotted the same color).
    linthresh : float
        For the 'symmetric_log' colormap normalization ,this specifies the range of values within which the colormap \
        is linear.
    linscale : float
        For the 'symmetric_log' colormap normalization, this allowws the linear range set by linthresh to be stretched \
        relative to the logarithmic range.
    """
    if norm is 'linear':
        return colors.Normalize(vmin=norm_min, vmax=norm_max)
    elif norm is 'log':
        if norm_min == 0.0:
            norm_min = 1.e-4
        return colors.LogNorm(vmin=norm_min, vmax=norm_max)
    elif norm is 'symmetric_log':
        return colors.SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=norm_min, vmax=norm_max)
    else:
        raise exc.PlottingException('The normalization (norm) supplied to the plotter is not a valid string (must be '
                                    'linear | log | symmetric_log')


def set_xy_labels_and_ticksize(xlabelsize, ylabelsize, xyticksize):
    """Set the x and y labels of the figure, and set the fontsize of those labels.

    The x and y labels are always the distance scales, thus the labels are either arc-seconds or kpc and depend on the \
    units the figure is plotted in.

    Parameters
    -----------
    units : str
        The units of the y / x axis of the plots, in arc-seconds ('arcsec') or kiloparsecs ('kpc').
    kpc_per_arcsec : float
        The conversion factor between arc-seconds and kiloparsecs, required to plot the units in kpc.
    xlabelsize : int
        The fontsize of the x axes label.
    ylabelsize : int
        The fontsize of the y axes label.
    xyticksize : int
        The font size of the x and y ticks on the figure axes.
    """
    plt.xlabel('x (pixels)', fontsize=xlabelsize)
    plt.ylabel('y (pixels)', fontsize=ylabelsize)
    plt.tick_params(labelsize=xyticksize)


def set_colorbar(cb_ticksize, cb_fraction, cb_pad, cb_tick_values, cb_tick_labels):
    """Setup the colorbar of the figure, specifically its ticksize and the size is appears relative to the figure.

    Parameters
    -----------
    cb_ticksize : int
        The size of the tick labels on the colorbar.
    cb_fraction : float
        The fraction of the figure that the colorbar takes up, which resizes the colorbar relative to the figure.
    cb_pad : float
        Pads the color bar in the figure, which resizes the colorbar relative to the figure.
    cb_tick_values : [float]
        Manually specified values of where the colorbar tick labels appear on the colorbar.
    cb_tick_labels : [float]
        Manually specified labels of the color bar tick labels, which appear where specified by cb_tick_values.
    """

    if cb_tick_values is None and cb_tick_labels is None:
        cb = plt.colorbar(fraction=cb_fraction, pad=cb_pad)
    elif cb_tick_values is not None and cb_tick_labels is not None:
        cb = plt.colorbar(fraction=cb_fraction, pad=cb_pad, ticks=cb_tick_values)
        cb.ax.set_yticklabels(cb_tick_labels)
    else:
        raise exc.PlottingException('Only 1 entry of cb_tick_values or cb_tick_labels was input. You must either supply'
                                    'both the values and labels, or neither.')

    cb.ax.tick_params(labelsize=cb_ticksize)


def convert_grid_units(array, grid_arc_seconds, units, kpc_per_arcsec):
    """Convert the grid from its input units (arc-seconds) to the input unit (e.g. retain arc-seconds) or convert to \
    another set of units (pixels or kilo parsecs).

    Parameters
    -----------
    array : ndarray or hyper.array.scaled_array.ScaledArray
        The 2D array of hyper which is plotted, the shape of which is used for converting the grid to units of pixels.
    grid_arc_seconds : ndarray or hyper.array.grid_stacks.RegularGrid
        The (y,x) coordinates of the grid in arc-seconds, in an array of shape (total_coordinates, 2).
    units : str
        The units of the y / x axis of the plots, in arc-seconds ('arcsec') or kiloparsecs ('kpc').
    kpc_per_arcsec : float
        The conversion factor between arc-seconds and kiloparsecs, required to plot the units in kpc.
    """
    if units is 'pixels':
        return array.grid_arc_seconds_to_grid_pixels(grid_arc_seconds=grid_arc_seconds)
    elif units is 'arcsec' or kpc_per_arcsec is None:
        return grid_arc_seconds
    elif units is 'kpc':
        return grid_arc_seconds * kpc_per_arcsec