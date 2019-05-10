import matplotlib.pyplot as plt
import numpy as np

from autocti import exc
from autocti.data.plotters import plotter_util


def extracted_array_mask_and_stack_axis_from_stack_region(stack_region, array, mask, ci_frame):

    total_rows = ci_frame.ci_pattern.total_rows
    total_columns = ci_frame.ci_pattern.total_columns

    if stack_region is 'all_across_columns':
        extracted_array = array
        extracted_mask = mask if mask is not None else None
        stack_axis = 1
    elif stack_region is 'parallel_front_edge':
        extracted_array = ci_frame.parallel_front_edge_arrays_from_frame(array=array, rows=(0, total_rows))
        extracted_mask = ci_frame.parallel_front_edge_arrays_from_frame(array=mask, rows=(0, total_rows)) if mask is not None else None
        stack_axis = 1
    elif stack_region is 'all_across_rows':
        extracted_array = array
        extracted_mask = mask if mask is not None else None
        stack_axis = 0
    else:
        raise exc.PlottingException('The stack region specified for the plotting of a stack was invalid')

    return extracted_array, extracted_mask, stack_axis

def plot_extracted_stack_from_array_ci_frame_and_stack_region(
        array, ci_frame, stack_region, mask=None, as_subplot=False,
        figsize=(7, 7),
        title='Stack', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        output_path=None, output_format='show', output_filename='stack'):

    extracted_array, extracted_mask, stack_axis = extracted_array_mask_and_stack_axis_from_stack_region(
        stack_region=stack_region, array=array, mask=mask, ci_frame=ci_frame)

    plot_stack_from_array(array=extracted_array, stack_axis=stack_axis, mask=extracted_mask, as_subplot=as_subplot,
                          figsize=figsize,
                          title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                          output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_stack_from_array(array, stack_axis, mask=None, as_subplot=False,
                         figsize=(7, 7),
                         title='Stack', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                         output_path=None, output_format='show', output_filename='stack'):

    if mask is None:
        stack = np.mean(array, axis=stack_axis)
    elif mask is not None:
        masked = np.ma.array(array, mask=mask)
        stack = np.asarray(masked.mean(axis=stack_axis))

    plot_stack(stack=stack, as_subplot=as_subplot,
               figsize=figsize,
               title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
               output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_stack(stack, as_subplot=False,
               figsize=(7, 7),
               title='Stack', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
               output_path=None, output_format='show', output_filename='stack'):
    """Plot an array of hyper as a figure.

    Parameters
    -----------
    stack : ndarray or hyper.array.scaled_array.ScaledArray
        The 2D array of hyper which is plotted.
    mask : ndarray of data.mask.Mask
        The masks applied to the hyper, the edge of which is plotted as a set of points over the plotted array.
    extract_stack_from_mask : bool
        The plotter array is extracted using the mask, such that masked values are plotted as zeros. This ensures \
        bright features outside the mask do not impact the color map of the plot.
    as_subplot : bool
        Whether the array is plotted as part of a subplot, in which case the grid figure is not opened / closed.
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

    if stack is None:
        return

    plot_figure(stack=stack, as_subplot=as_subplot, figsize=figsize)

    plotter_util.set_title(title=title, titlesize=titlesize)
    set_xy_labels_and_ticksize(xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize)
    plotter_util.output_figure(stack, as_subplot=as_subplot, output_path=output_path, output_filename=output_filename,
                               output_format=output_format)
    plotter_util.close_figure(as_subplot=as_subplot)


def plot_figure(stack, as_subplot, figsize):
    """Open a matplotlib figure and plot the array of hyper on it.

    Parameters
    -----------
    stack : ndarray or hyper.array.scaled_array.ScaledArray
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
        The normalization of the colormap used to plot the hyper, specifically whether it is stackar ('stackar'), log \
        ('log') or a symmetric log normalization ('symmetric_log').
    norm_min : float or None
        The minimum array value the colormap map spans (all values below this value are plotted the same color).
    norm_max : float or None
        The maximum array value the colormap map spans (all values above this value are plotted the same color).
    linthresh : float
        For the 'symmetric_log' colormap normalization ,this specifies the range of values within which the colormap \
        is stackar.
    linscale : float
        For the 'symmetric_log' colormap normalization, this allowws the stackar range set by linthresh to be stretched \
        relative to the logarithmic range.
    xticks_manual :  [] or None
        If input, the xticks do not use the array's default xticks but instead overwrite them as these values.
    yticks_manual :  [] or None
        If input, the yticks do not use the array's default yticks but instead overwrite them as these values.
    """
    plotter_util.setup_figure(figsize=figsize, as_subplot=as_subplot)
    plt.plot(stack)

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