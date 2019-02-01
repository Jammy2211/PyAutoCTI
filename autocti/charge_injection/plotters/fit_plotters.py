import numpy as np

from autocti.data.plotters import array_plotters


def plot_image(fit, fit_index, mask=None, as_subplot=False,
               figsize=(7, 7), aspect='equal',
               cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
               cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
               title='Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
               output_path=None, output_format='show', output_filename='fit_image'):
    """Plot the observed image of the ccd data.

    Set *autocti.data.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : CIFrame
        The image of the data.
    """

    masked_image = np.add(fit.image, 0.0, out=np.zeros_like(fit.image),
                          where=np.asarray(fit.mask) == 0)

    array_plotters.plot_array(array=masked_image, mask=mask, as_subplot=as_subplot,
                              figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)


def get_mask(fit, should_plot_mask):
    """Get the masks of the fit if the masks should be plotted on the fit.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractLensHyperFit
        The fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
    should_plot_mask : bool
        If *True*, the masks is plotted on the fit's datas.
    """
    if should_plot_mask:
        return fit.mask
    else:
        return None
