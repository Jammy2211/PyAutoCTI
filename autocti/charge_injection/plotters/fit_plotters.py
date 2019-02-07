import numpy as np

from autocti.data.plotters import array_plotters


def plot_image(
        fit, mask=None, extract_array_from_mask=False, as_subplot=False,
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

    array_plotters.plot_array(
        array=fit.image, mask=mask, extract_array_from_mask=extract_array_from_mask, as_subplot=as_subplot,
        figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_noise_map(
        fit, mask=None, extract_array_from_mask=False, as_subplot=False,
        figsize=(7, 7), aspect='equal',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
        title='Noise-Map', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        output_path=None, output_format='show', output_filename='fit_noise_map'):
    """Plot the observed noise_map of the ccd data.

    Set *autocti.data.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    noise_map : CIFrame
        The noise_map of the data.
    """

    array_plotters.plot_array(
        array=fit.noise_map, mask=mask, extract_array_from_mask=extract_array_from_mask, as_subplot=as_subplot,
        figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_signal_to_noise_map(
        fit, mask=None, extract_array_from_mask=False, as_subplot=False,
        figsize=(7, 7), aspect='equal',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
        title='Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        output_path=None, output_format='show', output_filename='fit_signal_to_noise_map'):
    """Plot the observed signal_to_noise_map of the ccd data.

    Set *autocti.data.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    signal_to_noise_map : CIFrame
        The signal_to_noise_map of the data.
    """

    array_plotters.plot_array(
        array=fit.signal_to_noise_map, mask=mask, extract_array_from_mask=extract_array_from_mask, as_subplot=as_subplot,
        figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_ci_pre_cti(
        fit, mask=None, extract_array_from_mask=False, as_subplot=False,
        figsize=(7, 7), aspect='equal',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
        title='Pre-CTI CI Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        output_path=None, output_format='show', output_filename='fit_ci_pre_cti'):
    """Plot the observed ci_pre_cti of the ccd data.

    Set *autocti.data.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    ci_pre_cti : CIFrame
        The ci_pre_cti of the data.
    """

    array_plotters.plot_array(
        array=fit.ci_pre_cti, mask=mask, extract_array_from_mask=extract_array_from_mask, as_subplot=as_subplot,
        figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_ci_post_cti(
        fit, mask=None, extract_array_from_mask=False, as_subplot=False,
        figsize=(7, 7), aspect='equal',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
        title='Post-CTI CI Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        output_path=None, output_format='show', output_filename='fit_ci_post_cti'):
    """Plot the observed ci_post_cti of the ccd data.

    Set *autocti.data.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    ci_post_cti : CIFrame
        The ci_post_cti of the data.
    """

    array_plotters.plot_array(
        array=fit.ci_post_cti, mask=mask, extract_array_from_mask=extract_array_from_mask, as_subplot=as_subplot,
        figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_residual_map(
        fit, mask=None, extract_array_from_mask=False, as_subplot=False,
        figsize=(7, 7), aspect='equal',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
        title='Residual Map', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        output_path=None, output_format='show', output_filename='fit_residual_map'):
    """Plot the observed residual_map of the ccd data.

    Set *autocti.data.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    residual_map : CIFrame
        The residual_map of the data.
    """

    array_plotters.plot_array(
        array=fit.residual_map, mask=mask, extract_array_from_mask=extract_array_from_mask, as_subplot=as_subplot,
        figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_chi_squared_map(
        fit, mask=None, extract_array_from_mask=False, as_subplot=False,
        figsize=(7, 7), aspect='equal',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
        title='Chi-Squared Map', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        output_path=None, output_format='show', output_filename='fit_chi_squared_map'):
    """Plot the observed chi_squared_map of the ccd data.

    Set *autocti.data.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    chi_squared_map : CIFrame
        The chi_squared_map of the data.
    """

    array_plotters.plot_array(
        array=fit.chi_squared_map, mask=mask, extract_array_from_mask=extract_array_from_mask, as_subplot=as_subplot,
        figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)
