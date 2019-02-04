from autocti.data.plotters import array_plotters


def plot_image(image, mask=None, as_subplot=False,
               figsize=(7, 7), aspect='equal',
               cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
               cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
               title='Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
               output_path=None, output_format='show', output_filename='image'):
    """Plot the observed image of the ccd data.

    Set *autocti.data.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : CIFrame
        The image of the data.
    """
    array_plotters.plot_array(array=image, mask=mask, as_subplot=as_subplot,
                              figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_noise_map(noise_map, mask=None, as_subplot=False,
                   figsize=(7, 7), aspect='equal',
                   cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                   title='Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                   output_path=None, output_format='show', output_filename='noise_map'):
    """Plot the observed noise_map of the ccd data.

    Set *autocti.data.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    noise_map : CIFrame
        The noise map of the data.
    """
    array_plotters.plot_array(array=noise_map, mask=mask, as_subplot=as_subplot,
                              figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_ci_pre_cti(ci_pre_cti, mask=None, as_subplot=False,
                    figsize=(7, 7), aspect='equal',
                    cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                    cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                    title='Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                    output_path=None, output_format='show', output_filename='ci_pre_cti'):
    """Plot the observed ci_pre_cti of the ccd data.

    Set *autocti.data.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    ci_pre_cti : CIFrame
        The ci_pre_cti of the data.
    """
    array_plotters.plot_array(array=ci_pre_cti, mask=mask, as_subplot=as_subplot,
                              figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_signal_to_noise_map(signal_to_noise_map, mask=None, as_subplot=False,
                             figsize=(7, 7), aspect='equal',
                             cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                             cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                             title='Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                             output_path=None, output_format='show', output_filename='signal_to_noise_map'):
    """Plot the observed signal_to_noise_map of the ccd data.

    Set *autocti.data.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    signal_to_noise_map : CIFrame
        The signal-to-noise map of the data.
    """
    array_plotters.plot_array(array=signal_to_noise_map, mask=mask, as_subplot=as_subplot,
                              figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)
