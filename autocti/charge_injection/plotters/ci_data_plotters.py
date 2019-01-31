from autofit import conf
from autocti.charge_injection.plotters import data_plotters

def plot_ci_data_individual(ci_data, mask=None, output_path=None, output_format='png'):
    """Plot each attribute of the ccd data as individual figures one by one (e.g. the data, noise_map-map, PSF, \
     Signal-to_noise-map, etc).

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    ci_data : data.CCDData
        The ccd data, which includes the observed data, noise_map-map, PSF, signal-to-noise_map-map, etc.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    """

    plot_data_image = conf.instance.general.get('output', 'plot_data_image', bool)
    plot_data_noise_map = conf.instance.general.get('output', 'plot_data_noise_map', bool)
    plot_data_ci_pre_cti = conf.instance.general.get('output', 'plot_data_ci_pre_cti', bool)
    plot_data_signal_to_noise_map = conf.instance.general.get('output', 'plot_data_signal_to_noise_map', bool)

    if plot_data_image:
        plot_image(ci_data=ci_data, mask=mask, output_path=output_path, output_format=output_format)

    if plot_data_noise_map:
        plot_noise_map(ci_data=ci_data, mask=mask, output_path=output_path, output_format=output_format)

    if plot_data_ci_pre_cti:
        plot_ci_pre_cti(ci_data=ci_data, output_path=output_path, output_format=output_format)

    if plot_data_signal_to_noise_map:
        plot_signal_to_noise_map(ci_data=ci_data, mask=mask, output_path=output_path, output_format=output_format)

def plot_image(ci_data, mask=None, as_subplot=False,
               figsize=(7, 7), aspect='equal',
               cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
               cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
               title='Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
               output_path=None, output_format='show', output_filename='ci_image'):
    """Plot the observed image of the ccd data.

    Set *autocti.data.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : CIFrame
        The image of the data.
    """
    data_plotters.plot_image(
        image=ci_data.image, mask=mask, as_subplot=as_subplot,
        figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
        linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_noise_map(ci_data, mask=None, as_subplot=False,
                   figsize=(7, 7), aspect='equal',
                   cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                   title='Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                   output_path=None, output_format='show', output_filename='ci_noise_map'):
    """Plot the observed noise_map of the ccd data.

    Set *autocti.data.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    noise_map : CIFrame
        The noise map of the data.
    """
    data_plotters.plot_noise_map(
        noise_map=ci_data.noise_map, mask=mask, as_subplot=as_subplot,
        figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
        linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_ci_pre_cti(ci_data, mask=None, as_subplot=False,
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
    data_plotters.plot_ci_pre_cti(
        ci_pre_cti=ci_data.ci_pre_cti, mask=mask, as_subplot=as_subplot,
        figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
        linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_signal_to_noise_map(ci_data, mask=None, as_subplot=False,
                             figsize=(7, 7), aspect='equal',
                             cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                             cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                             title='Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                             output_path=None, output_format='show', output_filename='ci_signal_to_noise_map'):
    """Plot the observed signal_to_noise_map of the ccd data.

    Set *autocti.data.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    signal_to_noise_map : CIFrame
        The signal-to-noise map of the data.
    """
    data_plotters.plot_signal_to_noise_map(
        signal_to_noise_map=ci_data.signal_to_noise_map, mask=mask,
        as_subplot=as_subplot,
       figsize=figsize, aspect=aspect,
       cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
       linthresh=linthresh, linscale=linscale,
       cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
       title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
       output_path=output_path, output_format=output_format, output_filename=output_filename)