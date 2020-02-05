import autofit as af
from autoarray.plot import plotters, mat_objs
from autocti.plot import cti_mat_objs
from autocti import exc

import matplotlib.pyplot as plt
from functools import wraps
import copy
import numpy as np


class CTIPlotter(object):
    def __init__(
        self,
        units=None,
        figure=None,
        cmap=None,
        cb=None,
        legend=None,
        ticks=None,
        labels=None,
        output=None,
        origin_scatterer=None,
        liner=None,
        parallel_overscan_liner=None,
        serial_prescan_liner=None,
        serial_overscan_liner=None,
    ):

        if isinstance(self, Plotter):
            from_subplot_config = False
        else:
            from_subplot_config = True

        self.units = units if units is not None else mat_objs.Units()

        self.figure = (
            figure
            if figure is not None
            else mat_objs.Figure(from_subplot_config=from_subplot_config)
        )

        self.cmap = (
            cmap
            if cmap is not None
            else mat_objs.ColorMap(from_subplot_config=from_subplot_config)
        )

        self.cb = (
            cb
            if cb is not None
            else mat_objs.ColorBar(from_subplot_config=from_subplot_config)
        )

        self.ticks = (
            ticks
            if ticks is not None
            else mat_objs.Ticks(from_subplot_config=from_subplot_config)
        )

        self.labels = (
            labels
            if labels is not None
            else mat_objs.Labels(from_subplot_config=from_subplot_config)
        )

        self.legend = (
            legend
            if legend is not None
            else mat_objs.Legend(from_subplot_config=from_subplot_config)
        )

        self.output = (
            output
            if output is not None
            else mat_objs.Output(bypass=isinstance(self, SubPlotter))
        )

        self.liner = (
            liner
            if liner is not None
            else mat_objs.Liner(
                section="liner", from_subplot_config=from_subplot_config
            )
        )

        self.origin_scatterer = (
            origin_scatterer
            if origin_scatterer is not None
            else mat_objs.OriginScatterer(from_subplot_config=from_subplot_config)
        )

        self.parallel_overscan_liner = (
            parallel_overscan_liner
            if parallel_overscan_liner is not None
            else cti_mat_objs.ParallelOverscanLiner(
                from_subplot_config=from_subplot_config
            )
        )

        self.serial_prescan_liner = (
            serial_prescan_liner
            if serial_prescan_liner is not None
            else cti_mat_objs.SerialPrescanLiner(
                from_subplot_config=from_subplot_config
            )
        )

        self.serial_overscan_liner = (
            serial_overscan_liner
            if serial_overscan_liner is not None
            else cti_mat_objs.SerialOverscanLiner(
                from_subplot_config=from_subplot_config
            )
        )

    def plot_frame(
        self,
        frame,
        lines=None,
        include_origin=False,
        include_parallel_overscan=False,
        include_serial_prescan=False,
        include_serial_overscan=False,
        bypass_output=False,
    ):
        """Plot an array of data_type as a figure.

        Parameters
        -----------
        settings : PlotterSettings
            Settings
        include : PlotterInclude
            Include
        labels : PlotterLabels
            labels
        outputs : PlotterOutputs
            outputs
        array : data_type.array.aa.Scaled
            The 2D array of data_type which is plotted.
        origin : (float, float).
            The origin of the coordinate system of the array, which is plotted as an 'x' on the image if input.
        mask : data_type.array.mask.Mask
            The mask applied to the array, the edge of which is plotted as a set of points over the plotted array.
        extract_array_from_mask : bool
            The plotter array is extracted using the mask, such that masked values are plotted as zeros. This ensures \
            bright features outside the mask do not impact the color map of the plotters.
        zoom_around_mask : bool
            If True, the 2D region of the array corresponding to the rectangle encompassing all unmasked values is \
            plotted, thereby zooming into the region of interest.
        border : bool
            If a mask is supplied, its borders pixels (e.g. the exterior edge) is plotted if this is *True*.
        positions : [[]]
            Lists of (y,x) coordinates on the image which are plotted as colored dots, to highlight specific pixels.
        grid : data_type.array.aa.Grid
            A grid of (y,x) coordinates which may be plotted over the plotted array.
        as_subplot : bool
            Whether the array is plotted as part of a subplot, in which case the grid figure is not opened / closed.
        unit_label : str
            The label for the unit_label of the y / x axis of the plots.
        unit_conversion_factor : float or None
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        figsize : (int, int)
            The size of the figure in (rows, columns).
        aspect : str
            The aspect ratio of the array, specifically whether it is forced to be square ('equal') or adapts its size to \
            the figure size ('auto').
        cmap : str
            The colormap the array is plotted using, which may be chosen from the standard matplotlib colormaps.
        norm : str
            The normalization of the colormap used to plotters the image, specifically whether it is linear ('linear'), log \
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
        xsize : int
            The fontsize of the x axes label.
        ysize : int
            The fontsize of the y axes label.
        xyticksize : int
            The font size of the x and y ticks on the figure axes.
        mask_scatterer : int
            The size of the points plotted to show the mask.
        xticks_manual :  [] or None
            If input, the xticks do not use the array's default xticks but instead overwrite them as these values.
        yticks_manual :  [] or None
            If input, the yticks do not use the array's default yticks but instead overwrite them as these values.
        output_path : str
            The path on the hard-disk where the figure is output.
        output_filename : str
            The filename of the figure that is output.
        output_format : str
            The format the figue is output:
            'show' - display on computer screen.
            'png' - output to hard-disk as a png.
            'fits' - output to hard-disk as a fits file.'

        Returns
        --------
        None

        Examples
        --------
            plotter.plot_frame(
            array=image, origin=(0.0, 0.0), mask=circular_mask,
            border=False, points=[[1.0, 1.0], [2.0, 2.0]], grid=None, as_subplot=False,
            unit_label='scaled', kpc_per_arcsec=None, figsize=(7,7), aspect='auto',
            cmap='jet', norm='linear, norm_min=None, norm_max=None, linthresh=None, linscale=None,
            cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
            title='Image', titlesize=16, xsize=16, ysize=16, xyticksize=16,
            mask_scatterer=10, border_pointsize=2, position_pointsize=10, grid_pointsize=10,
            xticks_manual=None, yticks_manual=None,
            output_path='/path/to/output', output_format='png', output_filename='image')
        """

        if frame is None or np.all(frame == 0):
            return

        if frame.pixel_scales is None and self.units.use_scaled:
            raise exc.FrameException(
                "You cannot plot an array using its scaled unit_label if the input array does not have "
                "a pixel scales attribute."
            )

        frame = frame.in_2d

        self.figure.open()
        aspect = self.figure.aspect_from_shape_2d(shape_2d=frame.shape_2d)
        norm_scale = self.cmap.norm_from_array(array=frame)

        plt.imshow(
            X=frame,
            aspect=aspect,
            cmap=self.cmap.cmap,
            norm=norm_scale,
            extent=frame.mask.geometry.extent,
        )

        plt.axis(frame.mask.geometry.extent)

        self.ticks.set_yticks(
            array=frame, extent=frame.mask.geometry.extent, units=self.units
        )
        self.ticks.set_xticks(
            array=frame, extent=frame.mask.geometry.extent, units=self.units
        )

        self.labels.set_title()
        self.labels.set_yunits(units=self.units, include_brackets=True)
        self.labels.set_xunits(units=self.units, include_brackets=True)

        self.cb.set()
        if include_origin:
            self.origin_scatterer.scatter_grids(grids=[frame.origin])

        if (
            include_parallel_overscan is not None
            and frame.parallel_overscan is not None
        ):
            self.parallel_overscan_liner.draw_rectangular_grid_lines(
                extent=frame.parallel_overscan, shape_2d=frame.shape_2d
            )

        if include_serial_prescan is not None and frame.serial_prescan is not None:
            self.serial_prescan_liner.draw_rectangular_grid_lines(
                extent=frame.serial_prescan, shape_2d=frame.shape_2d
            )

        if include_serial_overscan is not None and frame.serial_overscan is not None:
            self.serial_overscan_liner.draw_rectangular_grid_lines(
                extent=frame.serial_overscan, shape_2d=frame.shape_2d
            )

        if not bypass_output:
            self.output.to_figure(structure=frame)

        if not isinstance(self, SubPlotter) and not bypass_output:
            self.figure.close()

    def plot_line(
        self,
        y,
        x,
        label=None,
        plot_axis_type="semilogy",
        vertical_lines=None,
        vertical_line_labels=None,
        bypass_output=False,
    ):

        if y is None:
            return

        self.figure.open()
        self.labels.set_title()

        if x is None:
            x = np.arange(len(y))

        self.liner.draw_y_vs_x(y=y, x=x, plot_axis_type=plot_axis_type, label=label)

        self.labels.set_yunits(units=self.units, include_brackets=False)
        self.labels.set_xunits(units=self.units, include_brackets=False)

        self.liner.draw_vertical_lines(
            vertical_lines=vertical_lines, vertical_line_labels=vertical_line_labels
        )

        if label is not None or vertical_line_labels is not None:
            self.legend.set()

        self.ticks.set_xticks(
            array=None, extent=[np.min(x), np.max(x)], units=self.units
        )

        if not bypass_output:
            self.output.to_figure(structure=None)

        if not isinstance(self, SubPlotter) and not bypass_output:
            self.figure.close()
        if not bypass_output:
            self.output.to_figure(structure=None)

        if not isinstance(self, SubPlotter) and not bypass_output:
            self.figure.close()


class Plotter(CTIPlotter, plotters.Plotter):
    def __init__(
        self,
        units=None,
        figure=None,
        cmap=None,
        cb=None,
        ticks=None,
        labels=None,
        legend=None,
        output=None,
        origin_scatterer=None,
        liner=None,
        parallel_overscan_liner=None,
        serial_prescan_liner=None,
        serial_overscan_liner=None,
    ):

        super(Plotter, self).__init__(
            units=units,
            figure=figure,
            cmap=cmap,
            cb=cb,
            legend=legend,
            ticks=ticks,
            labels=labels,
            output=output,
            origin_scatterer=origin_scatterer,
            liner=liner,
            parallel_overscan_liner=parallel_overscan_liner,
            serial_prescan_liner=serial_prescan_liner,
            serial_overscan_liner=serial_overscan_liner,
        )


class SubPlotter(CTIPlotter, plotters.SubPlotter):
    def __init__(
        self,
        units=None,
        figure=None,
        cmap=None,
        cb=None,
        legend=None,
        ticks=None,
        labels=None,
        output=None,
        origin_scatterer=None,
        liner=None,
        parallel_overscan_liner=None,
        serial_prescan_liner=None,
        serial_overscan_liner=None,
    ):

        super(SubPlotter, self).__init__(
            units=units,
            figure=figure,
            cmap=cmap,
            cb=cb,
            legend=legend,
            ticks=ticks,
            labels=labels,
            output=output,
            origin_scatterer=origin_scatterer,
            liner=liner,
            parallel_overscan_liner=parallel_overscan_liner,
            serial_prescan_liner=serial_prescan_liner,
            serial_overscan_liner=serial_overscan_liner,
        )


class Include(object):
    def __init__(
        self,
        origin=None,
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
    ):

        self.origin = self.load_include(value=origin, name="origin")
        self.parallel_overscan = self.load_include(
            value=parallel_overscan, name="parallel_overscan"
        )
        self.serial_prescan = self.load_include(
            value=serial_prescan, name="serial_prescan"
        )
        self.serial_overscan = self.load_include(
            value=serial_overscan, name="serial_overscan"
        )

    @staticmethod
    def load_include(value, name):

        return (
            af.conf.instance.visualize_general.get(
                section_name="include", attribute_name=name, attribute_type=bool
            )
            if value is None
            else value
        )

    def parallel_overscan_from_frame(self, frame):

        if self.parallel_overscan:
            return frame.parallel_overscan
        else:
            return None

    def serial_prescan_from_frame(self, frame):

        if self.serial_prescan:
            return frame.serial_prescan
        else:
            return None

    def serial_overscan_from_frame(self, frame):

        if self.serial_overscan:
            return frame.serial_overscan
        else:
            return None


def set_include_and_plotter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        include_key = plotters.include_key_from_dictionary(dictionary=kwargs)

        if include_key is not None:
            include = kwargs[include_key]
        else:
            include = Include()
            include_key = "include"

        kwargs[include_key] = include

        plotter_key = plotters.plotter_key_from_dictionary(dictionary=kwargs)

        if plotter_key is not None:
            plotter = kwargs[plotter_key]
        else:
            plotter = Plotter()
            plotter_key = "plotter"

        kwargs[plotter_key] = plotter

        return func(*args, **kwargs)

    return wrapper


def set_include_and_sub_plotter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        include_key = plotters.include_key_from_dictionary(dictionary=kwargs)

        if include_key is not None:
            include = kwargs[include_key]
        else:
            include = Include()
            include_key = "include"

        kwargs[include_key] = include

        plotter_key = plotters.plotter_key_from_dictionary(dictionary=kwargs)

        if plotter_key is not None:
            plotter = kwargs[plotter_key]
        else:
            plotter = SubPlotter()
            plotter_key = "sub_plotter"

        kwargs[plotter_key] = plotter

        return func(*args, **kwargs)

    return wrapper


def plot_frame(frame, include=None, plotter=None):

    if include is None:
        include = Include()

    if plotter is None:
        plotter = Plotter()

    plotter.plot_frame(
        frame=frame,
        include_origin=include.origin,
        include_parallel_overscan=include.parallel_overscan,
        include_serial_prescan=include.serial_prescan,
        include_serial_overscan=include.serial_overscan,
    )


def plot_line(
    y,
    x,
    label=None,
    plot_axis_type="semilogy",
    vertical_lines=None,
    vertical_line_labels=None,
    plotter=None,
):

    if plotter is None:
        plotter = Plotter()

    plotter.plot_line(
        y=y,
        x=x,
        label=label,
        plot_axis_type=plot_axis_type,
        vertical_lines=vertical_lines,
        vertical_line_labels=vertical_line_labels,
    )
