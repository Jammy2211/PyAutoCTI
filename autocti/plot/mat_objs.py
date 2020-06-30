import matplotlib
from autoconf import conf

backend = conf.get_matplotlib_backend()

if not backend in "default":
    matplotlib.use(backend)

if conf.instance.general.get("hpc", "hpc_mode", bool):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
import inspect
import os

from autocti import exc


def load_setting(section, name, python_type, from_subplot_config):

    if not from_subplot_config:
        return load_figure_setting(section, name, python_type)
    else:
        return load_subplot_setting(section, name, python_type)


def load_figure_setting(section, name, python_type):
    return conf.instance.visualize_figures.get(section, name, python_type)


def load_subplot_setting(section, name, python_type):
    return conf.instance.visualize_subplots.get(section, name, python_type)


class Units:
    def __init__(self, use_scaled=None, conversion_factor=None, in_kpc=None):

        self.use_scaled = use_scaled
        self.conversion_factor = conversion_factor
        self.in_kpc = in_kpc

        if use_scaled is not None:
            self.use_scaled = use_scaled
        else:
            try:
                self.use_scaled = conf.instance.visualize_general.get(
                    "general", "use_scaled", bool
                )
            except:
                self.use_scaled = True

        try:
            self.in_kpc = (
                in_kpc
                if in_kpc is not None
                else conf.instance.visualize_general.get("units", "in_kpc", bool)
            )
        except:
            self.in_kpc = None


class Figure:
    def __init__(self, figsize=None, aspect=None, from_subplot_config=False):

        self.from_subplot_config = from_subplot_config

        self.figsize = figsize
        self.aspect = aspect

        self.figsize = (
            figsize
            if figsize is not None
            else load_setting("figures", "figsize", str, from_subplot_config)
        )
        if self.figsize == "auto":
            self.figsize = None
        elif isinstance(self.figsize, str):
            self.figsize = tuple(map(int, self.figsize[1:-1].split(",")))

        self.aspect = (
            aspect
            if aspect is not None
            else load_setting("figures", "aspect", str, from_subplot_config)
        )

    @classmethod
    def sub(cls, figsize=None, aspect=None):
        return Figure(figsize=figsize, aspect=aspect, from_subplot_config=True)

    def aspect_from_shape_2d(self, shape_2d):

        if self.aspect in "square":
            return float(shape_2d[1]) / float(shape_2d[0])
        else:
            return self.aspect

    def open(self):
        if not plt.fignum_exists(num=1):
            plt.figure(figsize=self.figsize)

    def close(self):
        if plt.fignum_exists(num=1):
            plt.close()


class ColorMap:
    def __init__(
        self,
        cmap=None,
        norm=None,
        norm_max=None,
        norm_min=None,
        linthresh=None,
        linscale=None,
        from_subplot_config=False,
    ):
        self.from_subplot_config = from_subplot_config

        self.cmap = (
            cmap
            if cmap is not None
            else load_setting("colormap", "cmap", str, from_subplot_config)
        )
        self.norm = (
            norm
            if norm is not None
            else load_setting("colormap", "norm", str, from_subplot_config)
        )
        self.norm_min = (
            norm_min
            if norm_min is not None
            else load_setting("colormap", "norm_min", float, from_subplot_config)
        )
        self.norm_max = (
            norm_max
            if norm_max is not None
            else load_setting("colormap", "norm_max", float, from_subplot_config)
        )
        self.linthresh = (
            linthresh
            if linthresh is not None
            else load_setting("colormap", "linthresh", float, from_subplot_config)
        )
        self.linscale = (
            linscale
            if linscale is not None
            else load_setting("colormap", "linscale", float, from_subplot_config)
        )

    @classmethod
    def sub(
        cls,
        cmap=None,
        norm=None,
        norm_max=None,
        norm_min=None,
        linthresh=None,
        linscale=None,
    ):
        return ColorMap(
            cmap=cmap,
            norm=norm,
            norm_min=norm_min,
            norm_max=norm_max,
            linthresh=linthresh,
            linscale=linscale,
            from_subplot_config=True,
        )

    def norm_from_array(self, array):
        """Get the normalization scale of the colormap. This will be hyper based on the input min / max normalization \
        values.

        For a 'symmetric_log' colormap, linthesh and linscale also change the colormap.

        If norm_min / norm_max are not supplied, the minimum / maximum values of the array of data_type are used.

        Parameters
        -----------
        array : data_type.array.aa.Scaled
            The 2D array of data_type which is plotted.
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

        if self.norm_min is None:
            norm_min = array.min()
        else:
            norm_min = self.norm_min

        if self.norm_max is None:
            norm_max = array.max()
        else:
            norm_max = self.norm_max

        if self.norm in "linear":
            return colors.Normalize(vmin=norm_min, vmax=norm_max)
        elif self.norm in "log":
            if norm_min == 0.0:
                norm_min = 1.0e-4
            return colors.LogNorm(vmin=norm_min, vmax=norm_max)
        elif self.norm in "symmetric_log":
            return colors.SymLogNorm(
                linthresh=self.linthresh,
                linscale=self.linscale,
                vmin=norm_min,
                vmax=norm_max,
            )
        else:
            raise exc.PlottingException(
                "The normalization (norm) supplied to the plotter is not a valid string (must be "
                "linear | log | symmetric_log"
            )


class ColorBar:
    def __init__(
        self,
        ticksize=None,
        fraction=None,
        pad=None,
        tick_values=None,
        tick_labels=None,
        from_subplot_config=False,
    ):

        self.from_subplot_config = from_subplot_config

        self.ticksize = (
            ticksize
            if ticksize is not None
            else load_setting("colorbar", "ticksize", int, from_subplot_config)
        )

        self.fraction = (
            fraction
            if fraction is not None
            else load_setting("colorbar", "fraction", float, from_subplot_config)
        )

        self.pad = (
            pad
            if pad is not None
            else load_setting("colorbar", "pad", float, from_subplot_config)
        )

        self.tick_values = tick_values
        self.tick_labels = tick_labels

    @classmethod
    def sub(
        cls, ticksize=None, fraction=None, pad=None, tick_values=None, tick_labels=None
    ):
        return ColorBar(
            ticksize=ticksize,
            fraction=fraction,
            pad=pad,
            tick_values=tick_values,
            tick_labels=tick_labels,
            from_subplot_config=True,
        )

    def set(self):
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

        if self.tick_values is None and self.tick_labels is None:
            cb = plt.colorbar(fraction=self.fraction, pad=self.pad)
        elif self.tick_values is not None and self.tick_labels is not None:
            cb = plt.colorbar(
                fraction=self.fraction, pad=self.pad, ticks=self.tick_values
            )
            cb.ax.set_yticklabels(labels=self.tick_labels)
        else:
            raise exc.PlottingException(
                "Only 1 entry of tick_values or tick_labels was input. You must either supply"
                "both the values and labels, or neither."
            )

        cb.ax.tick_params(labelsize=self.ticksize)

    def set_with_values(self, cmap, color_values):

        cax = cm.ScalarMappable(cmap=cmap)
        cax.set_array(color_values)

        if self.tick_values is None and self.tick_labels is None:
            plt.colorbar(mappable=cax, fraction=self.fraction, pad=self.pad)
        elif self.tick_values is not None and self.tick_labels is not None:
            cb = plt.colorbar(
                mappable=cax,
                fraction=self.fraction,
                pad=self.pad,
                ticks=self.tick_values,
            )
            cb.ax.set_yticklabels(self.tick_labels)


class Ticks:
    def __init__(
        self,
        ysize=None,
        xsize=None,
        y_manual=None,
        x_manual=None,
        from_subplot_config=False,
    ):

        self.from_subplot_config = from_subplot_config

        self.ysize = (
            ysize
            if ysize is not None
            else load_setting("ticks", "ysize", int, from_subplot_config)
        )

        self.xsize = (
            xsize
            if xsize is not None
            else load_setting("ticks", "xsize", int, from_subplot_config)
        )

        self.y_manual = y_manual
        self.x_manual = x_manual

    @classmethod
    def sub(cls, ysize=None, xsize=None, y_manual=None, x_manual=None):
        return Ticks(
            ysize=ysize,
            xsize=xsize,
            y_manual=y_manual,
            x_manual=x_manual,
            from_subplot_config=True,
        )

    def set_yticks(self, array, extent, units, symmetric_around_centre=False):
        """Get the extent of the dimensions of the array in the unit_label of the figure (e.g. arc-seconds or kpc).

        This is used to set the extent of the array and thus the y / x axis limits.

        Parameters
        -----------
        array : data_type.array.aa.Scaled
            The 2D array of data_type which is plotted.
        unit_label : str
            The label for the unit_label of the y / x axis of the plots.
        unit_conversion_factor : float
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        xticks_manual :  [] or None
            If input, the xticks do not use the array's default xticks but instead overwrite them as these values.
        yticks_manual :  [] or None
            If input, the yticks do not use the array's default yticks but instead overwrite them as these values.
        """

        plt.tick_params(labelsize=self.ysize)

        if symmetric_around_centre:
            return

        yticks = np.linspace(extent[2], extent[3], 5)

        if self.y_manual is not None:
            ytick_labels = np.asarray([self.y_manual[0], self.y_manual[3]])
        elif not units.use_scaled:
            ytick_labels = np.linspace(0, array.shape_2d[0], 5).astype("int")
        elif units.use_scaled and units.conversion_factor is None:
            ytick_labels = np.round(np.linspace(extent[2], extent[3], 5), 2)
        elif units.use_scaled and units.conversion_factor is not None:
            ytick_labels = np.round(
                np.linspace(
                    extent[2] * units.conversion_factor,
                    extent[3] * units.conversion_factor,
                    5,
                ),
                2,
            )

        else:
            raise exc.PlottingException(
                "The y and y ticks cannot be set using the input options."
            )

        plt.yticks(ticks=yticks, labels=ytick_labels)

    def set_xticks(self, array, extent, units, symmetric_around_centre=False):
        """Get the extent of the dimensions of the array in the unit_label of the figure (e.g. arc-seconds or kpc).

        This is used to set the extent of the array and thus the y / x axis limits.

        Parameters
        -----------
        array : data_type.array.aa.Scaled
            The 2D array of data_type which is plotted.
        unit_label : str
            The label for the unit_label of the y / x axis of the plots.
        unit_conversion_factor : float
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        xticks_manual :  [] or None
            If input, the xticks do not use the array's default xticks but instead overwrite them as these values.
        xticks_manual :  [] or None
            If input, the xticks do not use the array's default xticks but instead overwrite them as these values.
        """

        plt.tick_params(labelsize=self.xsize)

        if symmetric_around_centre:
            return

        xticks = np.linspace(extent[0], extent[1], 5)

        if self.x_manual is not None:
            xtick_labels = np.asarray([self.x_manual[0], self.x_manual[3]])
        elif not units.use_scaled:
            xtick_labels = np.linspace(0, array.shape_2d[0], 5).astype("int")
        elif units.use_scaled and units.conversion_factor is None:
            xtick_labels = np.round(np.linspace(extent[0], extent[1], 5), 2)
        elif units.use_scaled and units.conversion_factor is not None:
            xtick_labels = np.round(
                np.linspace(
                    extent[0] * units.conversion_factor,
                    extent[1] * units.conversion_factor,
                    5,
                ),
                2,
            )

        else:
            raise exc.PlottingException(
                "The y and y ticks cannot be set using the input options."
            )

        plt.xticks(ticks=xticks, labels=xtick_labels)


class Labels:
    def __init__(
        self,
        title=None,
        yunits=None,
        xunits=None,
        titlesize=None,
        ysize=None,
        xsize=None,
        from_subplot_config=False,
    ):

        self.from_subplot_config = from_subplot_config

        self.title = title
        self._yunits = yunits
        self._xunits = xunits

        self.titlesize = (
            titlesize
            if titlesize is not None
            else load_setting("labels", "titlesize", int, from_subplot_config)
        )

        self.ysize = (
            ysize
            if ysize is not None
            else load_setting("labels", "ysize", int, from_subplot_config)
        )

        self.xsize = (
            xsize
            if xsize is not None
            else load_setting("labels", "xsize", int, from_subplot_config)
        )

    @classmethod
    def sub(
        cls,
        title=None,
        yunits=None,
        xunits=None,
        titlesize=None,
        ysize=None,
        xsize=None,
    ):
        return Labels(
            title=title,
            yunits=yunits,
            xunits=xunits,
            titlesize=titlesize,
            ysize=ysize,
            xsize=xsize,
            from_subplot_config=True,
        )

    def title_from_func(self, func):
        if self.title is None:

            return func.__name__.capitalize()

        else:

            return self.title

    def yunits_from_func(self, func):

        if self._yunits is None:

            args = inspect.getfullargspec(func).args
            defaults = inspect.getfullargspec(func).defaults

            if defaults is not None:
                non_default_args = len(args) - len(defaults)
            else:
                non_default_args = 0

            if "label_yunits" in args:
                return defaults[args.index("label_yunits") - non_default_args]
            else:
                return None

        else:

            return self._yunits

    def xunits_from_func(self, func):

        if self._xunits is None:

            args = inspect.getfullargspec(func).args
            defaults = inspect.getfullargspec(func).defaults

            if defaults is not None:
                non_default_args = len(args) - len(defaults)
            else:
                non_default_args = 0

            if "label_xunits" in args:
                return defaults[args.index("label_xunits") - non_default_args]
            else:
                return None

        else:

            return self._xunits

    def yunits_from_units(self, units):

        if self._yunits is None:

            if units.in_kpc is not None:
                if units.in_kpc:
                    return "kpc"
                else:
                    return "arcsec"

            if units.use_scaled:
                return "scaled"
            else:
                return "pixels"

        else:

            return self._yunits

    def xunits_from_units(self, units):

        if self._xunits is None:

            if units.in_kpc is not None:
                if units.in_kpc:
                    return "kpc"
                else:
                    return "arcsec"

            if units.use_scaled:
                return "scaled"
            else:
                return "pixels"

        else:

            return self._xunits

    def set_title(self):
        """Set the title and title size of the figure.

        Parameters
        -----------
        title : str
            The text of the title.
        titlesize : int
            The size of of the title of the figure.
        """
        plt.title(label=self.title, fontsize=self.titlesize)

    def set_yunits(self, units, include_brackets):
        """Set the x and y labels of the figure, and set the fontsize of those self.label_

        The x and y labels are always the distance scales, thus the labels are either arc-seconds or kpc and depend on the \
        unit_label the figure is plotted in.

        Parameters
        -----------
        unit_label : str
            The label for the unit_label of the y / x axis of the plots.
        unit_conversion_factor : float
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        xsize : int
            The fontsize of the x axes label.
        ysize : int
            The fontsize of the y axes label.
        xyticksize : int
            The font size of the x and y ticks on the figure axes.
        """
        if include_brackets:
            plt.ylabel(
                "y (" + self.yunits_from_units(units=units) + ")", fontsize=self.ysize
            )
        else:
            plt.ylabel(self.yunits_from_units(units=units), fontsize=self.ysize)

    def set_xunits(self, units, include_brackets):
        """Set the x and y labels of the figure, and set the fontsize of those self.label_

        The x and y labels are always the distance scales, thus the labels are either arc-seconds or kpc and depend on the \
        unit_label the figure is plotted in.

        Parameters
        -----------
        unit_label : str
            The label for the unit_label of the y / x axis of the plots.
        unit_conversion_factor : float
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        xsize : int
            The fontsize of the x axes label.
        ysize : int
            The fontsize of the y axes label.
        xyticksize : int
            The font size of the x and y ticks on the figure axes.
        """
        if include_brackets:
            plt.xlabel(
                "x (" + self.xunits_from_units(units=units) + ")", fontsize=self.xsize
            )
        else:
            plt.xlabel(self.xunits_from_units(units=units), fontsize=self.xsize)


class Legend:
    def __init__(self, include=None, fontsize=None, from_subplot_config=False):

        self.from_subplot_config = from_subplot_config

        self.include = (
            include
            if include is not None
            else load_setting("legend", "include", bool, from_subplot_config)
        )

        self.fontsize = (
            fontsize
            if fontsize is not None
            else load_setting("legend", "fontsize", int, from_subplot_config)
        )

    @classmethod
    def sub(cls, include=None, fontsize=None):
        return Legend(include=include, fontsize=fontsize, from_subplot_config=True)

    def set(self):
        if self.include:
            plt.legend(fontsize=self.fontsize)


class Output:
    def __init__(self, path=None, filename=None, format=None, bypass=False):

        self.path = path

        if path is not None and path:
            try:
                os.makedirs(path)
            except FileExistsError:
                pass

        self.filename = filename
        self._format = format
        self.bypass = bypass

    @property
    def format(self):
        if self._format is None:
            return "show"
        else:
            return self._format

    def filename_from_func(self, func):

        if self.filename is None:
            return func.__name__
        else:

            return self.filename

    def to_figure(self, structure):
        """Output the figure, either as an image on the screen or to the hard-disk as a .png or .fits file.

        Parameters
        -----------
        structure : ndarray
            The 2D array of image to be output, required for outputting the image as a fits file.
        as_subplot : bool
            Whether the figure is part of subplot, in which case the figure is not output so that the entire subplot can \
            be output instead using the *output.output_figure(structure=None, is_sub_plotter=False)* function.
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
        if not self.bypass:
            if self.format is "show":
                plt.show()
            elif self.format is "png":
                plt.savefig(self.path + self.filename + ".png", bbox_inches="tight")
            elif self.format is "fits":
                if structure is not None:
                    structure.output_to_fits(
                        file_path=self.path + self.filename + ".fits", overwrite=True
                    )

    def subplot_to_figure(self):
        """Output the figure, either as an image on the screen or to the hard-disk as a .png or .fits file.

        Parameters
        -----------
        structure : ndarray
            The 2D array of image to be output, required for outputting the image as a fits file.
        as_subplot : bool
            Whether the figure is part of subplot, in which case the figure is not output so that the entire subplot can \
            be output instead using the *output.output_figure(structure=None, is_sub_plotter=False)* function.
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
        if self.format is "show":
            plt.show()
        elif self.format is "png":
            plt.savefig(self.path + self.filename + ".png", bbox_inches="tight")


def remove_spaces_and_commas_from_colors(colors):

    colors = [color.strip(",") for color in colors]
    colors = [color.strip(" ") for color in colors]
    return list(filter(None, colors))


class Scatterer:
    def __init__(
        self,
        size=None,
        marker=None,
        colors=None,
        section=None,
        from_subplot_config=False,
    ):

        self.from_subplot_config = from_subplot_config

        self.size = (
            size
            if size is not None
            else load_setting(section, "size", int, from_subplot_config)
        )

        self.marker = (
            marker
            if marker is not None
            else load_setting(section, "marker", str, from_subplot_config)
        )

        self.colors = (
            colors
            if colors is not None
            else load_setting(section, "colors", list, from_subplot_config)
        )

        self.colors = remove_spaces_and_commas_from_colors(colors=self.colors)

        if isinstance(self.colors, str):
            self.colors = [self.colors]

    def scatter_grids(self, grids):

        plt.scatter(
            y=np.asarray(grids)[:, 0],
            x=np.asarray(grids)[:, 1],
            s=self.size,
            c=self.colors[0],
            marker=self.marker,
        )


class OriginScatterer(Scatterer):
    def __init__(self, size=None, marker=None, colors=None, from_subplot_config=False):

        super(OriginScatterer, self).__init__(
            size=size,
            marker=marker,
            colors=colors,
            section="origin",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, size=None, marker=None, colors=None):
        return OriginScatterer(
            size=size, marker=marker, colors=colors, from_subplot_config=True
        )


class Liner:
    def __init__(
        self,
        width=None,
        style=None,
        colors=None,
        pointsize=None,
        section=None,
        from_subplot_config=False,
    ):

        if section is None:
            section = "liner"

        self.from_subplot_config = from_subplot_config

        self.width = (
            width
            if width is not None
            else load_setting(section, "width", int, from_subplot_config)
        )

        self.style = (
            style
            if style is not None
            else load_setting(section, "style", str, from_subplot_config)
        )

        self.colors = (
            colors
            if colors is not None
            else load_setting(section, "colors", list, from_subplot_config)
        )

        self.colors = remove_spaces_and_commas_from_colors(colors=self.colors)

        self.pointsize = (
            pointsize
            if pointsize is not None
            else load_setting(section, "pointsize", int, from_subplot_config)
        )

    @classmethod
    def sub(cls, width=None, style=None, colors=None, pointsize=None, section=None):
        return Liner(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            section=section,
            from_subplot_config=True,
        )

    def draw_y_vs_x(self, y, x, plot_axis_type, label=None):

        if plot_axis_type is "linear":
            plt.plot(x, y, c=self.colors[0], lw=self.width, ls=self.style, label=label)
        elif plot_axis_type is "semilogy":
            plt.semilogy(
                x, y, c=self.colors[0], lw=self.width, ls=self.style, label=label
            )
        elif plot_axis_type is "loglog":
            plt.loglog(
                x, y, c=self.colors[0], lw=self.width, ls=self.style, label=label
            )
        elif plot_axis_type is "scatter":
            plt.scatter(x, y, c=self.colors[0], s=self.pointsize, label=label)
        else:
            raise exc.PlottingException(
                "The plot_axis_type supplied to the plotter is not a valid string (must be linear "
                "| semilogy | loglog)"
            )

    def draw_vertical_lines(self, vertical_lines, vertical_line_labels=None):

        if vertical_lines is [] or vertical_lines is None:
            return

        if vertical_line_labels is None:
            vertical_line_labels = [None for i in range(len(vertical_lines))]

        for vertical_line, vertical_line_label in zip(
            vertical_lines, vertical_line_labels
        ):

            plt.axvline(
                x=vertical_line,
                label=vertical_line_label,
                c=self.colors[0],
                lw=self.width,
                ls=self.style,
            )

    def draw_rectangular_grid_lines(self, extent, shape_2d):

        ys = np.linspace(extent[2], extent[3], shape_2d[1] + 1)
        xs = np.linspace(extent[0], extent[1], shape_2d[0] + 1)

        # grid lines
        for x in xs:
            plt.plot(
                [x, x],
                [ys[0], ys[-1]],
                color=self.colors[0],
                lw=self.width,
                ls=self.style,
            )
        for y in ys:
            plt.plot(
                [xs[0], xs[-1]],
                [y, y],
                color=self.colors[0],
                lw=self.width,
                ls=self.style,
            )


class ParallelOverscanLiner(Liner):
    def __init__(
        self,
        width=None,
        style=None,
        colors=None,
        pointsize=None,
        from_subplot_config=False,
    ):

        super(ParallelOverscanLiner, self).__init__(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            section="parallel_overscan",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, width=None, style=None, colors=None, pointsize=None):
        return ParallelOverscanLiner(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            from_subplot_config=True,
        )


class SerialPrescanLiner(Liner):
    def __init__(
        self,
        width=None,
        style=None,
        colors=None,
        pointsize=None,
        from_subplot_config=False,
    ):

        super(SerialPrescanLiner, self).__init__(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            section="serial_prescan",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, width=None, style=None, colors=None, pointsize=None):
        return SerialPrescanLiner(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            from_subplot_config=True,
        )


class SerialOverscanLiner(Liner):
    def __init__(
        self,
        width=None,
        style=None,
        colors=None,
        pointsize=None,
        from_subplot_config=False,
    ):

        super(SerialOverscanLiner, self).__init__(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            section="serial_overscan",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, width=None, style=None, colors=None, pointsize=None):
        return SerialOverscanLiner(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            from_subplot_config=True,
        )
