from autoconf import conf
from os import path
import matplotlib.pyplot as plt
import os
import pytest
import shutil

import numpy as np
from autocti.util import array_util
from autocti import structures as struct
import autocti.plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plotter_setup():
    return "{}/files/plotter/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "files/plotter"), path.join(directory, "output")
    )


class TestAbstractPlotterAttributes:
    def test__units__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.units.use_scaled == True
        assert plotter.units.in_kpc == False
        assert plotter.units.conversion_factor == None

        plotter = aplt.Plotter(units=aplt.Units(in_kpc=True, conversion_factor=2.0))

        assert plotter.units.use_scaled == True
        assert plotter.units.in_kpc == True
        assert plotter.units.conversion_factor == 2.0

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.units.use_scaled == True
        assert sub_plotter.units.in_kpc == False
        assert sub_plotter.units.conversion_factor == None

        sub_plotter = aplt.SubPlotter(
            units=aplt.Units(use_scaled=False, conversion_factor=2.0)
        )

        assert sub_plotter.units.use_scaled == False
        assert sub_plotter.units.in_kpc == False
        assert sub_plotter.units.conversion_factor == 2.0

    def test__figure__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.figure.figsize == (7, 7)
        assert plotter.figure.aspect == "square"

        plotter = aplt.Plotter(figure=aplt.Figure(aspect="auto"))

        assert plotter.figure.figsize == (7, 7)
        assert plotter.figure.aspect == "auto"

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.figure.figsize == None
        assert sub_plotter.figure.aspect == "square"

        sub_plotter = aplt.SubPlotter(figure=aplt.Figure.sub(figsize=(6, 6)))

        assert sub_plotter.figure.figsize == (6, 6)
        assert sub_plotter.figure.aspect == "square"

    def test__colormap__from_config_or_via_manual_input(self):
        plotter = aplt.Plotter()

        assert plotter.cmap.cmap == "jet"
        assert plotter.cmap.norm == "linear"
        assert plotter.cmap.norm_min == None
        assert plotter.cmap.norm_max == None
        assert plotter.cmap.linthresh == 1.0
        assert plotter.cmap.linscale == 2.0

        plotter = aplt.Plotter(
            cmap=aplt.ColorMap(
                cmap="cold",
                norm="log",
                norm_min=0.1,
                norm_max=1.0,
                linthresh=1.5,
                linscale=2.0,
            )
        )

        assert plotter.cmap.cmap == "cold"
        assert plotter.cmap.norm == "log"
        assert plotter.cmap.norm_min == 0.1
        assert plotter.cmap.norm_max == 1.0
        assert plotter.cmap.linthresh == 1.5
        assert plotter.cmap.linscale == 2.0

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.cmap.cmap == "jet"
        assert sub_plotter.cmap.norm == "linear"
        assert sub_plotter.cmap.norm_min == None
        assert sub_plotter.cmap.norm_max == None
        assert sub_plotter.cmap.linthresh == 3.0
        assert sub_plotter.cmap.linscale == 4.0

        sub_plotter = aplt.SubPlotter(
            cmap=aplt.ColorMap.sub(
                cmap="cold", norm="log", norm_min=0.1, norm_max=1.0, linscale=2.0
            )
        )

        assert sub_plotter.cmap.cmap == "cold"
        assert sub_plotter.cmap.norm == "log"
        assert sub_plotter.cmap.norm_min == 0.1
        assert sub_plotter.cmap.norm_max == 1.0
        assert sub_plotter.cmap.linthresh == 3.0
        assert sub_plotter.cmap.linscale == 2.0

    def test__colorbar__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.cb.ticksize == 1
        assert plotter.cb.fraction == 3.0
        assert plotter.cb.pad == 4.0
        assert plotter.cb.tick_values == None
        assert plotter.cb.tick_labels == None

        plotter = aplt.Plotter(
            cb=aplt.ColorBar(
                ticksize=20,
                fraction=0.001,
                pad=10.0,
                tick_values=(1.0, 2.0),
                tick_labels=(3.0, 4.0),
            )
        )

        assert plotter.cb.ticksize == 20
        assert plotter.cb.fraction == 0.001
        assert plotter.cb.pad == 10.0
        assert plotter.cb.tick_values == (1.0, 2.0)
        assert plotter.cb.tick_labels == (3.0, 4.0)

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.cb.ticksize == 1
        assert sub_plotter.cb.fraction == 3.0
        assert sub_plotter.cb.pad == 10.0

        sub_plotter = aplt.SubPlotter(cb=aplt.ColorBar.sub(fraction=0.001, pad=10.0))

        assert sub_plotter.cb.ticksize == 1
        assert sub_plotter.cb.fraction == 0.001
        assert sub_plotter.cb.pad == 10.0

    def test__ticks__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.ticks.ysize == 14
        assert plotter.ticks.xsize == 15
        assert plotter.ticks.y_manual == None
        assert plotter.ticks.x_manual == None

        plotter = aplt.Plotter(
            ticks=aplt.Ticks(
                ysize=24, xsize=25, y_manual=[1.0, 2.0], x_manual=[3.0, 4.0]
            )
        )

        assert plotter.ticks.ysize == 24
        assert plotter.ticks.xsize == 25
        assert plotter.ticks.y_manual == [1.0, 2.0]
        assert plotter.ticks.x_manual == [3.0, 4.0]

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.ticks.ysize == 24
        assert sub_plotter.ticks.xsize == 25
        assert sub_plotter.ticks.y_manual == None
        assert sub_plotter.ticks.x_manual == None

        sub_plotter = aplt.SubPlotter(
            ticks=aplt.Ticks.sub(xsize=25, y_manual=[1.0, 2.0], x_manual=[3.0, 4.0])
        )

        assert sub_plotter.ticks.ysize == 24
        assert sub_plotter.ticks.xsize == 25
        assert sub_plotter.ticks.y_manual == [1.0, 2.0]
        assert sub_plotter.ticks.x_manual == [3.0, 4.0]

    def test__labels__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.labels.title == None
        assert plotter.labels._yunits == None
        assert plotter.labels._xunits == None
        assert plotter.labels.titlesize == 11
        assert plotter.labels.ysize == 12
        assert plotter.labels.xsize == 13

        plotter = aplt.Plotter(
            labels=aplt.Labels(
                title="OMG", yunits="hi", xunits="hi2", titlesize=1, ysize=2, xsize=3
            )
        )

        assert plotter.labels.title == "OMG"
        assert plotter.labels._yunits == "hi"
        assert plotter.labels._xunits == "hi2"
        assert plotter.labels.titlesize == 1
        assert plotter.labels.ysize == 2
        assert plotter.labels.xsize == 3

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.labels.title == None
        assert sub_plotter.labels._yunits == None
        assert sub_plotter.labels._xunits == None
        assert sub_plotter.labels.titlesize == 15
        assert sub_plotter.labels.ysize == 22
        assert sub_plotter.labels.xsize == 23

        sub_plotter = aplt.SubPlotter(
            labels=aplt.Labels.sub(
                title="OMG", yunits="hi", xunits="hi2", ysize=2, xsize=3
            )
        )

        assert sub_plotter.labels.title == "OMG"
        assert sub_plotter.labels._yunits == "hi"
        assert sub_plotter.labels._xunits == "hi2"
        assert sub_plotter.labels.titlesize == 15
        assert sub_plotter.labels.ysize == 2
        assert sub_plotter.labels.xsize == 3

    def test__legend__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.legend.include == True
        assert plotter.legend.fontsize == 12

        plotter = aplt.Plotter(legend=aplt.Legend(include=False, fontsize=11))

        assert plotter.legend.include == False
        assert plotter.legend.fontsize == 11

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.legend.include == False
        assert sub_plotter.legend.fontsize == 13

        sub_plotter = aplt.SubPlotter(legend=aplt.Legend.sub(include=True))

        assert sub_plotter.legend.include == True
        assert sub_plotter.legend.fontsize == 13

    def test__origin_scatterer__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.origin_scatterer.size == 80
        assert plotter.origin_scatterer.marker == "x"
        assert plotter.origin_scatterer.colors == ["k"]

        plotter = aplt.Plotter(
            origin_scatterer=aplt.OriginScatterer(size=1, marker=".", colors="k")
        )

        assert plotter.origin_scatterer.size == 1
        assert plotter.origin_scatterer.marker == "."
        assert plotter.origin_scatterer.colors == ["k"]

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.origin_scatterer.size == 81
        assert sub_plotter.origin_scatterer.marker == "."
        assert sub_plotter.origin_scatterer.colors == ["r"]

        sub_plotter = aplt.SubPlotter(
            origin_scatterer=aplt.OriginScatterer.sub(marker="o", colors=["r", "b"])
        )

        assert sub_plotter.origin_scatterer.size == 81
        assert sub_plotter.origin_scatterer.marker == "o"
        assert sub_plotter.origin_scatterer.colors == ["r", "b"]

    def test__liner__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.liner.width == 1
        assert plotter.liner.style == "-"
        assert plotter.liner.colors == ["k", "w"]
        assert plotter.liner.pointsize == 20

        plotter = aplt.Plotter(
            liner=aplt.Liner(width=1, style=".", colors=["k", "b"], pointsize=3)
        )

        assert plotter.liner.width == 1
        assert plotter.liner.style == "."
        assert plotter.liner.colors == ["k", "b"]
        assert plotter.liner.pointsize == 3

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.liner.width == 1
        assert sub_plotter.liner.style == "-"
        assert sub_plotter.liner.colors == ["k"]
        assert sub_plotter.liner.pointsize == 20

        sub_plotter = aplt.SubPlotter(
            liner=aplt.Liner.sub(style=".", colors="r", pointsize=21)
        )

        assert sub_plotter.liner.width == 1
        assert sub_plotter.liner.style == "."
        assert sub_plotter.liner.colors == ["r"]
        assert sub_plotter.liner.pointsize == 21

    def test__output__correctly(self):

        plotter = aplt.Plotter()

        assert plotter.output.path == None
        assert plotter.output._format == None
        assert plotter.output.format == "show"
        assert plotter.output.filename == None

        plotter = aplt.Plotter(
            output=aplt.Output(path="Path", format="png", filename="file")
        )

        assert plotter.output.path == "Path"
        assert plotter.output._format == "png"
        assert plotter.output.format == "png"
        assert plotter.output.filename == "file"

        if os.path.exists(plotter.output.path):
            shutil.rmtree(plotter.output.path)

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.output.path == None
        assert sub_plotter.output._format == None
        assert sub_plotter.output.format == "show"
        assert sub_plotter.output.filename == None

        sub_plotter = aplt.SubPlotter(
            output=aplt.Output(path="Path", format="png", filename="file")
        )

        assert sub_plotter.output.path == "Path"
        assert sub_plotter.output._format == "png"
        assert sub_plotter.output.format == "png"
        assert sub_plotter.output.filename == "file"

        if os.path.exists(plotter.output.path):
            shutil.rmtree(plotter.output.path)

    def test__parallel_overscan_liner__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.parallel_overscan_liner.width == 1
        assert plotter.parallel_overscan_liner.style == "-"
        assert plotter.parallel_overscan_liner.colors == ["k"]
        assert plotter.parallel_overscan_liner.pointsize == 20

        plotter = aplt.Plotter(
            parallel_overscan_liner=aplt.ParallelOverscanLiner(
                width=1, style=".", colors="k", pointsize=3
            )
        )

        assert plotter.parallel_overscan_liner.width == 1
        assert plotter.parallel_overscan_liner.style == "."
        assert plotter.parallel_overscan_liner.colors == ["k"]
        assert plotter.parallel_overscan_liner.pointsize == 3

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.parallel_overscan_liner.width == 4
        assert sub_plotter.parallel_overscan_liner.style == "-"
        assert sub_plotter.parallel_overscan_liner.colors == ["w"]
        assert sub_plotter.parallel_overscan_liner.pointsize == 23

        sub_plotter = aplt.SubPlotter(
            parallel_overscan_liner=aplt.ParallelOverscanLiner.sub(
                style=".", colors="r", pointsize=21
            )
        )

        assert sub_plotter.parallel_overscan_liner.width == 4
        assert sub_plotter.parallel_overscan_liner.style == "."
        assert sub_plotter.parallel_overscan_liner.colors == ["r"]
        assert sub_plotter.parallel_overscan_liner.pointsize == 21

    def test__serial_prescan_liner__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.serial_prescan_liner.width == 2
        assert plotter.serial_prescan_liner.style == "--"
        assert plotter.serial_prescan_liner.colors == ["g"]
        assert plotter.serial_prescan_liner.pointsize == 21

        plotter = aplt.Plotter(
            serial_prescan_liner=aplt.SerialPrescanLiner(
                width=1, style=".", colors="k", pointsize=3
            )
        )

        assert plotter.serial_prescan_liner.width == 1
        assert plotter.serial_prescan_liner.style == "."
        assert plotter.serial_prescan_liner.colors == ["k"]
        assert plotter.serial_prescan_liner.pointsize == 3

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.serial_prescan_liner.width == 5
        assert sub_plotter.serial_prescan_liner.style == "--"
        assert sub_plotter.serial_prescan_liner.colors == ["b"]
        assert sub_plotter.serial_prescan_liner.pointsize == 24

        sub_plotter = aplt.SubPlotter(
            serial_prescan_liner=aplt.SerialPrescanLiner.sub(
                style=".", colors="r", pointsize=21
            )
        )

        assert sub_plotter.serial_prescan_liner.width == 5
        assert sub_plotter.serial_prescan_liner.style == "."
        assert sub_plotter.serial_prescan_liner.colors == ["r"]
        assert sub_plotter.serial_prescan_liner.pointsize == 21

    def test__serial_overscan_liner__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.serial_overscan_liner.width == 3
        assert plotter.serial_overscan_liner.style == "-"
        assert plotter.serial_overscan_liner.colors == ["y"]
        assert plotter.serial_overscan_liner.pointsize == 23

        plotter = aplt.Plotter(
            serial_overscan_liner=aplt.SerialOverscanLiner(
                width=1, style=".", colors="k", pointsize=3
            )
        )

        assert plotter.serial_overscan_liner.width == 1
        assert plotter.serial_overscan_liner.style == "."
        assert plotter.serial_overscan_liner.colors == ["k"]
        assert plotter.serial_overscan_liner.pointsize == 3

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.serial_overscan_liner.width == 6
        assert sub_plotter.serial_overscan_liner.style == "-"
        assert sub_plotter.serial_overscan_liner.colors == ["r"]
        assert sub_plotter.serial_overscan_liner.pointsize == 25

        sub_plotter = aplt.SubPlotter(
            serial_overscan_liner=aplt.SerialOverscanLiner.sub(
                style=".", colors="r", pointsize=21
            )
        )

        assert sub_plotter.serial_overscan_liner.width == 6
        assert sub_plotter.serial_overscan_liner.style == "."
        assert sub_plotter.serial_overscan_liner.colors == ["r"]
        assert sub_plotter.serial_overscan_liner.pointsize == 21


class TestAbstractPlotterPlots:
    def test__plot_frame__works_with_all_extras_included(self, plot_path, plot_patch):

        frame = struct.Frame.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0))

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="frame1", format="png")
        )

        plotter.plot_frame(frame=frame, include_origin=True)

        assert plot_path + "frame1.png" in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="frame2", format="png")
        )

        plotter.plot_frame(frame=frame, include_origin=True)

        assert plot_path + "frame2.png" in plot_patch.paths

        aplt.Frame(
            frame=frame,
            plotter=aplt.Plotter(
                output=aplt.Output(path=plot_path, filename="frame3", format="png")
            ),
        )

        assert plot_path + "frame3.png" in plot_patch.paths

    def test__plot_frame__fits_files_output_correctly(self, plot_path):

        plot_path = plot_path + "/fits/"

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        frame = struct.Frame.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0))

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="frame", format="fits")
        )

        plotter.plot_frame(frame=frame)

        frame = array_util.numpy_array_2d_from_fits(
            file_path=plot_path + "/frame.fits", hdu=0
        )

        assert (frame == np.ones(shape=(31, 31))).all()

        mask = struct.Mask.unmasked(shape_2d=(31, 31), pixel_scales=(1.0, 1.0))

        masked_frame = struct.MaskedFrame.manual(array=frame, mask=mask)

        plotter.plot_frame(frame=masked_frame)

        frame = array_util.numpy_array_2d_from_fits(
            file_path=plot_path + "/frame.fits", hdu=0
        )

        assert frame.shape == (31, 31)

    def test__plot_line__works_with_all_extras_included(self, plot_path, plot_patch):

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="line1", format="png")
        )

        plotter.plot_line(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="loglog",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
        )

        assert plot_path + "line1.png" in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="line2", format="png")
        )

        plotter.plot_line(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="semilogy",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
        )

        assert plot_path + "line2.png" in plot_patch.paths

        aplt.Line(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="loglog",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
            plotter=aplt.Plotter(
                output=aplt.Output(path=plot_path, filename="line3", format="png")
            ),
        )

        assert plot_path + "line3.png" in plot_patch.paths


class TestAbstractPlotterNew:
    def test__plotter_with_new_labels__new_labels_if_input__sizes_dont_change(self):

        plotter = aplt.Plotter(
            labels=aplt.Labels(
                title="OMG", yunits="hi", xunits="hi2", titlesize=1, ysize=2, xsize=3
            )
        )

        plotter = plotter.plotter_with_new_labels()

        assert plotter.labels.title == "OMG"
        assert plotter.labels._yunits == "hi"
        assert plotter.labels._xunits == "hi2"
        assert plotter.labels.titlesize == 1
        assert plotter.labels.ysize == 2
        assert plotter.labels.xsize == 3

        plotter = plotter.plotter_with_new_labels(
            title="OMG0", yunits="hi0", xunits="hi20", titlesize=10, ysize=20, xsize=30
        )

        assert plotter.labels.title == "OMG0"
        assert plotter.labels._yunits == "hi0"
        assert plotter.labels._xunits == "hi20"
        assert plotter.labels.titlesize == 10
        assert plotter.labels.ysize == 20
        assert plotter.labels.xsize == 30

        plotter = plotter.plotter_with_new_labels(
            title="OMG0", yunits="hi0", xunits="hi20", titlesize=10
        )

        assert plotter.labels.title == "OMG0"
        assert plotter.labels._yunits == "hi0"
        assert plotter.labels._xunits == "hi20"
        assert plotter.labels.titlesize == 10
        assert plotter.labels.ysize == 20
        assert plotter.labels.xsize == 30

    def test__plotter_with_new_cmap__new_labels_if_input__sizes_dont_change(self):

        plotter = aplt.Plotter(
            cmap=aplt.ColorMap(
                cmap="cold",
                norm="log",
                norm_min=0.1,
                norm_max=1.0,
                linthresh=1.5,
                linscale=2.0,
            )
        )

        assert plotter.cmap.cmap == "cold"
        assert plotter.cmap.norm == "log"
        assert plotter.cmap.norm_min == 0.1
        assert plotter.cmap.norm_max == 1.0
        assert plotter.cmap.linthresh == 1.5
        assert plotter.cmap.linscale == 2.0

        plotter = plotter.plotter_with_new_cmap(
            cmap="jet",
            norm="linear",
            norm_min=0.12,
            norm_max=1.2,
            linthresh=1.2,
            linscale=2.2,
        )

        assert plotter.cmap.cmap == "jet"
        assert plotter.cmap.norm == "linear"
        assert plotter.cmap.norm_min == 0.12
        assert plotter.cmap.norm_max == 1.2
        assert plotter.cmap.linthresh == 1.2
        assert plotter.cmap.linscale == 2.2

        plotter = plotter.plotter_with_new_cmap(cmap="sand", norm="log", norm_min=0.13)

        assert plotter.cmap.cmap == "sand"
        assert plotter.cmap.norm == "log"
        assert plotter.cmap.norm_min == 0.13
        assert plotter.cmap.norm_max == 1.2
        assert plotter.cmap.linthresh == 1.2
        assert plotter.cmap.linscale == 2.2

    def test__plotter_with_new_outputs__new_outputs_are_setup_correctly_if_input(self):

        plotter = aplt.Plotter(
            output=aplt.Output(path="Path", format="png", filename="file")
        )

        plotter = plotter.plotter_with_new_output()

        assert plotter.output.path == "Path"
        assert plotter.output._format == "png"
        assert plotter.output.format == "png"
        assert plotter.output.filename == "file"

        if os.path.exists(plotter.output.path):
            shutil.rmtree(plotter.output.path)

        plotter = plotter.plotter_with_new_output(path="Path0", filename="file0")

        assert plotter.output.path == "Path0"
        assert plotter.output._format == "png"
        assert plotter.output.format == "png"
        assert plotter.output.filename == "file0"

        if os.path.exists(plotter.output.path):
            shutil.rmtree(plotter.output.path)

        plotter = plotter.plotter_with_new_output(
            path="Path1", filename="file1", format="fits"
        )

        assert plotter.output.path == "Path1"
        assert plotter.output._format == "fits"
        assert plotter.output.format == "fits"
        assert plotter.output.filename == "file1"

        if os.path.exists(plotter.output.path):
            shutil.rmtree(plotter.output.path)

    def test__plotter_with_new_units__new_outputs_are_setup_correctly_if_input(self):

        plotter = aplt.Plotter(
            aplt.Units(use_scaled=True, in_kpc=True, conversion_factor=1.0)
        )

        assert plotter.units.use_scaled == True
        assert plotter.units.in_kpc == True
        assert plotter.units.conversion_factor == 1.0

        plotter = plotter.plotter_with_new_units(
            use_scaled=False, in_kpc=False, conversion_factor=2.0
        )

        assert plotter.units.use_scaled == False
        assert plotter.units.in_kpc == False
        assert plotter.units.conversion_factor == 2.0

        plotter = plotter.plotter_with_new_units(conversion_factor=3.0)

        assert plotter.units.use_scaled == False
        assert plotter.units.in_kpc == False
        assert plotter.units.conversion_factor == 3.0

    def test__open_and_close_subplot_figures(self):

        plotter = aplt.Plotter()
        plotter.figure.open()

        assert plt.fignum_exists(num=1) == True

        plotter.figure.close()

        assert plt.fignum_exists(num=1) == False

        plotter = aplt.SubPlotter()

        assert plt.fignum_exists(num=1) == False

        plotter.open_subplot_figure(number_subplots=4)

        assert plt.fignum_exists(num=1) == True

        plotter.figure.close()

        assert plt.fignum_exists(num=1) == False


class TestSubPlotter:
    def test__subplot_figsize_for_number_of_subplots(self):

        plotter = aplt.SubPlotter()

        figsize = plotter.get_subplot_figsize(number_subplots=1)

        assert figsize == (18, 8)

        figsize = plotter.get_subplot_figsize(number_subplots=4)

        assert figsize == (13, 10)

        plotter = aplt.SubPlotter(figure=aplt.Figure(figsize=(20, 20)))

        figsize = plotter.get_subplot_figsize(number_subplots=4)

        assert figsize == (20, 20)

    def test__plotter_number_of_subplots(self):

        plotter = aplt.SubPlotter()

        rows, columns = plotter.get_subplot_rows_columns(number_subplots=1)

        assert rows == 1
        assert columns == 2

        rows, columns = plotter.get_subplot_rows_columns(number_subplots=4)

        assert rows == 2
        assert columns == 2


class TestInclude:
    def test__parallel_overscan_from_frame(self):

        frame = struct.Frame.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0))

        include = aplt.Include(parallel_overscan=True)

        parallel_overscan = include.parallel_overscan_from_frame(frame=frame)

        assert parallel_overscan == None

        frame = struct.Frame.ones(
            shape_2d=(31, 31), pixel_scales=(1.0, 1.0), parallel_overscan=(0, 1, 2, 3)
        )

        include = aplt.Include(parallel_overscan=False)

        parallel_overscan = include.parallel_overscan_from_frame(frame=frame)

        assert parallel_overscan == None

        include = aplt.Include(parallel_overscan=True)

        parallel_overscan = include.parallel_overscan_from_frame(frame=frame)

        assert parallel_overscan == (0, 1, 2, 3)

    def test__serial_prescan_from_frame(self):

        frame = struct.Frame.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0))

        include = aplt.Include(serial_prescan=True)

        serial_prescan = include.serial_prescan_from_frame(frame=frame)

        assert serial_prescan == None

        frame = struct.Frame.ones(
            shape_2d=(31, 31), pixel_scales=(1.0, 1.0), serial_prescan=(0, 1, 2, 3)
        )

        include = aplt.Include(serial_prescan=False)

        serial_prescan = include.serial_prescan_from_frame(frame=frame)

        assert serial_prescan == None

        include = aplt.Include(serial_prescan=True)

        serial_prescan = include.serial_prescan_from_frame(frame=frame)

        assert serial_prescan == (0, 1, 2, 3)

    def test__serial_overscan_from_frame(self):

        frame = struct.Frame.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0))

        include = aplt.Include(serial_overscan=True)

        serial_overscan = include.serial_overscan_from_frame(frame=frame)

        assert serial_overscan == None

        frame = struct.Frame.ones(
            shape_2d=(31, 31), pixel_scales=(1.0, 1.0), serial_overscan=(0, 1, 2, 3)
        )

        include = aplt.Include(serial_overscan=False)

        serial_overscan = include.serial_overscan_from_frame(frame=frame)

        assert serial_overscan == None

        include = aplt.Include(serial_overscan=True)

        serial_overscan = include.serial_overscan_from_frame(frame=frame)

        assert serial_overscan == (0, 1, 2, 3)


from autocti.plot import plotters


class TestDecorator:
    def test__kpc_per_arcsec_extacted_from_object_if_available(self):

        dictionary = {"hi": 1}

        kpc_per_arcsec = plotters.kpc_per_arcsec_of_object_from_dictionary(
            dictionary=dictionary
        )

        assert kpc_per_arcsec == None

        class MockObj:
            def __init__(self, param1):

                self.param1 = param1

        obj = MockObj(param1=1)

        dictionary = {"hi": 1, "hello": obj}

        kpc_per_arcsec = plotters.kpc_per_arcsec_of_object_from_dictionary(
            dictionary=dictionary
        )

        assert kpc_per_arcsec == None

        class MockObj:
            def __init__(self, param1, kpc_per_arcsec):

                self.param1 = param1
                self.kpc_per_arcsec = kpc_per_arcsec

        obj = MockObj(param1=1, kpc_per_arcsec=2)

        dictionary = {"hi": 1, "hello": obj}

        kpc_per_arcsec = plotters.kpc_per_arcsec_of_object_from_dictionary(
            dictionary=dictionary
        )

        assert kpc_per_arcsec == 2
