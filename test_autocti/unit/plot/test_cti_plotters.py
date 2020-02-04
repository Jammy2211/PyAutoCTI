import autoarray as aa
from os import path
import os
import pytest
import shutil

import numpy as np
import autocti as ac
import autocti.plot as aplt


directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plotter_setup():
    return "{}/..//test_files/plot/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(autouse=True)
def set_config_path():
    aa.conf.instance = aa.conf.Config(
        path.join(directory, "../test_files/plot"), path.join(directory, "output")
    )


class TestCTIPlotterAttributes:

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
        assert sub_plotter.parallel_overscan_liner.style == "x"
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
        assert plotter.serial_overscan_liner.style == "x"
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


class TestCTIPlotterPlots:
    def test__plot_frame__works_with_all_extras_included(self, plot_path, plot_patch):

        frame = ac.frame.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0))

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="frame1", format="png")
        )

        plotter.plot_frame(
            frame=frame,
            include_origin=True,
        )

        assert plot_path + "frame1.png" in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="frame2", format="png")
        )

        plotter.plot_frame(
            frame=frame,
            include_origin=True,
        )

        assert plot_path + "frame2.png" in plot_patch.paths

        aplt.frame(
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

        frame = ac.frame.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0))

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="frame", format="fits")
        )

        plotter.plot_frame(frame=frame)

        frame = aa.util.array.numpy_array_2d_from_fits(
            file_path=plot_path + "/frame.fits", hdu=0
        )

        assert (frame == np.ones(shape=(31, 31))).all()

        mask = aa.mask.circular(
            shape_2d=(31, 31), pixel_scales=(1.0, 1.0), radius=5.0, centre=(2.0, 2.0)
        )

        masked_frame = ac.masked.frame.manual(array=frame, mask=mask)

        plotter.plot_frame(frame=masked_frame)

        frame = aa.util.array.numpy_array_2d_from_fits(
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

        aplt.line(
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


class TestInclude:
    def test__parallel_overscan_from_frame(self):

        frame = ac.frame.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0))

        include = aplt.Include(parallel_overscan=True)

        parallel_overscan = include.parallel_overscan_from_frame(frame=frame)

        assert parallel_overscan == None

        frame = ac.frame.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0), parallel_overscan=(0, 1, 2, 3))

        include = aplt.Include(parallel_overscan=False)

        parallel_overscan = include.parallel_overscan_from_frame(frame=frame)

        assert parallel_overscan == None

        include = aplt.Include(parallel_overscan=True)

        parallel_overscan = include.parallel_overscan_from_frame(frame=frame)

        assert parallel_overscan == (0, 1, 2, 3)

    def test__serial_prescan_from_frame(self):

        frame = ac.frame.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0))

        include = aplt.Include(serial_prescan=True)

        serial_prescan = include.serial_prescan_from_frame(frame=frame)

        assert serial_prescan == None

        frame = ac.frame.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0), serial_prescan=(0, 1, 2, 3))

        include = aplt.Include(serial_prescan=False)

        serial_prescan = include.serial_prescan_from_frame(frame=frame)

        assert serial_prescan == None

        include = aplt.Include(serial_prescan=True)

        serial_prescan = include.serial_prescan_from_frame(frame=frame)

        assert serial_prescan == (0, 1, 2, 3)

    def test__serial_overscan_from_frame(self):

        frame = ac.frame.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0))

        include = aplt.Include(serial_overscan=True)

        serial_overscan = include.serial_overscan_from_frame(frame=frame)

        assert serial_overscan == None

        frame = ac.frame.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0), serial_overscan=(0, 1, 2, 3))

        include = aplt.Include(serial_overscan=False)

        serial_overscan = include.serial_overscan_from_frame(frame=frame)

        assert serial_overscan == None

        include = aplt.Include(serial_overscan=True)

        serial_overscan = include.serial_overscan_from_frame(frame=frame)

        assert serial_overscan == (0, 1, 2, 3)