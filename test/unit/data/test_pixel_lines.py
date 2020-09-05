import numpy as np
import pytest

from autocti.data.pixel_lines import PixelLine, PixelLineCollection


class TestPixelLine:
    def test__pixel_line__init_etc(self):
        line = PixelLine(
            data=[1.23, 4.56, 7.89],
            origin=None,
            location=[3, 2],
            date=None,
            background=0.5,
            flux=None,
        )

        assert line.length == 3
        assert line.flux == 7.89

        line = PixelLine(
            data=None,
            origin=None,
            location=None,
            date=None,
            background=None,
            flux=None,
        )

        assert line.length is None

        line.data = [1, 2, 3, 4, 5]

        assert line.length == 5


class TestPixelLineCollection:
    def test__pixel_line_collection__init_etc(self):
        line_1 = PixelLine(data=[1, 2, 3], origin="a")
        line_2 = PixelLine(data=[4, 5, 6], origin="a")
        line_3 = PixelLine(data=[7, 8, 9], origin="b")
        line_4 = PixelLine(data=[0, 0, 0], origin="b")

        lines = PixelLineCollection(lines=[line_1, line_2, line_3, line_4])

        assert lines.n_lines == 4

        assert lines.data[0] == line_1.data
        assert lines.data[1] == line_2.data
        assert lines.data[2] == line_3.data
        assert lines.data[3] == line_4.data

        assert lines.origins == ["a", "a", "b", "b"]

        assert lines.dates == [None, None, None, None]
