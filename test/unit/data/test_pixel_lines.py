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
        line_1 = PixelLine(data=[1, 2, 3], origin=100)
        line_2 = PixelLine(data=[4, 5, 6], origin=100)
        line_3 = PixelLine(data=[7, 8, 9], origin=200)
        line_4 = PixelLine(data=[0, 0, 0], origin=200)

        lines = PixelLineCollection(lines=[line_1, line_2, line_3, line_4])

        assert lines.n_lines == 4

        assert lines.data[0] == pytest.approx(line_1.data)
        assert lines.data[1] == pytest.approx(line_2.data)
        assert lines.data[2] == pytest.approx(line_3.data)
        assert lines.data[3] == pytest.approx(line_4.data)

        assert lines.origins == pytest.approx([100, 100, 200, 200])

        assert all(lines.dates == [None, None, None, None])

    def test__pixel_line_collection__init_from_data(self):
        line_1 = PixelLine(data=[1, 2, 3], origin=100)
        line_2 = PixelLine(data=[4, 5, 6], origin=100)
        line_3 = PixelLine(data=[7, 8, 9], origin=200)
        line_4 = PixelLine(data=[0, 0, 0], origin=200)

        lines = PixelLineCollection(
            data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]],
            origins=[100, 100, 200, 200],
        )

        assert lines.n_lines == 4

        assert lines.data[0] == pytest.approx(line_1.data)
        assert lines.data[1] == pytest.approx(line_2.data)
        assert lines.data[2] == pytest.approx(line_3.data)
        assert lines.data[3] == pytest.approx(line_4.data)

        assert lines.lines[0].data == pytest.approx(line_1.data)
        assert lines.lines[1].data == pytest.approx(line_2.data)
        assert lines.lines[2].data == pytest.approx(line_3.data)
        assert lines.lines[3].data == pytest.approx(line_4.data)

        assert lines.lines[0].origin == line_1.origin
        assert lines.lines[1].origin == line_2.origin
        assert lines.lines[2].origin == line_3.origin
        assert lines.lines[3].origin == line_4.origin

        assert lines.lines[0].date is None
        assert lines.lines[1].date is None
        assert lines.lines[2].date is None
        assert lines.lines[3].date is None

    def test__find_consistent_lines(self):
        line_1 = PixelLine(data=[1, 2, 3], location=[0, 0], origin=100)
        line_2 = PixelLine(data=[4, 5, 6], location=[1, 1], origin=100)
        line_3 = PixelLine(data=[7, 8, 9], location=[2, 2], origin=100)
        line_4 = PixelLine(data=[0, 0, 0], location=[0, 0], origin=200)
        line_5 = PixelLine(data=[3, 2, 1], location=[1, 1], origin=200)
        line_6 = PixelLine(data=[3, 1, 0], location=[3, 4], origin=200)
        line_7 = PixelLine(data=[2, 1, 0], location=[0, 0], origin=300)
        line_8 = PixelLine(data=[1, 1, 0], location=[1, 2], origin=300)
        line_9 = PixelLine(data=[1, 0, 0], location=[4, 4], origin=300)

        lines = PixelLineCollection(
            lines=[
                line_1,
                line_2,
                line_3,
                line_4,
                line_5,
                line_6,
                line_7,
                line_8,
                line_9,
            ]
        )

        # Lines present in every image
        consistent_lines = lines.find_consistent_lines(fraction_present=1)

        assert consistent_lines == pytest.approx([0, 3, 6])
        assert lines.origins[consistent_lines] == pytest.approx([100, 200, 300])

        # Lines present in at least 2/3 images
        consistent_lines = lines.find_consistent_lines(fraction_present=2 / 3)

        assert consistent_lines == pytest.approx([0, 1, 3, 4, 6])

        # Lines present in at least 1/2 images
        consistent_lines = lines.find_consistent_lines(fraction_present=0.5)

        assert consistent_lines == pytest.approx([0, 1, 3, 4, 6])

        # Lines present in at least 1/3 images (in this case, all lines)
        consistent_lines = lines.find_consistent_lines(fraction_present=1 / 3)

        assert consistent_lines == pytest.approx([0, 1, 2, 3, 4, 5, 6, 7, 8])
