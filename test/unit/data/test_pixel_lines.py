import numpy as np
import pytest

from autocti.data.pixel_lines import PixelLine


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
