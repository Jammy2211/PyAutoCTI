import numpy as np
import pytest
import autocti as ac
from autocti import exc


class TestAbstractExtractor:
    def test__total_pixels_minimum(self):

        layout = ac.Extractor1DFrontEdge(region_list=[(1, 2)])

        assert layout.total_pixels_min == 1

        layout = ac.Extractor1DFrontEdge(region_list=[(1, 3)])

        assert layout.total_pixels_min == 2

        layout = ac.Extractor1DFrontEdge(region_list=[(1, 3), (0, 5)])

        assert layout.total_pixels_min == 2

        layout = ac.Extractor1DFrontEdge(region_list=[(1, 3), (4, 5)])

        assert layout.total_pixels_min == 1
