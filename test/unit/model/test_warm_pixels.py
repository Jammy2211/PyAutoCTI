import numpy as np
import pytest
import os
from autoconf import conf
import matplotlib.pyplot as plt

from autocti.data.pixel_lines import PixelLine, PixelLineCollection
from autocti.model.warm_pixels import find_warm_pixels
from autoarray.instruments import acs

# For other tests related to warm pixels, see also test__find_consistent_lines() 
# and test__generate_stacked_lines_from_bins() in test_pixel_lines.py.


class TestFindWarmPixels:
    def test__find_warm_pixels__hst_acs(self):

        # Path to this file
        path = os.path.dirname(os.path.realpath(__file__))

        # Set up some configuration options for the automatic fits dataset loading
        path += "/../../../examples"
        conf.instance = conf.Config(config_path=f"{path}/config")

        # Load the HST ACS dataset
        path += "/acs"
        name = "jc0a01h8q_raw"
        frame = acs.FrameACS.from_fits(
            file_path=f"{path}/{name}.fits", quadrant_letter="A"
        )

        # Extract an example patch of the full image
        row_start, row_end, column_start, column_end = -300, -100, -300, -100
        frame = frame[row_start:row_end, column_start:column_end]
        frame.mask = frame.mask[row_start:row_end, column_start:column_end]

        # Find the warm pixel trails
        warm_pixels = find_warm_pixels(image=frame)

        assert len(warm_pixels) == 857
