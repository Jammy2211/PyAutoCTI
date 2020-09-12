""" 
Find warm pixels in an image from the Hubble Space Telescope (HST) Advanced 
Camera for Surveys (ACS) instrument.

A small patch of the image is plotted with the warm pixels marked with red Xs.
"""

import numpy as np
import pytest
import os
from autoconf import conf
import matplotlib.pyplot as plt

from autocti.data.pixel_lines import PixelLine, PixelLineCollection
from autocti.model.warm_pixels import find_warm_pixels
from autoarray.instruments import acs

# Path to this file
path = os.path.dirname(os.path.realpath(__file__))

# Set up some configuration options for the automatic fits dataset loading
conf.instance = conf.Config(config_path=f"{path}/config")

# Load the HST ACS dataset
name = "acs/jc0a01h8q_raw"
frame = acs.FrameACS.from_fits(file_path=f"{path}/{name}.fits", quadrant_letter="A")

# Extract an example patch of the full image
row_start, row_end, column_start, column_end = -300, -100, -300, -100
frame = frame[row_start:row_end, column_start:column_end]
frame.mask = frame.mask[row_start:row_end, column_start:column_end]

# Find the warm pixel trails and store in a line collection object
warm_pixels = PixelLineCollection(
    lines=find_warm_pixels(image=frame, ignore_bad_columns=0)
)

print("Found %d warm pixels" % warm_pixels.n_lines)

# Plot the image and the found warm pixels
plt.figure()
im = plt.imshow(X=frame, aspect="equal", vmin=2300, vmax=2800)
plt.scatter(
    warm_pixels.locations[:, 1],
    warm_pixels.locations[:, 0],
    c="r",
    marker="x",
    s=8,
    linewidth=0.2,
)
plt.colorbar(im)
plt.axis("off")
plt.savefig(f"{path}/find_warm_pixels.png", dpi=400)
plt.close()
print(f"Saved {path}/find_warm_pixels.png")
