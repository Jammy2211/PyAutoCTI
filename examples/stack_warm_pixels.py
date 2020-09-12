""" 
Find and stack warm pixels in bins from a set of images from the Hubble Space 
Telescope (HST) Advanced Camera for Surveys (ACS) instrument.

This is a three-step process:
1. Possible warm pixels are first found in each image (by finding ~delta 
   function local maxima). 
2. The warm pixels are then extracted by ensuring they appear in at least 2/3 of 
   the different images, to discard noise peaks etc. 
3. Finally, they are stacked in bins by (in this example) distance from the 
   readout register and their flux. 

The trails in each bin are then plotted, labelled by their bin start values. 

See the docstrings of find_warm_pixels() in autocti/model/warm_pixels.py and
of find_consistent_lines() and generate_stacked_lines_from_bins() in 
PixelLineCollection in autocti/data/pixel_lines.py for full details.
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

# Initialise the collection of warm pixel trails
warm_pixels = PixelLineCollection()


print("1.")
# Find the warm pixels in each image
for name in [
    "j9epn8s6q_raw",
    "j9epqbgjq_raw",
    "j9epr7stq_raw",
    "j9epu6bvq_raw",
    "j9epn8s7q_raw",
    "j9epqbgkq_raw",
    "j9epr7suq_raw",
    "j9epu6bwq_raw",
    "j9epn8s9q_raw",
    "j9epqbgmq_raw",
    "j9epr7swq_raw",
    "j9epu6byq_raw",
    "j9epn8sbq_raw",
    "j9epqbgoq_raw",
    "j9epr7syq_raw",
    "j9epu6c0q_raw",
]:
    # Load the HST ACS dataset
    frame = acs.FrameACS.from_fits(
        file_path=f"{path}/acs/{name}.fits", quadrant_letter="A"
    )
    date = 2400000.5 + frame.exposure_info.modified_julian_date

    # Find the warm pixel trails
    new_warm_pixels = find_warm_pixels(image=frame, origin=name, date=date)

    print("Found %d possible warm pixels in %s" % (len(new_warm_pixels), name))

    # Add them to the collection
    warm_pixels.append(new_warm_pixels)

# For reference, could save the lines and then load at a later time to continue
if not True:
    Save
    warm_pixels.save("warm_pixel_lines")

    # Load
    warm_pixels = PixelLineCollection()
    warm_pixels.load("warm_pixel_lines")


print("2.")
# Find the consistent warm pixels present in at least 2/3 of the images
consistent_lines = warm_pixels.find_consistent_lines(fraction_present=2 / 3)
print(
    "Found %d consistent warm pixels out of %d possibles"
    % (len(consistent_lines), warm_pixels.n_lines)
)

# Extract the consistent warm pixels
warm_pixels.lines = warm_pixels.lines[consistent_lines]


print("3.")
# Stack the lines in bins by distance from readout and total flux
n_row_bins = 5
n_flux_bins = 5
n_bins = n_row_bins * n_flux_bins
(
    stacked_lines,
    row_bin_low,
    date_bin_low,
    background_bin_low,
    flux_bin_low,
) = warm_pixels.generate_stacked_lines_from_bins(
    n_row_bins=n_row_bins, n_flux_bins=n_flux_bins, return_bin_info=True
)

print(
    "Stacked lines in %d bins with %d empty bins"
    % (n_bins, n_bins - stacked_lines.n_lines)
)

# Plot the stacked trails
plt.figure()
for i, line in enumerate(stacked_lines.lines):
    if i < 10:
        ls = "-"
    elif i < 20:
        ls = "--"
    else:
        ls = ":"
    plt.plot(
        np.arange(line.length),
        line.data,
        lw=1,
        ls=ls,
        label="%d, %.0f" % (line.location[0], line.flux),
    )
plt.yscale("log")
plt.legend(loc="upper right", ncol=3, title="bin start: row, flux", prop={"size": 7})
plt.ylim(0.2, None)
plt.savefig(f"{path}/stack_warm_pixels.png", dpi=400)
plt.close()
print(f"Saved {path}/stack_warm_pixels.png")
