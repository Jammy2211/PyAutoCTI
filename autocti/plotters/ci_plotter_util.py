import matplotlib.pyplot as plt
import numpy as np

from autocti import exc
from autoarray.util import array_util


def line_regions_from_plots(
    plot_parallel_front_edge_line,
    plot_parallel_trails_line,
    plot_serial_front_edge_line,
    plot_serial_trails_line,
):

    line_regions = []

    if plot_parallel_front_edge_line:
        line_regions.append("parallel_front_edge")

    if plot_parallel_trails_line:
        line_regions.append("parallel_trails")

    if plot_serial_front_edge_line:
        line_regions.append("serial_front_edge")

    if plot_serial_trails_line:
        line_regions.append("serial_trails")

    return line_regions
