import matplotlib.pyplot as plt
import numpy as np

from autocti import exc
from autoarray.util import array_util


def line_regions_from_should_plots(
    should_plot_parallel_front_edge_line,
    should_plot_parallel_trails_line,
    should_plot_serial_front_edge_line,
    should_plot_serial_trails_line,
):

    line_regions = []

    if should_plot_parallel_front_edge_line:
        line_regions.append("parallel_front_edge")

    if should_plot_parallel_trails_line:
        line_regions.append("parallel_trails")

    if should_plot_serial_front_edge_line:
        line_regions.append("serial_front_edge")

    if should_plot_serial_trails_line:
        line_regions.append("serial_trails")

    return line_regions
