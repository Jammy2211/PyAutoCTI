import matplotlib.pyplot as plt

from autocti.util import exc
from autocti.plot import ci_plotter_util


def plot_line_from_ci_frame(ci_frame, line_region, include=None, plotter=None):

    if line_region is "parallel_front_edge":
        line = ci_frame.parallel_front_edge_line_binned_over_columns()
    elif line_region is "parallel_trails":
        line = ci_frame.parallel_trails_line_binned_over_columns()
    elif line_region is "serial_front_edge":
        line = ci_frame.serial_front_edge_line_binned_over_rows()
    elif line_region is "serial_trails":
        line = ci_frame.serial_trails_line_binned_over_rows()
    else:
        raise exc.PlottingException(
            "The line region specified for the plotting of a line was invalid"
        )

    plotter.plot_line(y=line, x=range(len(line)))
