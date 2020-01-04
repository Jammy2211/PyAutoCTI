import matplotlib.pyplot as plt

from autocti import exc
from autoarray.plotters import line_yx_plotters
from autocti.plotters import ci_plotter_util


def plot_line_from_ci_frame(
    ci_frame,
    line_region,
    as_subplot=False,
    figsize=(7, 7),
    title="Stack",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="line",
):

    if line_region is "parallel_front_edge":
        line = ci_frame.parallel_front_edge_line_binned_over_columns
    elif line_region is "parallel_trails":
        line = ci_frame.parallel_trails_line_binned_over_columns
    elif line_region is "serial_front_edge":
        line = ci_frame.serial_front_edge_line_binned_over_rows
    elif line_region is "serial_trails":
        line = ci_frame.serial_trails_line_binned_over_rows
    else:
        raise exc.PlottingException(
            "The line region specified for the plotting of a line was invalid"
        )

    line_yx_plotters.plot_line(
        y=line,
        as_subplot=as_subplot,
        figsize=figsize,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )
