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
