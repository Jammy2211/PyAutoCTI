def phase_tag_from_phase_settings(
    columns,
    rows,
    parallel_front_edge_mask_rows,
    parallel_trails_mask_rows,
    serial_front_edge_mask_columns,
    serial_trails_mask_columns,
    parallel_total_density_range,
    serial_total_density_range,
    cosmic_ray_parallel_buffer,
    cosmic_ray_serial_buffer,
    cosmic_ray_diagonal_buffer,
):

    columns_tag = columns_tag_from_columns(columns=columns)
    rows_tag = rows_tag_from_rows(rows=rows)

    parallel_front_edge_mask_rows_tag = parallel_front_edge_mask_rows_tag_from_parallel_front_edge_mask_rows(
        parallel_front_edge_mask_rows=parallel_front_edge_mask_rows
    )
    parallel_trails_mask_rows_tag = parallel_trails_mask_rows_tag_from_parallel_trails_mask_rows(
        parallel_trails_mask_rows=parallel_trails_mask_rows
    )
    serial_front_edge_mask_columns_tag = serial_front_edge_mask_columns_tag_from_serial_front_edge_mask_columns(
        serial_front_edge_mask_columns=serial_front_edge_mask_columns
    )
    serial_trails_mask_columns_tag = serial_trails_mask_columns_tag_from_serial_trails_mask_columns(
        serial_trails_mask_columns=serial_trails_mask_columns
    )

    parallel_total_density_range_tag = parallel_total_density_range_tag_from_parallel_total_density_range(
        parallel_total_density_range=parallel_total_density_range
    )
    serial_total_density_range_tag = serial_total_density_range_tag_from_serial_total_density_range(
        serial_total_density_range=serial_total_density_range
    )

    cosmic_ray_buffer_tag = cosmic_ray_buffer_tag_from_cosmic_ray_buffers(
        cosmic_ray_parallel_buffer=cosmic_ray_parallel_buffer,
        cosmic_ray_serial_buffer=cosmic_ray_serial_buffer,
        cosmic_ray_diagonal_buffer=cosmic_ray_diagonal_buffer,
    )

    return (
        "phase_tag"
        + columns_tag
        + rows_tag
        + parallel_front_edge_mask_rows_tag
        + parallel_trails_mask_rows_tag
        + serial_front_edge_mask_columns_tag
        + serial_trails_mask_columns_tag
        + parallel_total_density_range_tag
        + serial_total_density_range_tag
        + cosmic_ray_buffer_tag
    )


def columns_tag_from_columns(columns):
    """Generate a columns tag, to customize phase names based on the number of columns of simulator extracted in the fit,
    which is used to speed up parallel CTI fits.

    This changes the phase name 'phase_name' as follows:

    columns = None -> phase_name
    columns = 10 -> phase_name_col_10
    columns = 60 -> phase_name_col_60
    """
    if columns == None:
        return ""
    else:
        x0 = str(columns[0])
        x1 = str(columns[1])
        return "__columns_(" + x0 + "," + x1 + ")"


def rows_tag_from_rows(rows):
    """Generate a rows tag, to customize phase names based on the number of rows of simulator extracted in the fit,
    which is used to speed up serial CTI fits.

    This changes the phase name 'phase_name' as follows:

    rows = None -> phase_name
    rows = (0, 10) -> phase_name_rows_(0,10)
    rows = (20, 60) -> phase_name_rows_(20,60)
    """
    if rows == None:
        return ""
    else:
        x0 = str(rows[0])
        x1 = str(rows[1])
        return "__rows_(" + x0 + "," + x1 + ")"


def parallel_front_edge_mask_rows_tag_from_parallel_front_edge_mask_rows(
    parallel_front_edge_mask_rows
):
    """Generate a parallel_front_edge_mask_rows tag, to customize phase names based on the number of rows in the charge
    injection region at the front edge of the parallel clocking direction are masked during the fit,

    This changes the phase name 'phase_name' as follows:

    parallel_front_edge_mask_rows = None -> phase_name
    parallel_front_edge_mask_rows = (0, 10) -> phase_name_parallel_front_edge_mask_rows_(0,10)
    parallel_front_edge_mask_rows = (20, 60) -> phase_name_parallel_front_edge_mask_rows_(20,60)
    """
    if parallel_front_edge_mask_rows == None:
        return ""
    else:
        x0 = str(parallel_front_edge_mask_rows[0])
        x1 = str(parallel_front_edge_mask_rows[1])
        return "__par_front_mask_rows_(" + x0 + "," + x1 + ")"


def parallel_trails_mask_rows_tag_from_parallel_trails_mask_rows(
    parallel_trails_mask_rows
):
    """Generate a parallel_trails_mask_rows tag, to customize phase names based on the number of rows in the charge
    injection region in the trails of the parallel clocking direction are masked during the fit,

    This changes the phase name 'phase_name' as follows:

    parallel_trails_mask_rows = None -> phase_name
    parallel_trails_mask_rows = (0, 10) -> phase_name_parallel_trails_mask_rows_(0,10)
    parallel_trails_mask_rows = (20, 60) -> phase_name_parallel_trails_mask_rows_(20,60)
    """
    if parallel_trails_mask_rows == None:
        return ""
    else:
        x0 = str(parallel_trails_mask_rows[0])
        x1 = str(parallel_trails_mask_rows[1])
        return "__par_trails_mask_rows_(" + x0 + "," + x1 + ")"


def serial_front_edge_mask_columns_tag_from_serial_front_edge_mask_columns(
    serial_front_edge_mask_columns
):
    """Generate a serial_front_edge_mask_columns tag, to customize phase names based on the number of columns in the
    charge  injection region at the front edge of the serial clocking direction are masked during the fit,

    This changes the phase name 'phase_name' as follows:

    serial_front_edge_mask_columns = None -> phase_name
    serial_front_edge_mask_columns = (0, 10) -> phase_name_serial_front_edge_mask_columns_(0,10)
    serial_front_edge_mask_columns = (20, 60) -> phase_name_serial_front_edge_mask_columns_(20,60)
    """
    if serial_front_edge_mask_columns == None:
        return ""
    else:
        x0 = str(serial_front_edge_mask_columns[0])
        x1 = str(serial_front_edge_mask_columns[1])
        return "__ser_front_mask_col_(" + x0 + "," + x1 + ")"


def serial_trails_mask_columns_tag_from_serial_trails_mask_columns(
    serial_trails_mask_columns
):
    """Generate a serial_trails_mask_columns tag, to customize phase names based on the number of columns in the charge
    injection region in the trails of the serial clocking direction are masked during the fit,

    This changes the phase name 'phase_name' as follows:

    serial_trails_mask_columns = None -> phase_name
    serial_trails_mask_columns = (0, 10) -> phase_name_serial_trails_mask_columns_(0,10)
    serial_trails_mask_columns = (20, 60) -> phase_name_serial_trails_mask_columns_(20,60)
    """
    if serial_trails_mask_columns == None:
        return ""
    else:
        x0 = str(serial_trails_mask_columns[0])
        x1 = str(serial_trails_mask_columns[1])
        return "__ser_trails_mask_col_(" + x0 + "," + x1 + ")"


def parallel_total_density_range_tag_from_parallel_total_density_range(
    parallel_total_density_range
):
    """Generate a parallel_total_density_range tag, to customize phase names based on the range of values in total \
    density that are allowed for the non-linear search in the parallel direction.

    This changes the phase name 'phase_name' as follows:

    parallel_total_density_range = None -> phase_name
    parallel_total_density_range = (0, 10) -> phase_name_parallel_total_density_range_(0,10)
    parallel_total_density_range = (20, 60) -> phase_name_parallel_total_density_range_(20,60)
    """
    if parallel_total_density_range == None:
        return ""
    else:
        x0 = str(parallel_total_density_range[0])
        x1 = str(parallel_total_density_range[1])
        return "__par_range_(" + x0 + "," + x1 + ")"


def serial_total_density_range_tag_from_serial_total_density_range(
    serial_total_density_range
):
    """Generate a serial_total_density_range tag, to customize phase names based on the range of values in total \
    density that are allowed for the non-linear search in the serial direction.

    This changes the phase name 'phase_name' as follows:

    serial_total_density_range = None -> phase_name
    serial_total_density_range = (0, 10) -> phase_name_serial_total_density_range_(0,10)
    serial_total_density_range = (20, 60) -> phase_name_serial_total_density_range_(20,60)
    """
    if serial_total_density_range == None:
        return ""
    else:
        x0 = str(serial_total_density_range[0])
        x1 = str(serial_total_density_range[1])
        return "__ser_range_(" + x0 + "," + x1 + ")"


def cosmic_ray_buffer_tag_from_cosmic_ray_buffers(
    cosmic_ray_parallel_buffer, cosmic_ray_serial_buffer, cosmic_ray_diagonal_buffer
):
    """Generate a cosmic ray buffer tag, to customize phase names based on the size of the cosmic ray masks in the \
    parallel, serial and diagonal directions

    This changes the phase name 'phase_name' as follows:

    cosmic_ray_parallel_buffer = 1, cosmic_ray_serial_buffer=2, cosmic_ray_diagonal_buffer=3 = -> phase_name_cr_p1s2d3
    cosmic_ray_parallel_buffer = 10, cosmic_ray_serial_buffer=5, cosmic_ray_diagonal_buffer=1 = -> phase_name_cr_p10s5d1
    """

    if (
        cosmic_ray_diagonal_buffer is None
        and cosmic_ray_serial_buffer is None
        and cosmic_ray_diagonal_buffer is None
    ):
        return ""

    if cosmic_ray_parallel_buffer is None:
        cosmic_ray_parallel_buffer_tag = ""
    else:
        cosmic_ray_parallel_buffer_tag = "p" + str(cosmic_ray_parallel_buffer)

    if cosmic_ray_serial_buffer is None:
        cosmic_ray_serial_buffer_tag = ""
    else:
        cosmic_ray_serial_buffer_tag = "s" + str(cosmic_ray_serial_buffer)

    if cosmic_ray_diagonal_buffer is None:
        cosmic_ray_diagonal_buffer_tag = ""
    else:
        cosmic_ray_diagonal_buffer_tag = "d" + str(cosmic_ray_diagonal_buffer)

    return (
        "__cr_"
        + cosmic_ray_parallel_buffer_tag
        + cosmic_ray_serial_buffer_tag
        + cosmic_ray_diagonal_buffer_tag
    )
