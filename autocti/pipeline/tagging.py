def phase_tag_from_phase_settings(columns, rows, cosmic_ray_parallel_buffer, cosmic_ray_serial_buffer,
                                  cosmic_ray_diagonal_buffer):

    columns_tag = columns_tag_from_columns(columns=columns)
    rows_tag = rows_tag_from_rows(rows=rows)

    cosmic_ray_buffer_tag = cosmic_ray_buffer_tag_from_cosmic_ray_buffers(
        cosmic_ray_parallel_buffer=cosmic_ray_parallel_buffer, cosmic_ray_serial_buffer=cosmic_ray_serial_buffer,
        cosmic_ray_diagonal_buffer=cosmic_ray_diagonal_buffer)

    return columns_tag + rows_tag + cosmic_ray_buffer_tag

def columns_tag_from_columns(columns):
    """Generate a columns tag, to customize phase names based on the number of columns of data extracted in the fit,
    which is used to speed up parallel CTI fits.

    This changes the phase name 'phase_name' as follows:

    columns = None -> phase_name
    columns = 10 -> phase_name_col_10
    columns = 60 -> phase_name_col_60
    """
    if columns == None:
        return ''
    else:
        return '_col_' + str(int(columns))
    
def rows_tag_from_rows(rows):
    """Generate a rows tag, to customize phase names based on the number of rows of data extracted in the fit,
    which is used to speed up serial CTI fits.

    This changes the phase name 'phase_name' as follows:

    rows = None -> phase_name
    rows = (0, 10) -> phase_name_rows_(0,10)
    rows = (20, 60) -> phase_name_rows_(20,60)
    """
    if rows == None:
        return ''
    else:
        x0 = str(rows[0])
        x1 = str(rows[1])
        return ('_rows_(' + x0 + ',' + x1 + ')')

def cosmic_ray_buffer_tag_from_cosmic_ray_buffers(cosmic_ray_parallel_buffer, cosmic_ray_serial_buffer,
                                                  cosmic_ray_diagonal_buffer):
    """Generate a cosmic ray buffer tag, to customize phase names based on the size of the cosmic ray masks in the \
    parallel, serial and diagonal directions

    This changes the phase name 'phase_name' as follows:

    cosmic_ray_parallel_buffer = 1, cosmic_ray_serial_buffer=2, cosmic_ray_diagonal_buffer=3 = -> phase_name_cr_p1s2d3
    cosmic_ray_parallel_buffer = 10, cosmic_ray_serial_buffer=5, cosmic_ray_diagonal_buffer=1 = -> phase_name_cr_p10s5d1
    """

    if cosmic_ray_diagonal_buffer is None and cosmic_ray_serial_buffer is None and cosmic_ray_diagonal_buffer is None:
        return ''
    
    if cosmic_ray_parallel_buffer is None:
        cosmic_ray_parallel_buffer_tag = ''
    else:
        cosmic_ray_parallel_buffer_tag = 'p' + str(cosmic_ray_parallel_buffer)
    
    if cosmic_ray_serial_buffer is None:
        cosmic_ray_serial_buffer_tag = ''
    else:
        cosmic_ray_serial_buffer_tag = 's' + str(cosmic_ray_serial_buffer)

    if cosmic_ray_diagonal_buffer is None:
        cosmic_ray_diagonal_buffer_tag = ''
    else:
        cosmic_ray_diagonal_buffer_tag = 'd' + str(cosmic_ray_diagonal_buffer)

    return '_cr_' + cosmic_ray_parallel_buffer_tag + cosmic_ray_serial_buffer_tag + cosmic_ray_diagonal_buffer_tag
