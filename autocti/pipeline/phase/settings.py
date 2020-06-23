from autoconf import conf


class PhaseSettingsCIImaging:
    def __init__(
        self,
        columns=None,
        rows=None,
        parallel_front_edge_mask_rows=None,
        parallel_trails_mask_rows=None,
        parallel_total_density_range=None,
        serial_front_edge_mask_columns=None,
        serial_trails_mask_columns=None,
        serial_total_density_range=None,
        cosmic_ray_parallel_buffer=10,
        cosmic_ray_serial_buffer=10,
        cosmic_ray_diagonal_buffer=3,
    ):

        self.columns = columns
        self.rows = rows
        self.parallel_front_edge_mask_rows = parallel_front_edge_mask_rows
        self.parallel_trails_mask_rows = parallel_trails_mask_rows
        self.serial_front_edge_mask_columns = serial_front_edge_mask_columns
        self.serial_trails_mask_columns = serial_trails_mask_columns
        self.parallel_total_density_range = parallel_total_density_range
        self.serial_total_density_range = serial_total_density_range
        self.cosmic_ray_parallel_buffer = cosmic_ray_parallel_buffer
        self.cosmic_ray_serial_buffer = cosmic_ray_serial_buffer
        self.cosmic_ray_diagonal_buffer = cosmic_ray_diagonal_buffer

    @property
    def phase_tag(self):

        return (
            conf.instance.tag.get("phase", "settings")
            + self.columns_tag
            + self.rows_tag
            + self.parallel_front_edge_mask_rows_tag
            + self.parallel_trails_mask_rows_tag
            + self.serial_front_edge_mask_columns_tag
            + self.serial_trails_mask_columns_tag
            + self.parallel_total_density_range_tag
            + self.serial_total_density_range_tag
            + self.cosmic_ray_buffer_tag
        )

    @property
    def columns_tag(self):
        """Generate a columns tag, to customize phase names based on the number of columns of simulator extracted in the fit,
        which is used to speed up parallel CTI fits.

        This changes the phase settings folder as follows:

        columns = None -> settings
        columns = 10 -> settings__col_10
        columns = 60 -> settings__col_60
        """
        if self.columns == None:
            return ""
        else:
            x0 = str(self.columns[0])
            x1 = str(self.columns[1])
            return f"__{conf.instance.tag.get('phase', 'columns')}_({x0},{x1})"

    @property
    def rows_tag(self):
        """Generate a rows tag, to customize phase names based on the number of rows of simulator extracted in the fit,
        which is used to speed up serial CTI fits.

        This changes the phase settings folder as follows:

        rows = None -> settings
        rows = (0, 10) -> settings__rows_(0,10)
        rows = (20, 60) -> settings__rows_(20,60)
        """
        if self.rows == None:
            return ""
        else:
            x0 = str(self.rows[0])
            x1 = str(self.rows[1])
            return f"__{conf.instance.tag.get('phase', 'rows')}_({x0},{x1})"

    @property
    def parallel_front_edge_mask_rows_tag(self):
        """Generate a parallel_front_edge_mask_rows tag, to customize phase names based on the number of rows in the charge
        injection region at the front edge of the parallel clocking direction are masked during the fit,

        This changes the phase settings folder as follows:

        parallel_front_edge_mask_rows = None -> settings
        parallel_front_edge_mask_rows = (0, 10) -> settings__parallel_front_edge_mask_rows_(0,10)
        parallel_front_edge_mask_rows = (20, 60) -> settings__parallel_front_edge_mask_rows_(20,60)
        """
        if self.parallel_front_edge_mask_rows == None:
            return ""
        else:
            x0 = str(self.parallel_front_edge_mask_rows[0])
            x1 = str(self.parallel_front_edge_mask_rows[1])
            return f"__{conf.instance.tag.get('phase', 'parallel_front_edge_mask_rows')}_({x0},{x1})"

    @property
    def parallel_trails_mask_rows_tag(self):
        """Generate a parallel_trails_mask_rows tag, to customize phase names based on the number of rows in the charge
        injection region in the trails of the parallel clocking direction are masked during the fit,

        This changes the phase settings folder as follows:

        parallel_trails_mask_rows = None -> settings
        parallel_trails_mask_rows = (0, 10) -> settings__parallel_trails_mask_rows_(0,10)
        parallel_trails_mask_rows = (20, 60) -> settings__parallel_trails_mask_rows_(20,60)
        """
        if self.parallel_trails_mask_rows == None:
            return ""
        else:
            x0 = str(self.parallel_trails_mask_rows[0])
            x1 = str(self.parallel_trails_mask_rows[1])
            return f"__{conf.instance.tag.get('phase', 'parallel_trails_mask_rows')}_({x0},{x1})"

    @property
    def serial_front_edge_mask_columns_tag(self):
        """Generate a serial_front_edge_mask_columns tag, to customize phase names based on the number of columns in the
        charge  injection region at the front edge of the serial clocking direction are masked during the fit,

        This changes the phase settings folder as follows:

        serial_front_edge_mask_columns = None -> settings
        serial_front_edge_mask_columns = (0, 10) -> settings__serial_front_edge_mask_columns_(0,10)
        serial_front_edge_mask_columns = (20, 60) -> settings__serial_front_edge_mask_columns_(20,60)
        """
        if self.serial_front_edge_mask_columns == None:
            return ""
        else:
            x0 = str(self.serial_front_edge_mask_columns[0])
            x1 = str(self.serial_front_edge_mask_columns[1])
            return f"__{conf.instance.tag.get('phase', 'serial_front_edge_mask_rows')}_({x0},{x1})"

    @property
    def serial_trails_mask_columns_tag(self):
        """Generate a serial_trails_mask_columns tag, to customize phase names based on the number of columns in the charge
        injection region in the trails of the serial clocking direction are masked during the fit,

        This changes the phase settings folder as follows:

        serial_trails_mask_columns = None -> settings
        serial_trails_mask_columns = (0, 10) -> settings__serial_trails_mask_columns_(0,10)
        serial_trails_mask_columns = (20, 60) -> settings__serial_trails_mask_columns_(20,60)
        """
        if self.serial_trails_mask_columns == None:
            return ""
        else:
            x0 = str(self.serial_trails_mask_columns[0])
            x1 = str(self.serial_trails_mask_columns[1])
            return f"__{conf.instance.tag.get('phase', 'serial_trails_mask_rows')}_({x0},{x1})"

    @property
    def parallel_total_density_range_tag(self):
        """Generate a parallel_total_density_range tag, to customize phase names based on the range of values in total \
        density that are allowed for the non-linear search in the parallel direction.

        This changes the phase settings folder as follows:

        parallel_total_density_range = None -> settings
        parallel_total_density_range = (0, 10) -> settings__parallel_total_density_range_(0,10)
        parallel_total_density_range = (20, 60) -> settings__parallel_total_density_range_(20,60)
        """
        if self.parallel_total_density_range == None:
            return ""
        else:
            x0 = str(self.parallel_total_density_range[0])
            x1 = str(self.parallel_total_density_range[1])
            return f"__{conf.instance.tag.get('phase', 'parallel_total_density_range')}_({x0},{x1})"

    @property
    def serial_total_density_range_tag(self):
        """Generate a serial_total_density_range tag, to customize phase names based on the range of values in total \
        density that are allowed for the non-linear search in the serial direction.

        This changes the phase settings folder as follows:

        serial_total_density_range = None -> settings
        serial_total_density_range = (0, 10) -> settings__serial_total_density_range_(0,10)
        serial_total_density_range = (20, 60) -> settings__serial_total_density_range_(20,60)
        """
        if self.serial_total_density_range == None:
            return ""
        else:
            x0 = str(self.serial_total_density_range[0])
            x1 = str(self.serial_total_density_range[1])
            return f"__{conf.instance.tag.get('phase', 'serial_total_density_range')}_({x0},{x1})"

    @property
    def cosmic_ray_buffer_tag(self):
        """Generate a cosmic ray buffer tag, to customize phase names based on the size of the cosmic ray masks in the \
        parallel, serial and diagonal directions

        This changes the phase settings folder as follows:

        cosmic_ray_parallel_buffer = 1, cosmic_ray_serial_buffer=2, cosmic_ray_diagonal_buffer=3 = -> settings__cr_p1s2d3
        cosmic_ray_parallel_buffer = 10, cosmic_ray_serial_buffer=5, cosmic_ray_diagonal_buffer=1 = -> settings__cr_p10s5d1
        """

        if (
            self.cosmic_ray_diagonal_buffer is None
            and self.cosmic_ray_serial_buffer is None
            and self.cosmic_ray_diagonal_buffer is None
        ):
            return ""

        if self.cosmic_ray_parallel_buffer is None:
            cosmic_ray_parallel_buffer_tag = ""
        else:
            cosmic_ray_parallel_buffer_tag = f"{conf.instance.tag.get('phase', 'cosmic_ray_parallel_buffer')}{self.cosmic_ray_parallel_buffer}"

        if self.cosmic_ray_serial_buffer is None:
            cosmic_ray_serial_buffer_tag = ""
        else:
            cosmic_ray_serial_buffer_tag = f"{conf.instance.tag.get('phase', 'cosmic_ray_serial_buffer')}{self.cosmic_ray_serial_buffer}"

        if self.cosmic_ray_diagonal_buffer is None:
            cosmic_ray_diagonal_buffer_tag = ""
        else:
            cosmic_ray_diagonal_buffer_tag = f"{conf.instance.tag.get('phase', 'cosmic_ray_diagonal_buffer')}{self.cosmic_ray_diagonal_buffer}"

        return (
            "__"
            + conf.instance.tag.get("phase", "cosmic_ray_buffer")
            + "_"
            + cosmic_ray_parallel_buffer_tag
            + cosmic_ray_serial_buffer_tag
            + cosmic_ray_diagonal_buffer_tag
        )
