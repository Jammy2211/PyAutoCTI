import autocti as ac


class TestTags:
    def test__columns_tag(self):

        settings = ac.PhaseSettingsCIImaging(columns=None)
        assert settings.columns_tag == ""
        settings = ac.PhaseSettingsCIImaging(columns=(10, 20))
        assert settings.columns_tag == "__cols_(10,20)"
        settings = ac.PhaseSettingsCIImaging(columns=(60, 65))
        assert settings.columns_tag == "__cols_(60,65)"

    def test__rows_tag(self):

        settings = ac.PhaseSettingsCIImaging(rows=None)
        assert settings.rows_tag == ""
        settings = ac.PhaseSettingsCIImaging(rows=(0, 5))
        assert settings.rows_tag == "__rows_(0,5)"
        settings = ac.PhaseSettingsCIImaging(rows=(10, 20))
        assert settings.rows_tag == "__rows_(10,20)"

    def test__parallel_front_edge_mask_rows_tag(self):

        settings = ac.PhaseSettingsCIImaging(parallel_front_edge_mask_rows=None)
        assert settings.parallel_front_edge_mask_rows_tag == ""
        settings = ac.PhaseSettingsCIImaging(parallel_front_edge_mask_rows=(0, 5))
        assert (
            settings.parallel_front_edge_mask_rows_tag == "__par_front_mask_rows_(0,5)"
        )
        settings = ac.PhaseSettingsCIImaging(parallel_front_edge_mask_rows=(10, 20))
        assert (
            settings.parallel_front_edge_mask_rows_tag
            == "__par_front_mask_rows_(10,20)"
        )

    def test__parallel_trails_mask_rows_tag(self):

        settings = ac.PhaseSettingsCIImaging(parallel_trails_mask_rows=None)
        assert settings.parallel_trails_mask_rows_tag == ""

        settings = ac.PhaseSettingsCIImaging(parallel_trails_mask_rows=(0, 5))
        assert settings.parallel_trails_mask_rows_tag == "__par_trails_mask_rows_(0,5)"
        settings = ac.PhaseSettingsCIImaging(parallel_trails_mask_rows=(10, 20))
        assert (
            settings.parallel_trails_mask_rows_tag == "__par_trails_mask_rows_(10,20)"
        )

    def test__serial_front_edge_mask_columns_tag(self):

        settings = ac.PhaseSettingsCIImaging(serial_front_edge_mask_columns=None)
        assert settings.serial_front_edge_mask_columns_tag == ""

        settings = ac.PhaseSettingsCIImaging(serial_front_edge_mask_columns=(0, 5))
        assert (
            settings.serial_front_edge_mask_columns_tag == "__ser_front_mask_col_(0,5)"
        )

        settings = ac.PhaseSettingsCIImaging(serial_front_edge_mask_columns=(10, 20))
        assert (
            settings.serial_front_edge_mask_columns_tag
            == "__ser_front_mask_col_(10,20)"
        )

    def test__serial_trails_mask_columns_tag(self):

        settings = ac.PhaseSettingsCIImaging(serial_trails_mask_columns=None)
        assert settings.serial_trails_mask_columns_tag == ""

        settings = ac.PhaseSettingsCIImaging(serial_trails_mask_columns=(0, 5))
        assert settings.serial_trails_mask_columns_tag == "__ser_trails_mask_col_(0,5)"

        settings = ac.PhaseSettingsCIImaging(serial_trails_mask_columns=(10, 20))
        assert (
            settings.serial_trails_mask_columns_tag == "__ser_trails_mask_col_(10,20)"
        )

    def test__parallel_total_density_range_tag(self):

        settings = ac.PhaseSettingsCIImaging(parallel_total_density_range=None)
        assert settings.parallel_total_density_range_tag == ""
        settings = ac.PhaseSettingsCIImaging(parallel_total_density_range=(0, 5))
        assert settings.parallel_total_density_range_tag == "__par_range_(0,5)"
        settings = ac.PhaseSettingsCIImaging(parallel_total_density_range=(10, 20))
        assert settings.parallel_total_density_range_tag == "__par_range_(10,20)"

    def test__serial_total_density_range_tag(self):

        settings = ac.PhaseSettingsCIImaging(serial_total_density_range=None)
        assert settings.serial_total_density_range_tag == ""
        settings = ac.PhaseSettingsCIImaging(serial_total_density_range=(0, 5))
        assert settings.serial_total_density_range_tag == "__ser_range_(0,5)"
        settings = ac.PhaseSettingsCIImaging(serial_total_density_range=(10, 20))
        assert settings.serial_total_density_range_tag == "__ser_range_(10,20)"

    def test__cosmic_ray_buffer_tag(self):

        settings = ac.PhaseSettingsCIImaging(
            cosmic_ray_parallel_buffer=None,
            cosmic_ray_serial_buffer=None,
            cosmic_ray_diagonal_buffer=None,
        )
        assert settings.cosmic_ray_buffer_tag == ""

        settings = ac.PhaseSettingsCIImaging(
            cosmic_ray_parallel_buffer=1,
            cosmic_ray_serial_buffer=None,
            cosmic_ray_diagonal_buffer=3,
        )
        assert settings.cosmic_ray_buffer_tag == "__cr_p1d3"

        settings = ac.PhaseSettingsCIImaging(
            cosmic_ray_parallel_buffer=10,
            cosmic_ray_serial_buffer=20,
            cosmic_ray_diagonal_buffer=None,
        )
        assert settings.cosmic_ray_buffer_tag == "__cr_p10s20"

        settings = ac.PhaseSettingsCIImaging(
            cosmic_ray_parallel_buffer=1,
            cosmic_ray_serial_buffer=2,
            cosmic_ray_diagonal_buffer=3,
        )
        assert settings.cosmic_ray_buffer_tag == "__cr_p1s2d3"

        settings = ac.PhaseSettingsCIImaging(
            cosmic_ray_parallel_buffer=10,
            cosmic_ray_serial_buffer=5,
            cosmic_ray_diagonal_buffer=1,
        )
        assert settings.cosmic_ray_buffer_tag == "__cr_p10s5d1"

    def test__mixture_of_values(self):

        settings = ac.PhaseSettingsCIImaging(
            columns=(0, 1),
            rows=(0, 1),
            parallel_front_edge_mask_rows=None,
            parallel_trails_mask_rows=None,
            serial_front_edge_mask_columns=None,
            serial_trails_mask_columns=None,
            parallel_total_density_range=None,
            serial_total_density_range=None,
            cosmic_ray_parallel_buffer=None,
            cosmic_ray_serial_buffer=None,
            cosmic_ray_diagonal_buffer=None,
        )

        assert settings.phase_tag == "settings__cols_(0,1)__rows_(0,1)"

        settings = ac.PhaseSettingsCIImaging(
            columns=(1, 2),
            rows=(0, 1),
            parallel_front_edge_mask_rows=(0, 1),
            parallel_trails_mask_rows=None,
            serial_front_edge_mask_columns=None,
            serial_trails_mask_columns=(5, 10),
            parallel_total_density_range=None,
            serial_total_density_range=None,
            cosmic_ray_parallel_buffer=None,
            cosmic_ray_serial_buffer=None,
            cosmic_ray_diagonal_buffer=None,
        )

        assert (
            settings.phase_tag
            == "settings__cols_(1,2)__rows_(0,1)__par_front_mask_rows_(0,1)__ser_trails_mask_col_(5,10)"
        )

        settings = ac.PhaseSettingsCIImaging(
            columns=None,
            rows=(0, 1),
            parallel_front_edge_mask_rows=None,
            parallel_trails_mask_rows=None,
            serial_front_edge_mask_columns=None,
            serial_trails_mask_columns=None,
            parallel_total_density_range=None,
            serial_total_density_range=None,
            cosmic_ray_parallel_buffer=1,
            cosmic_ray_serial_buffer=2,
            cosmic_ray_diagonal_buffer=3,
        )

        assert settings.phase_tag == "settings__rows_(0,1)__cr_p1s2d3"

        settings = ac.PhaseSettingsCIImaging(
            columns=None,
            rows=(1, 2),
            parallel_front_edge_mask_rows=None,
            parallel_trails_mask_rows=None,
            serial_front_edge_mask_columns=None,
            serial_trails_mask_columns=None,
            parallel_total_density_range=None,
            serial_total_density_range=None,
            cosmic_ray_parallel_buffer=4,
            cosmic_ray_serial_buffer=5,
            cosmic_ray_diagonal_buffer=6,
        )

        assert settings.phase_tag == "settings__rows_(1,2)__cr_p4s5d6"

        settings = ac.PhaseSettingsCIImaging(
            columns=None,
            rows=(1, 2),
            parallel_front_edge_mask_rows=None,
            parallel_trails_mask_rows=None,
            serial_front_edge_mask_columns=None,
            serial_trails_mask_columns=None,
            parallel_total_density_range=(0, 1),
            serial_total_density_range=(2, 3),
            cosmic_ray_parallel_buffer=4,
            cosmic_ray_serial_buffer=5,
            cosmic_ray_diagonal_buffer=6,
        )

        assert (
            settings.phase_tag
            == "settings__rows_(1,2)__par_range_(0,1)__ser_range_(2,3)__cr_p4s5d6"
        )

        settings = ac.PhaseSettingsCIImaging(
            columns=None,
            rows=(0, 1),
            parallel_front_edge_mask_rows=None,
            parallel_trails_mask_rows=(1, 2),
            serial_front_edge_mask_columns=(2, 4),
            serial_trails_mask_columns=None,
            parallel_total_density_range=None,
            serial_total_density_range=None,
            cosmic_ray_parallel_buffer=4,
            cosmic_ray_serial_buffer=5,
            cosmic_ray_diagonal_buffer=6,
        )

        assert (
            settings.phase_tag
            == "settings__rows_(0,1)__par_trails_mask_rows_(1,2)__ser_front_mask_col_(2,4)__cr_p4s5d6"
        )
