from autocti.pipeline import tagging


class TestPhaseTag:
    def test__mixture_of_values(self):

        phase_tag = tagging.phase_tag_from_phase_settings(
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

        assert phase_tag == "phase_tag__columns_(0,1)__rows_(0,1)"

        phase_tag = tagging.phase_tag_from_phase_settings(
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
            phase_tag
            == "phase_tag__columns_(1,2)__rows_(0,1)__par_front_mask_rows_(0,1)__ser_trails_mask_col_(5,10)"
        )

        phase_tag = tagging.phase_tag_from_phase_settings(
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

        assert phase_tag == "phase_tag__rows_(0,1)__cr_p1s2d3"

        phase_tag = tagging.phase_tag_from_phase_settings(
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

        assert phase_tag == "phase_tag__rows_(1,2)__cr_p4s5d6"

        phase_tag = tagging.phase_tag_from_phase_settings(
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
            phase_tag
            == "phase_tag__rows_(1,2)__par_range_(0,1)__ser_range_(2,3)__cr_p4s5d6"
        )

        phase_tag = tagging.phase_tag_from_phase_settings(
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
            phase_tag
            == "phase_tag__rows_(0,1)__par_trails_mask_rows_(1,2)__ser_front_mask_col_(2,4)__cr_p4s5d6"
        )


class TestTaggers:
    def test__columns_tagger(self):

        tag = tagging.columns_tag_from_columns(columns=None)
        assert tag == ""
        tag = tagging.columns_tag_from_columns(columns=(10, 20))
        assert tag == "__columns_(10,20)"
        tag = tagging.columns_tag_from_columns(columns=(60, 65))
        assert tag == "__columns_(60,65)"

    def test__rows_tagger(self):

        tag = tagging.rows_tag_from_rows(rows=None)
        assert tag == ""
        tag = tagging.rows_tag_from_rows(rows=(0, 5))
        assert tag == "__rows_(0,5)"
        tag = tagging.rows_tag_from_rows(rows=(10, 20))
        assert tag == "__rows_(10,20)"

    def test__parallel_front_edge_mask_rows_tagger(self):

        tag = tagging.parallel_front_edge_mask_rows_tag_from_parallel_front_edge_mask_rows(
            parallel_front_edge_mask_rows=None
        )
        assert tag == ""
        tag = tagging.parallel_front_edge_mask_rows_tag_from_parallel_front_edge_mask_rows(
            parallel_front_edge_mask_rows=(0, 5)
        )
        assert tag == "__par_front_mask_rows_(0,5)"
        tag = tagging.parallel_front_edge_mask_rows_tag_from_parallel_front_edge_mask_rows(
            parallel_front_edge_mask_rows=(10, 20)
        )
        assert tag == "__par_front_mask_rows_(10,20)"

    def test__parallel_trails_mask_rows_tagger(self):

        tag = tagging.parallel_trails_mask_rows_tag_from_parallel_trails_mask_rows(
            parallel_trails_mask_rows=None
        )
        assert tag == ""

        tag = tagging.parallel_trails_mask_rows_tag_from_parallel_trails_mask_rows(
            parallel_trails_mask_rows=(0, 5)
        )
        assert tag == "__par_trails_mask_rows_(0,5)"
        tag = tagging.parallel_trails_mask_rows_tag_from_parallel_trails_mask_rows(
            parallel_trails_mask_rows=(10, 20)
        )
        assert tag == "__par_trails_mask_rows_(10,20)"

    def test__serial_front_edge_mask_columns_tagger(self):

        tag = tagging.serial_front_edge_mask_columns_tag_from_serial_front_edge_mask_columns(
            serial_front_edge_mask_columns=None
        )
        assert tag == ""

        tag = tagging.serial_front_edge_mask_columns_tag_from_serial_front_edge_mask_columns(
            serial_front_edge_mask_columns=(0, 5)
        )
        assert tag == "__ser_front_mask_col_(0,5)"

        tag = tagging.serial_front_edge_mask_columns_tag_from_serial_front_edge_mask_columns(
            serial_front_edge_mask_columns=(10, 20)
        )
        assert tag == "__ser_front_mask_col_(10,20)"

    def test__serial_trails_mask_columns_tagger(self):

        tag = tagging.serial_trails_mask_columns_tag_from_serial_trails_mask_columns(
            serial_trails_mask_columns=None
        )
        assert tag == ""

        tag = tagging.serial_trails_mask_columns_tag_from_serial_trails_mask_columns(
            serial_trails_mask_columns=(0, 5)
        )
        assert tag == "__ser_trails_mask_col_(0,5)"

        tag = tagging.serial_trails_mask_columns_tag_from_serial_trails_mask_columns(
            serial_trails_mask_columns=(10, 20)
        )
        assert tag == "__ser_trails_mask_col_(10,20)"

    def test__parallel_total_density_range_tagger(self):

        tag = tagging.parallel_total_density_range_tag_from_parallel_total_density_range(
            parallel_total_density_range=None
        )
        assert tag == ""
        tag = tagging.parallel_total_density_range_tag_from_parallel_total_density_range(
            parallel_total_density_range=(0, 5)
        )
        assert tag == "__par_range_(0,5)"
        tag = tagging.parallel_total_density_range_tag_from_parallel_total_density_range(
            parallel_total_density_range=(10, 20)
        )
        assert tag == "__par_range_(10,20)"

    def test__serial_total_density_range_tagger(self):

        tag = tagging.serial_total_density_range_tag_from_serial_total_density_range(
            serial_total_density_range=None
        )
        assert tag == ""
        tag = tagging.serial_total_density_range_tag_from_serial_total_density_range(
            serial_total_density_range=(0, 5)
        )
        assert tag == "__ser_range_(0,5)"
        tag = tagging.serial_total_density_range_tag_from_serial_total_density_range(
            serial_total_density_range=(10, 20)
        )
        assert tag == "__ser_range_(10,20)"

    def test__cosmic_ray_buffer_tagger(self):

        tag = tagging.cosmic_ray_buffer_tag_from_cosmic_ray_buffers(
            cosmic_ray_parallel_buffer=None,
            cosmic_ray_serial_buffer=None,
            cosmic_ray_diagonal_buffer=None,
        )
        assert tag == ""

        tag = tagging.cosmic_ray_buffer_tag_from_cosmic_ray_buffers(
            cosmic_ray_parallel_buffer=1,
            cosmic_ray_serial_buffer=None,
            cosmic_ray_diagonal_buffer=3,
        )
        assert tag == "__cr_p1d3"

        tag = tagging.cosmic_ray_buffer_tag_from_cosmic_ray_buffers(
            cosmic_ray_parallel_buffer=10,
            cosmic_ray_serial_buffer=20,
            cosmic_ray_diagonal_buffer=None,
        )
        assert tag == "__cr_p10s20"

        tag = tagging.cosmic_ray_buffer_tag_from_cosmic_ray_buffers(
            cosmic_ray_parallel_buffer=1,
            cosmic_ray_serial_buffer=2,
            cosmic_ray_diagonal_buffer=3,
        )
        assert tag == "__cr_p1s2d3"

        tag = tagging.cosmic_ray_buffer_tag_from_cosmic_ray_buffers(
            cosmic_ray_parallel_buffer=10,
            cosmic_ray_serial_buffer=5,
            cosmic_ray_diagonal_buffer=1,
        )
        assert tag == "__cr_p10s5d1"
