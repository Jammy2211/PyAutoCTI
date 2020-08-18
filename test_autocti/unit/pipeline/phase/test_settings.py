import autocti as ac


class TestTags:
    def test__parallel_front_edge_mask_rows_tag(self):

        settings = ac.SettingsPhaseCIImaging(parallel_front_edge_mask_rows=None)
        assert settings.parallel_front_edge_mask_rows_tag == ""
        settings = ac.SettingsPhaseCIImaging(parallel_front_edge_mask_rows=(0, 5))
        assert (
            settings.parallel_front_edge_mask_rows_tag == "__par_front_mask_rows_(0,5)"
        )
        settings = ac.SettingsPhaseCIImaging(parallel_front_edge_mask_rows=(10, 20))
        assert (
            settings.parallel_front_edge_mask_rows_tag
            == "__par_front_mask_rows_(10,20)"
        )

    def test__parallel_trails_mask_rows_tag(self):

        settings = ac.SettingsPhaseCIImaging(parallel_trails_mask_rows=None)
        assert settings.parallel_trails_mask_rows_tag == ""

        settings = ac.SettingsPhaseCIImaging(parallel_trails_mask_rows=(0, 5))
        assert settings.parallel_trails_mask_rows_tag == "__par_trails_mask_rows_(0,5)"
        settings = ac.SettingsPhaseCIImaging(parallel_trails_mask_rows=(10, 20))
        assert (
            settings.parallel_trails_mask_rows_tag == "__par_trails_mask_rows_(10,20)"
        )

    def test__serial_front_edge_mask_columns_tag(self):

        settings = ac.SettingsPhaseCIImaging(serial_front_edge_mask_columns=None)
        assert settings.serial_front_edge_mask_columns_tag == ""

        settings = ac.SettingsPhaseCIImaging(serial_front_edge_mask_columns=(0, 5))
        assert (
            settings.serial_front_edge_mask_columns_tag == "__ser_front_mask_col_(0,5)"
        )

        settings = ac.SettingsPhaseCIImaging(serial_front_edge_mask_columns=(10, 20))
        assert (
            settings.serial_front_edge_mask_columns_tag
            == "__ser_front_mask_col_(10,20)"
        )

    def test__serial_trails_mask_columns_tag(self):

        settings = ac.SettingsPhaseCIImaging(serial_trails_mask_columns=None)
        assert settings.serial_trails_mask_columns_tag == ""

        settings = ac.SettingsPhaseCIImaging(serial_trails_mask_columns=(0, 5))
        assert settings.serial_trails_mask_columns_tag == "__ser_trails_mask_col_(0,5)"

        settings = ac.SettingsPhaseCIImaging(serial_trails_mask_columns=(10, 20))
        assert (
            settings.serial_trails_mask_columns_tag == "__ser_trails_mask_col_(10,20)"
        )

    def test__parallel_total_density_range_tag(self):

        settings = ac.SettingsPhaseCIImaging(parallel_total_density_range=None)
        assert settings.parallel_total_density_range_tag == ""
        settings = ac.SettingsPhaseCIImaging(parallel_total_density_range=(0, 5))
        assert settings.parallel_total_density_range_tag == "__par_range_(0,5)"
        settings = ac.SettingsPhaseCIImaging(parallel_total_density_range=(10, 20))
        assert settings.parallel_total_density_range_tag == "__par_range_(10,20)"

    def test__serial_total_density_range_tag(self):

        settings = ac.SettingsPhaseCIImaging(serial_total_density_range=None)
        assert settings.serial_total_density_range_tag == ""
        settings = ac.SettingsPhaseCIImaging(serial_total_density_range=(0, 5))
        assert settings.serial_total_density_range_tag == "__ser_range_(0,5)"
        settings = ac.SettingsPhaseCIImaging(serial_total_density_range=(10, 20))
        assert settings.serial_total_density_range_tag == "__ser_range_(10,20)"

    def test__mixture_of_values(self):

        settings = ac.SettingsPhaseCIImaging(
            mask=ac.SettingsMask(
                cosmic_ray_parallel_buffer=None,
                cosmic_ray_serial_buffer=None,
                cosmic_ray_diagonal_buffer=None,
            ),
            masked_ci_imaging=ac.ci.SettingsMaskedCIImaging(
                parallel_columns=(0, 1), serial_rows=(0, 1)
            ),
            parallel_front_edge_mask_rows=None,
            parallel_trails_mask_rows=None,
            serial_front_edge_mask_columns=None,
            serial_trails_mask_columns=None,
            parallel_total_density_range=None,
            serial_total_density_range=None,
        )

        assert settings.phase_tag == "settings__cols_(0,1)__rows_(0,1)"

        settings = ac.SettingsPhaseCIImaging(
            mask=ac.SettingsMask(
                cosmic_ray_parallel_buffer=None,
                cosmic_ray_serial_buffer=None,
                cosmic_ray_diagonal_buffer=None,
            ),
            masked_ci_imaging=ac.ci.SettingsMaskedCIImaging(
                parallel_columns=(1, 2), serial_rows=(0, 1)
            ),
            parallel_front_edge_mask_rows=(0, 1),
            parallel_trails_mask_rows=None,
            serial_front_edge_mask_columns=None,
            serial_trails_mask_columns=(5, 10),
            parallel_total_density_range=None,
            serial_total_density_range=None,
        )

        assert (
            settings.phase_tag
            == "settings__cols_(1,2)__rows_(0,1)__par_front_mask_rows_(0,1)__ser_trails_mask_col_(5,10)"
        )

        settings = ac.SettingsPhaseCIImaging(
            mask=ac.SettingsMask(
                cosmic_ray_parallel_buffer=1,
                cosmic_ray_serial_buffer=2,
                cosmic_ray_diagonal_buffer=3,
            ),
            masked_ci_imaging=ac.ci.SettingsMaskedCIImaging(
                parallel_columns=None, serial_rows=(0, 1)
            ),
            parallel_front_edge_mask_rows=None,
            parallel_trails_mask_rows=None,
            serial_front_edge_mask_columns=None,
            serial_trails_mask_columns=None,
            parallel_total_density_range=None,
            serial_total_density_range=None,
        )

        assert settings.phase_tag == "settings__cr_p1s2d3__rows_(0,1)"

        settings = ac.SettingsPhaseCIImaging(
            mask=ac.SettingsMask(
                cosmic_ray_parallel_buffer=4,
                cosmic_ray_serial_buffer=5,
                cosmic_ray_diagonal_buffer=6,
            ),
            masked_ci_imaging=ac.ci.SettingsMaskedCIImaging(
                parallel_columns=None, serial_rows=(1, 2)
            ),
            parallel_front_edge_mask_rows=None,
            parallel_trails_mask_rows=None,
            serial_front_edge_mask_columns=None,
            serial_trails_mask_columns=None,
            parallel_total_density_range=None,
            serial_total_density_range=None,
        )

        assert settings.phase_tag == "settings__cr_p4s5d6__rows_(1,2)"

        settings = ac.SettingsPhaseCIImaging(
            mask=ac.SettingsMask(
                cosmic_ray_parallel_buffer=4,
                cosmic_ray_serial_buffer=5,
                cosmic_ray_diagonal_buffer=6,
            ),
            masked_ci_imaging=ac.ci.SettingsMaskedCIImaging(
                parallel_columns=None, serial_rows=(1, 2)
            ),
            parallel_front_edge_mask_rows=None,
            parallel_trails_mask_rows=None,
            serial_front_edge_mask_columns=None,
            serial_trails_mask_columns=None,
            parallel_total_density_range=(0, 1),
            serial_total_density_range=(2, 3),
        )

        assert (
            settings.phase_tag
            == "settings__cr_p4s5d6__rows_(1,2)__par_range_(0,1)__ser_range_(2,3)"
        )

        settings = ac.SettingsPhaseCIImaging(
            mask=ac.SettingsMask(
                cosmic_ray_parallel_buffer=4,
                cosmic_ray_serial_buffer=5,
                cosmic_ray_diagonal_buffer=6,
            ),
            masked_ci_imaging=ac.ci.SettingsMaskedCIImaging(
                parallel_columns=None, serial_rows=(0, 1)
            ),
            parallel_front_edge_mask_rows=None,
            parallel_trails_mask_rows=(1, 2),
            serial_front_edge_mask_columns=(2, 4),
            serial_trails_mask_columns=None,
            parallel_total_density_range=None,
            serial_total_density_range=None,
        )

        assert (
            settings.phase_tag
            == "settings__cr_p4s5d6__rows_(0,1)__par_trails_mask_rows_(1,2)__ser_front_mask_col_(2,4)"
        )
