import autocti as ac
from autocti import exc

import pytest


class TestSettingsCTI:
    def test__parallel_total_density_range_tag(self):

        settings = ac.SettingsCTI(parallel_total_density_range=None)
        assert settings.parallel_total_density_range_tag == ""
        settings = ac.SettingsCTI(parallel_total_density_range=(0, 5))
        assert settings.parallel_total_density_range_tag == "__par_range_(0,5)"
        settings = ac.SettingsCTI(parallel_total_density_range=(10, 20))
        assert settings.parallel_total_density_range_tag == "__par_range_(10,20)"

    def test__serial_total_density_range_tag(self):

        settings = ac.SettingsCTI(serial_total_density_range=None)
        assert settings.serial_total_density_range_tag == ""
        settings = ac.SettingsCTI(serial_total_density_range=(0, 5))
        assert settings.serial_total_density_range_tag == "__ser_range_(0,5)"
        settings = ac.SettingsCTI(serial_total_density_range=(10, 20))
        assert settings.serial_total_density_range_tag == "__ser_range_(10,20)"

    def test__tag(self):
        settings = ac.SettingsCTI(
            parallel_total_density_range=(10, 20), serial_total_density_range=(10, 20)
        )
        assert settings.tag == "cti[__par_range_(10,20)__ser_range_(10,20)]"

    def test__parallel_and_serial_checks_raise_exception(self, ci_imaging_7x7):

        settings = ac.SettingsCTI(parallel_total_density_range=(1.0, 2.0))

        parallel_traps = [
            ac.TrapInstantCaptureWrap(density=0.75),
            ac.TrapInstantCaptureWrap(density=0.75),
        ]
        serial_traps = []

        settings.check_total_density_within_range(
            parallel_traps=parallel_traps, serial_traps=serial_traps
        )

        parallel_traps = [
            ac.TrapInstantCaptureWrap(density=1.1),
            ac.TrapInstantCaptureWrap(density=1.1),
        ]

        with pytest.raises(exc.PriorException):
            settings.check_total_density_within_range(
                parallel_traps=parallel_traps, serial_traps=serial_traps
            )

        settings = ac.SettingsCTI(serial_total_density_range=(1.0, 2.0))

        parallel_traps = []
        serial_traps = [
            ac.TrapInstantCaptureWrap(density=0.75),
            ac.TrapInstantCaptureWrap(density=0.75),
        ]

        settings.check_total_density_within_range(
            parallel_traps=parallel_traps, serial_traps=serial_traps
        )

        serial_traps = [
            ac.TrapInstantCaptureWrap(density=1.1),
            ac.TrapInstantCaptureWrap(density=1.1),
        ]

        with pytest.raises(exc.PriorException):
            settings.check_total_density_within_range(
                parallel_traps=parallel_traps, serial_traps=serial_traps
            )


class TestTags:
    def test__mixture_of_values(self):

        settings = ac.SettingsPhaseCIImaging(
            settings_cti=ac.SettingsCTI(
                parallel_total_density_range=None, serial_total_density_range=None
            ),
            settings_mask=ac.SettingsMask(
                cosmic_ray_parallel_buffer=None,
                cosmic_ray_serial_buffer=None,
                cosmic_ray_diagonal_buffer=None,
            ),
            settings_ci_mask=ac.ci.SettingsCIMask(
                parallel_front_edge_rows=None,
                parallel_trails_rows=None,
                serial_front_edge_columns=None,
                serial_trails_columns=None,
            ),
            settings_masked_ci_imaging=ac.ci.SettingsMaskedCIImaging(
                parallel_columns=(0, 1), serial_rows=(0, 1)
            ),
        )

        assert (
            settings.phase_tag
            == "settings__cti[]_mask[]_ci_mask[]_ci_imaging[__cols_(0,1)__rows_(0,1)]"
        )

        settings = ac.SettingsPhaseCIImaging(
            settings_cti=ac.SettingsCTI(
                parallel_total_density_range=None, serial_total_density_range=None
            ),
            settings_mask=ac.SettingsMask(
                cosmic_ray_parallel_buffer=None,
                cosmic_ray_serial_buffer=None,
                cosmic_ray_diagonal_buffer=None,
            ),
            settings_ci_mask=ac.ci.SettingsCIMask(
                parallel_front_edge_rows=(0, 1),
                parallel_trails_rows=None,
                serial_front_edge_columns=None,
                serial_trails_columns=(5, 10),
            ),
            settings_masked_ci_imaging=ac.ci.SettingsMaskedCIImaging(
                parallel_columns=(1, 2), serial_rows=(0, 1)
            ),
        )

        assert (
            settings.phase_tag
            == "settings__cti[]_mask[]_ci_mask[__par_front_mask_rows_(0,1)__ser_trails_mask_col_(5,10)]_ci_imaging[__cols_(1,2)__rows_(0,1)]"
        )

        settings = ac.SettingsPhaseCIImaging(
            settings_cti=ac.SettingsCTI(
                parallel_total_density_range=None, serial_total_density_range=None
            ),
            settings_mask=ac.SettingsMask(
                cosmic_ray_parallel_buffer=1,
                cosmic_ray_serial_buffer=2,
                cosmic_ray_diagonal_buffer=3,
            ),
            settings_ci_mask=ac.ci.SettingsCIMask(
                parallel_front_edge_rows=None,
                parallel_trails_rows=None,
                serial_front_edge_columns=None,
                serial_trails_columns=None,
            ),
            settings_masked_ci_imaging=ac.ci.SettingsMaskedCIImaging(
                parallel_columns=None, serial_rows=(0, 1)
            ),
        )

        assert (
            settings.phase_tag
            == "settings__cti[]_mask[__cr_p1s2d3]_ci_mask[]_ci_imaging[__rows_(0,1)]"
        )

        settings = ac.SettingsPhaseCIImaging(
            settings_cti=ac.SettingsCTI(
                parallel_total_density_range=None, serial_total_density_range=None
            ),
            settings_mask=ac.SettingsMask(
                cosmic_ray_parallel_buffer=4,
                cosmic_ray_serial_buffer=5,
                cosmic_ray_diagonal_buffer=6,
            ),
            settings_ci_mask=ac.ci.SettingsCIMask(
                parallel_front_edge_rows=None,
                parallel_trails_rows=None,
                serial_front_edge_columns=None,
                serial_trails_columns=None,
            ),
            settings_masked_ci_imaging=ac.ci.SettingsMaskedCIImaging(
                parallel_columns=None, serial_rows=(1, 2)
            ),
        )

        assert (
            settings.phase_tag
            == "settings__cti[]_mask[__cr_p4s5d6]_ci_mask[]_ci_imaging[__rows_(1,2)]"
        )
