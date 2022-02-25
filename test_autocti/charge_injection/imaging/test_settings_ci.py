import autocti as ac


def test__settings_ci__modify_via_fit_type():

    settings = ac.ci.SettingsImagingCI(parallel_pixels=None, serial_pixels=None)
    settings = settings.modify_via_fit_type(is_parallel_fit=False, is_serial_fit=False)

    assert settings.parallel_pixels is None
    assert settings.serial_pixels is None

    settings = ac.ci.SettingsImagingCI(parallel_pixels=(0, 1), serial_pixels=(0, 1))
    settings = settings.modify_via_fit_type(is_parallel_fit=False, is_serial_fit=False)

    assert settings.parallel_pixels == (0, 1)
    assert settings.serial_pixels == (0, 1)

    settings = ac.ci.SettingsImagingCI(parallel_pixels=(0, 1), serial_pixels=(0, 1))
    settings = settings.modify_via_fit_type(is_parallel_fit=True, is_serial_fit=False)

    assert settings.parallel_pixels == (0, 1)
    assert settings.serial_pixels is None

    settings = ac.ci.SettingsImagingCI(parallel_pixels=(0, 1), serial_pixels=(0, 1))
    settings = settings.modify_via_fit_type(is_parallel_fit=False, is_serial_fit=True)

    assert settings.parallel_pixels is None
    assert settings.serial_pixels == (0, 1)

    settings = ac.ci.SettingsImagingCI(parallel_pixels=(0, 1), serial_pixels=(0, 1))
    settings = settings.modify_via_fit_type(is_parallel_fit=True, is_serial_fit=True)

    assert settings.parallel_pixels is None
    assert settings.serial_pixels is None
