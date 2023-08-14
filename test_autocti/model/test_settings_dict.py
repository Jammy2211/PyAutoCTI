import os

import pytest

import autocti as ac


@pytest.fixture(name="settings_dict")
def make_settings_dict():
    return {
        "type": "autocti.model.settings.SettingsCTI2D",
        "parallel_total_density_range": None,
        "serial_total_density_range": None,
    }


def test_settings_from_dict(settings_dict):
    assert isinstance(ac.SettingsCTI2D.from_dict(settings_dict), ac.SettingsCTI2D)


def test_file():
    filename = "/tmp/temp.json"
    ac.SettingsCTI2D().output_to_json(filename)

    try:
        assert isinstance(ac.SettingsCTI2D().from_json(filename), ac.SettingsCTI2D)
    finally:
        os.remove(filename)
