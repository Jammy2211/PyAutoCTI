import os

import pytest

import autocti as ac
from autoconf.dictable import from_dict, output_to_json, from_json


@pytest.fixture(name="settings_dict")
def make_settings_dict():
    return {
        "class_path": "autocti.model.settings.SettingsCTI2D",
        "type": "instance",
        "arguments": {
            "parallel_total_density_range": None,
            "serial_total_density_range": None,
        },
    }


def test_settings_from_dict(settings_dict):
    assert isinstance(from_dict(settings_dict), ac.SettingsCTI2D)


def test_file():
    filename = "/tmp/temp.json"
    output_to_json(ac.SettingsCTI2D(), filename)

    try:
        assert isinstance(from_json(filename), ac.SettingsCTI2D)
    finally:
        os.remove(filename)
