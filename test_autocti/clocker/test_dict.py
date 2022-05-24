import os

import pytest
from autocti import Clocker2D


@pytest.fixture(name="clocker_dict")
def make_clocker_dict():
    return {
        "type": "autocti.clocker.two_d.Clocker2D",
        "iterations": 5,
        "parallel_roe": {
            "type": "arcticpy.src.roe.ROE",
            "dwell_times": {
                "type": "numpy.ndarray",
                "array": [1.0],
                "dtype": "float64",
            },
            "empty_traps_between_columns": True,
            "empty_traps_for_first_transfers": False,
            "force_release_away_from_readout": True,
            "use_integer_express_matrix": False,
        },
        "parallel_express": 0,
        "parallel_window_offset": 0,
        "parallel_window_start": 0,
        "parallel_window_stop": -1,
        "parallel_poisson_traps": False,
        "parallel_fast_mode": False,
        "serial_roe": {
            "type": "arcticpy.src.roe.ROE",
            "dwell_times": {
                "type": "numpy.ndarray",
                "array": [1.0],
                "dtype": "float64",
            },
            "empty_traps_between_columns": True,
            "empty_traps_for_first_transfers": False,
            "force_release_away_from_readout": True,
            "use_integer_express_matrix": False,
        },
        "serial_express": 0,
        "serial_window_offset": 0,
        "serial_window_start": 0,
        "serial_window_stop": -1,
        "serial_fast_mode": False,
        "verbosity": 0,
        "poisson_seed": -1,
        "euclid_orientation_hack": False,
    }


def test_clocker_from_dict(clocker_dict):
    assert isinstance(Clocker2D.from_dict(clocker_dict), Clocker2D)


def test_file():
    filename = "/tmp/temp.json"
    Clocker2D().output_to_json(filename)

    try:
        assert isinstance(Clocker2D.from_json(filename), Clocker2D)
    finally:
        os.remove(filename)
