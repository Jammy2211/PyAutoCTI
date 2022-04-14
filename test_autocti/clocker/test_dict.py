import os

import pytest

from autocti import Clocker2D


@pytest.fixture(
    name="clocker_dict"
)
def make_clocker_dict():
    return {
        'euclid_orientation_hack': False,
        'iterations': 5,
        'parallel_express': 0,
        'parallel_fast_pixels': None,
        'parallel_poisson_traps': False,
        'parallel_roe': {
            'dwell_times': {
                'array': [1.0],
                'dtype': 'float64',
                'type': 'numpy.ndarray'
            },
            'empty_traps_between_columns': True,
            'empty_traps_for_first_transfers': False,
            'force_release_away_from_readout': True,
            'type': 'arcticpy.src.roe.ROE',
            'use_integer_express_matrix': False
        },
        'parallel_window_start': 0,
        'parallel_window_stop': -1,
        'poisson_seed': -1,
        'serial_express': 0,
        'serial_fast_pixels': None,
        'serial_roe': {
            'dwell_times': {
                'array': [1.0],
                'dtype': 'float64',
                'type': 'numpy.ndarray'
            },
            'empty_traps_between_columns': True,
            'empty_traps_for_first_transfers': False,
            'force_release_away_from_readout': True,
            'type': 'arcticpy.src.roe.ROE',
            'use_integer_express_matrix': False
        },
        'serial_window_start': 0,
        'serial_window_stop': -1,
        'type': 'autocti.clocker.two_d.Clocker2D',
        'verbosity': 0
    }


def test_clocker_as_dict(clocker_dict):
    assert Clocker2D().dict() == clocker_dict


def test_clocker_from_dict(clocker_dict):
    assert isinstance(
        Clocker2D.from_dict(
            clocker_dict
        ),
        Clocker2D
    )


def test_file():
    filename = "/tmp/temp.json"
    Clocker2D().output_to_json(filename)

    try:
        assert isinstance(
            Clocker2D.from_json(filename),
            Clocker2D
        )
    finally:
        os.remove(filename)
