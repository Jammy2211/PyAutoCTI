import numpy as np
import pytest
import autocti as ac
from autocti import exc


def test__trail_size_to_array_edge():
    layout = ac.Layout1D(shape_1d=(5,), region_list=[ac.Region1D(region=(0, 3))])

    assert layout.trail_size_to_array_edge == 2

    layout = ac.Layout1D(shape_1d=(7,), region_list=[ac.Region1D(region=(0, 3))])

    assert layout.trail_size_to_array_edge == 4

    layout = ac.Layout1D(
        shape_1d=(15,),
        region_list=[
            ac.Region1D(region=(0, 2)),
            ac.Region1D(region=(5, 8)),
            ac.Region1D(region=(11, 14)),
        ],
    )

    assert layout.trail_size_to_array_edge == 1

    layout = ac.Layout1D(
        shape_1d=(20,),
        region_list=[
            ac.Region1D(region=(0, 2)),
            ac.Region1D(region=(5, 8)),
            ac.Region1D(region=(11, 14)),
        ],
    )

    assert layout.trail_size_to_array_edge == 6
