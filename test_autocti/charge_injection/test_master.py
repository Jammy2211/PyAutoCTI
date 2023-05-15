import numpy as np
import pytest

import autocti as ac

from autocti.charge_injection.master import master_ci_from


def test__master_ci_from():
    ci_0 = ac.Array2D.no_mask(values=[[1.0, 1.0], [2.0, 2.0]], pixel_scales=1.0)
    ci_1 = ac.Array2D.no_mask(values=[[2.0, 2.0], [4.0, 4.0]], pixel_scales=1.0)

    master_ci = master_ci_from(ci_list=[ci_0, ci_1])

    assert master_ci.native == pytest.approx(np.array([[1.5, 1.5], [3.0, 3.0]]), 1.0e-4)
