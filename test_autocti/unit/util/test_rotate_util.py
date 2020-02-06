import autocti as ac
import numpy as np
import pytest


def test__rotate_array__all_4_rotations_with_rotation_back():

    arr = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0]])

    arr_bl = ac.util.rotate.rotate_array_from_roe_corner(array=arr, roe_corner=(1, 0))

    assert arr_bl == pytest.approx(
        np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0]]), 1.0e-4
    )

    arr_bl = ac.util.rotate.rotate_array_from_roe_corner(
        array=arr_bl, roe_corner=(1, 0)
    )

    assert arr_bl == pytest.approx(
        np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0]]), 1.0e-4
    )

    arr_br = ac.util.rotate.rotate_array_from_roe_corner(array=arr, roe_corner=(1, 1))

    assert arr_br == pytest.approx(
        np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 0.0, 0.0]]), 1.0e-4
    )

    arr_br = ac.util.rotate.rotate_array_from_roe_corner(
        array=arr_br, roe_corner=(1, 1)
    )

    assert arr_br == pytest.approx(
        np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0]]), 1.0e-4
    )

    arr_tl = ac.util.rotate.rotate_array_from_roe_corner(array=arr, roe_corner=(0, 0))

    assert arr_tl == pytest.approx(
        np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 1.0, 0.0]]), 1.0e-4
    )

    arr_tl = ac.util.rotate.rotate_array_from_roe_corner(
        array=arr_tl, roe_corner=(0, 0)
    )

    assert arr_tl == pytest.approx(
        np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0]]), 1.0e-4
    )

    arr_tr = ac.util.rotate.rotate_array_from_roe_corner(array=arr, roe_corner=(0, 1))

    assert arr_tr == pytest.approx(
        np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 1.0], [0.0, 1.0, 0.0]]), 1.0e-4
    )

    arr_tr = ac.util.rotate.rotate_array_from_roe_corner(
        array=arr_tr, roe_corner=(0, 1)
    )

    assert arr_tr == pytest.approx(
        np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0]]), 1.0e-4
    )


def test__rotate_region__all_4_rotations_with_rotation_back():

    region = (0, 2, 1, 3)

    shape_2d = (8, 10)

    region_bl = ac.util.rotate.rotate_region_from_roe_corner(
        region=region, shape_2d=shape_2d, roe_corner=(1, 0)
    )

    assert region_bl == (0, 2, 1, 3)

    region_bl = ac.util.rotate.rotate_region_from_roe_corner(
        region=region_bl, shape_2d=shape_2d, roe_corner=(1, 0)
    )

    assert region_bl == (0, 2, 1, 3)

    region_br = ac.util.rotate.rotate_region_from_roe_corner(
        region=region, shape_2d=shape_2d, roe_corner=(1, 1)
    )

    assert region_br == (0, 2, 7, 9)

    region_br = ac.util.rotate.rotate_region_from_roe_corner(
        region=region_br, shape_2d=shape_2d, roe_corner=(1, 1)
    )

    assert region_br == (0, 2, 1, 3)

    region_tl = ac.util.rotate.rotate_region_from_roe_corner(
        region=region, shape_2d=shape_2d, roe_corner=(0, 0)
    )

    assert region_tl == (6, 8, 1, 3)

    region_tl = ac.util.rotate.rotate_region_from_roe_corner(
        region=region_tl, shape_2d=shape_2d, roe_corner=(0, 0)
    )

    assert region_tl == (0, 2, 1, 3)

    region_tr = ac.util.rotate.rotate_region_from_roe_corner(
        region=region, shape_2d=shape_2d, roe_corner=(0, 1)
    )

    assert region_tr == (6, 8, 7, 9)

    region_tr = ac.util.rotate.rotate_region_from_roe_corner(
        region=region_tr, shape_2d=shape_2d, roe_corner=(0, 1)
    )

    assert region_tr == (0, 2, 1, 3)


def test__rotate_ci_pattern__all_4_rotations_with_rotation_back():
    ci_pattern = ac.CIPatternUniform(
        regions=[(0, 1, 1, 2), (0, 2, 0, 2)], normalization=10.0
    )

    shape_2d = (2, 2)

    ci_pattern_bl = ac.util.rotate.rotate_ci_pattern_from_roe_corner(
        ci_pattern=ci_pattern, shape_2d=shape_2d, roe_corner=(1, 0)
    )

    assert ci_pattern_bl.regions == [(0, 1, 1, 2), (0, 2, 0, 2)]
    assert ci_pattern_bl.normalization == 10.0

    ci_pattern_bl = ac.util.rotate.rotate_ci_pattern_from_roe_corner(
        ci_pattern=ci_pattern_bl, shape_2d=shape_2d, roe_corner=(1, 0)
    )

    assert ci_pattern_bl.regions == [(0, 1, 1, 2), (0, 2, 0, 2)]

    ci_pattern_br = ac.util.rotate.rotate_ci_pattern_from_roe_corner(
        ci_pattern=ci_pattern, shape_2d=shape_2d, roe_corner=(1, 1)
    )

    assert ci_pattern_br.regions == [(0, 1, 0, 1), (0, 2, 0, 2)]

    ci_pattern_br = ac.util.rotate.rotate_ci_pattern_from_roe_corner(
        ci_pattern=ci_pattern_br, shape_2d=shape_2d, roe_corner=(1, 1)
    )

    assert ci_pattern_br.regions == [(0, 1, 1, 2), (0, 2, 0, 2)]

    ci_pattern_tl = ac.util.rotate.rotate_ci_pattern_from_roe_corner(
        ci_pattern=ci_pattern, shape_2d=shape_2d, roe_corner=(0, 0)
    )

    assert ci_pattern_tl.regions == [(1, 2, 1, 2), (0, 2, 0, 2)]

    ci_pattern_tl = ac.util.rotate.rotate_ci_pattern_from_roe_corner(
        ci_pattern=ci_pattern_tl, shape_2d=shape_2d, roe_corner=(0, 0)
    )

    assert ci_pattern_tl.regions == [(0, 1, 1, 2), (0, 2, 0, 2)]

    ci_pattern_tr = ac.util.rotate.rotate_ci_pattern_from_roe_corner(
        ci_pattern=ci_pattern, shape_2d=shape_2d, roe_corner=(0, 1)
    )

    assert ci_pattern_tr.regions == [(1, 2, 0, 1), (0, 2, 0, 2)]

    ci_pattern_tr = ac.util.rotate.rotate_ci_pattern_from_roe_corner(
        ci_pattern=ci_pattern_tr, shape_2d=shape_2d, roe_corner=(0, 1)
    )

    assert ci_pattern_tr.regions == [(0, 1, 1, 2), (0, 2, 0, 2)]
