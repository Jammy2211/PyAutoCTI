from copy import deepcopy

from autocti.structures import region as reg


def rotate_array_from_roe_corner(array, roe_corner):

    if roe_corner == (1, 0):
        return array.copy()
    elif roe_corner == (0, 0):
        return array[::-1, :].copy()
    elif roe_corner == (1, 1):
        return array[:, ::-1].copy()
    elif roe_corner == (0, 1):
        array = array[::-1, :].copy()
        return array[:, ::-1]


def rotate_region_from_roe_corner(region, shape_2d, roe_corner):

    if region is None:
        return None

    if roe_corner == (1, 0):
        return reg.Region(region=region)
    elif roe_corner == (0, 0):
        return reg.Region(
            region=(
                shape_2d[0] - region[1],
                shape_2d[0] - region[0],
                region[2],
                region[3],
            )
        )
    elif roe_corner == (1, 1):
        return reg.Region(
            region=(
                region[0],
                region[1],
                shape_2d[1] - region[3],
                shape_2d[1] - region[2],
            )
        )
    elif roe_corner == (0, 1):
        return reg.Region(
            region=(
                shape_2d[0] - region[1],
                shape_2d[0] - region[0],
                shape_2d[1] - region[3],
                shape_2d[1] - region[2],
            )
        )


def rotate_ci_pattern_from_roe_corner(ci_pattern, shape_2d, roe_corner):

    new_ci_pattern = deepcopy(ci_pattern)

    new_ci_pattern.regions = [
        rotate_region_from_roe_corner(
            region=region, shape_2d=shape_2d, roe_corner=roe_corner
        )
        for region in ci_pattern.regions
    ]

    return new_ci_pattern
