import numpy as np
import math
from typing import Dict, List, Optional, Tuple
import autoarray as aa

from autocti.layout.two_d import Layout2D


def generate_column(size: int, norm: float, row_slope: float) -> np.ndarray:
    """
    Generate a column of non-uniform charge, including row non-uniformity.

    The pixel-numbering used to generate non-uniformity across the charge injection rows runs from 1 -> size

    Parameters
    -----------
    size
        The size of the non-uniform column of charge
    normalization
        The input normalization of the column's charge e.g. the level of charge injected.

    """
    return norm * (np.arange(1, size + 1)) ** row_slope


def region_ci_from(
    region_dimensions: Tuple[int, int],
    injection_norm_list: List[float],
    row_slope: Optional[float] = 0.0,
) -> np.ndarray:
    """
    Generate a non-uniform charge injection region from an input list of normalization values across columns.
    """

    ci_rows = region_dimensions[0]
    ci_region = np.zeros(region_dimensions)

    for column_index, injection_norm in enumerate(injection_norm_list):

        ci_region[0:ci_rows, column_index] = generate_column(
            size=ci_rows, norm=injection_norm, row_slope=row_slope
        )

    return ci_region


def region_list_ci_via_electronics_from(
    injection_on: int,
    injection_off: int,
    injection_total: int,
    parallel_size: int,
    serial_size: int,
    serial_prescan_size: int,
    serial_overscan_size: int,
    roe_corner: Tuple[int, int],
):

    region_list_ci = []

    injection_start_count = 0

    for index in range(injection_total):

        if roe_corner == (0, 0):

            ci_region = (
                parallel_size - (injection_start_count + injection_on),
                parallel_size - injection_start_count,
                serial_prescan_size,
                serial_size - serial_overscan_size,
            )

        elif roe_corner == (1, 0):

            ci_region = (
                injection_start_count,
                injection_start_count + injection_on,
                serial_prescan_size,
                serial_size - serial_overscan_size,
            )

        elif roe_corner == (0, 1):

            ci_region = (
                parallel_size - (injection_start_count + injection_on),
                parallel_size - injection_start_count,
                serial_overscan_size,
                serial_size - serial_prescan_size,
            )

        elif roe_corner == (1, 1):

            ci_region = (
                injection_start_count,
                injection_start_count + injection_on,
                serial_overscan_size,
                serial_size - serial_prescan_size,
            )

        region_list_ci.append(ci_region)

        injection_start_count += injection_on + injection_off

    return region_list_ci
