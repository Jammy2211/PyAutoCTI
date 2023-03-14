from typing import Tuple

import autoarray as aa


def binned_region_1d_fpr_from(pixels: Tuple[int, int]) -> aa.Region1D:
    """
    The `Extract` objects allow one to extract a `Dataset1D` from a 2D CTI dataset, which is used to perform
    CTI modeling in 1D.

    This is performed by binning up the data via the `binned_array_1d_from` function.

    In order to create the 1D dataset a `Layout1D` is required, which requires the `region_list` containing the
    charge regions on the 1D dataset (e.g. where the FPR appears in 1D after binning).

    The function returns the this region if the 1D dataset is extracted from the FPRs. This is the full range of
    the `pixels` tuple, unless negative entries are included, meaning that pixels before the FPRs are also extracted.

    Parameters
    ----------
    pixels
        The row / column pixel index to extract the FPRs between (e.g. `pixels=(0, 3)` extracts the 1st, 2nd and 3rd
        FPR rows / columns)
    """
    if pixels[0] <= 0 and pixels[1] <= 0:
        return None
    elif pixels[0] >= 0:
        return aa.Region1D(region=(0, pixels[1]))
    return aa.Region1D(region=(abs(pixels[0]), pixels[1] + abs(pixels[0])))


def binned_region_1d_eper_from(pixels: Tuple[int, int]) -> aa.Region1D:
    """
    `Extract` objects extract arrays and other values from a 2D CTI dataset, which is used to perform
    CTI modeling in 1D.

    This is performed by binning up the data via the `binned_array_1d_from` function.

    In order to create the 1D dataset a `Layout1D` is required, which requires the `region_list` containing the
    charge regions on the 1D dataset (e.g. where the FPR appears in 1D after binning).

    The function returns this region if the 1D dataset is extracted from the EPERs. The charge region is only included
    if there are negative entries in the `pixels` tuple, meaning that pixels before the EPERs (e.g. the FPR) are
    extracted.

    Parameters
    ----------
    pixels
        The row / column pixel index to extract the FPRs between (e.g. `pixels=(0, 3)` extracts the 1st, 2nd and 3rd
        FPR rows / columns)
    """
    if pixels[0] >= 0:
        return None
    elif pixels[1] >= 0:
        return aa.Region1D(region=(0, -pixels[0]))
    return aa.Region1D(region=(0, pixels[1] - pixels[0]))
