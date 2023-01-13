from typing import List

from autoarray.structures.arrays.uniform_2d import Array2D


def master_ci_from(ci_list: List[Array2D]) -> Array2D:
    """
    Determine the master charge injection frame from a list of observations of charge injection images, which all use
    the same charge injection parameters and therefore should all be identical except for read noise.

    The master frame is computed by summing all images in the list and dividing by the number of images, e.g. it is
    the mean value across the images in every pixel.

    There are many effects this function does not account for (bias, NL, cosmics, CTI, etc.) that will most likely
    need to be accounted for before a more realistic implementation is possible.

    Parameters
    ----------
    ci_list
        A list of charge injection images all of which are taken using the same charge injection
        parameters / electronics.

    Returns
    -------
    The estimated master charge injection image.
    """

    master_ci = sum(ci_list) / len(ci_list)

    return Array2D.no_mask(
        values=master_ci.native, pixel_scales=ci_list[0].pixel_scales
    )
