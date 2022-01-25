import numpy as np
from typing import List, Union

from autoarray.instruments import euclid
from autoarray.layout import layout_util
from autoarray.structures.arrays.two_d.array_2d import Array2D

from arcticpy.src import ccd
from arcticpy.src import traps
from autocti.util.clocker import Clocker2D

from autocti.charge_injection.layout import Layout2DCI
from autocti.charge_injection.layout import Layout2DCINonUniform

from autocti.charge_injection.layout import region_list_ci_from

"""
Note on the rotations of arrays:

The function 'non_uniform_array_for_ou_sim' returns an array that is rotated according to the iquad parameter
(which ELVIS uses to define which quadrant the data corresponds to). This uniquely defines the orientation of the 
ndarray necessary to add parallel and serial CTI in the correct direction. These are defined accoridng to:

    http://euclid.esac.esa.int/dm/dpdd/latest/le1dpd/dpcards/le1_visrawframe.html

This function will return a array, which OU-Sim will then add the following effects too:

    - Add cosmic rays.
    - Bias.
    - Non-linearity.
    - Crosstalk?

The addition of CTI can then be performed using either the 'add_cti_to_array_for_ou_sim' function, or the standard
function OU-Sim use to add CTI (I guess they ultimately both flow through arctic in an identical way, albeit our
code uses SWIG and omits the need to define arCTIc parameter files.

Due to the rotations performed above, this means all images produced will be ndarrays oriented in the same way. I 
am not clear on how standard ELViS data products are oriented, but it may be that we require rotations before writing
them to fits to oriented them in the way they are observed (VIS_CTI has tools for this, as I'm sure ELViS does too).
"""


def quadrant_id_from(iquad: int) -> str:
    """
    The ELVIS simulator uses the `iquad` parameter to determine how images are rotated before clocking via arctic.

    This script converts this parameter to the the `quadrant_id` used by PyAutoCTI, which in turn gives the appropriate
    `roe_corner` for rotation.

    The mapping of `iquad` to `quadrant_id` does not depend on the CCD id, because ELVIS has already performed
    extractions / rotations on the quadrant data beforehand.

    Parameters
    ----------
    iquad
        The ELVIS parameter defining the quadrant of the data and therefore the rotateion before arctic clocking.

    Returns
    -------
    str
        The quadrant ID string, which is either E, G, H or G
    """

    if iquad == 0:
        return "E"
    elif iquad == 1:
        return "F"
    elif iquad == 2:
        return "H"
    elif iquad == 3:
        return "G"


def charge_injection_array_from(
    iquad: int,
    injection_normalization: float,
    injection_total: int = 5,
    injection_on: int = 200,
    injection_off: int = 200,
    parallel_size: int = 2086,
    serial_size: int = 2128,
    serial_prescan_size: int = 51,
    serial_overscan_size: int = 29,
    pixel_scales: float = 0.1,
    use_non_uniform_pattern: bool = True,
    ci_seed: int = -1,
) -> Union[np.ndarray, Array2D]:
    """
    Returns a charge injection line image suitable for OU-SIM to run through the ElVIS simulator.

    By default, this array has dimensions (2086, 2128), representing a Euclid quadrant with a serial prescan of
    size 51 pixels, a serial overscan with 29 pixels and parallel overscan with 20 pixels.

    The charge injection line pattern is simulated using the VIS_CTI Processing element, and includes
    effects such as a non-uniform charge injection pattern. The charge injection is simulated in 3 distinct
    regions on the quadrant.

    This function assumes the same orientation for the charge injection line image, irrespective of the Euclid
    CCDPhase ID and Quadrant ID. The orientation that it assumes has arctic clock the ndarray towards [0, 0]. However,
    based on an input CCDPhase ID and Quadrant ID, the array is rotated to match Euclid clocking.

    Parameters
    ----------
    ccd_id : str
        The CCDPhase ID of Euclid (runs 1 through 6)
    quadrant_id : str
        The quadrant id (E, F, G, H)
    injection_normalization
        The normalization of the charge injection region.
    parallel_size : int
        The size of the image in the parallel clocking direction (e.g. number of rows).
    serial_size : int
        The size of the image in the serial clocking direction (e.g. number of columns).
    serial_overscan_size : int
        The size of the serial overscan
    pixel_scales : (float, float)
        The arc-second to pixel scale conversion factor.

    Returns
    -------
    ndarray
        The charge injection line image oriented to match a given Euclid quadrant.
    """
    shape_native = (parallel_size, serial_size)

    """
    Specify the charge injection regions on the CCDPhase, which in this case is 5 equally spaced rectangular blocks.
    
    At the end of this function the ndarray containing the charge injection data is rotated based on the quadrant_id.
    We therefore do not need rotated `regions_ci`'s from the function below, and input `roe_corner=(1,0)`, which 
    corresponds to quadrant E which is never rotated.
    """

    regions_ci = region_list_ci_from(
        injection_on=injection_on,
        injection_off=injection_off,
        injection_total=injection_total,
        parallel_size=parallel_size,
        serial_size=serial_size,
        serial_prescan_size=serial_prescan_size,
        serial_overscan_size=serial_overscan_size,
        roe_corner=(1, 0),
    )

    """
    Use the charge injection normalization_list and regions to create `Layout2DCINonUniform` of every image we'll simulate.
    """
    if use_non_uniform_pattern:
        layout = Layout2DCINonUniform(
            shape_2d=shape_native,
            normalization=injection_normalization,
            region_list=regions_ci,
            row_slope=0.0,
            column_sigma=100.0,
            maximum_normalization=84700,
        )
    else:
        layout = Layout2DCI(
            shape_2d=shape_native,
            normalization=injection_normalization,
            region_list=regions_ci,
        )

    """
    Create every pre-cti charge injection image using each `Layout2DCI`
    """
    pre_cti_data = layout.pre_cti_data_from(
        shape_native=shape_native, pixel_scales=pixel_scales, ci_seed=ci_seed
    )

    """
    The OU-SIM parameter iquad defines the quadrant_id of the data (e.g. "E", "F", "G" or "H").
    """
    quadrant_id = quadrant_id_from(iquad=iquad)

    roe_corner = euclid.roe_corner_from(ccd_id="1", quadrant_id=quadrant_id)

    """
    The array is rotated back to its original reference frame via the roe_corner, so other OU-Sim processing 
    works correctly.
    """
    return layout_util.rotate_array_via_roe_corner_from(
        array=pre_cti_data.native, roe_corner=roe_corner
    )


def add_cti_to_pre_cti_data(
    pre_cti_data: Union[np.ndarray, Array2D],
    iquad: int,
    clocker: Clocker2D,
    parallel_trap_list: List[traps.AbstractTrap],
    parallel_ccd: ccd.CCDPhase,
    serial_trap_list: List[traps.AbstractTrap],
    serial_ccd: ccd.CCDPhase,
) -> Union[np.ndarray, Array2D]:

    quadrant_id = quadrant_id_from(iquad=iquad)

    roe_corner = euclid.roe_corner_from(ccd_id="1", quadrant_id=quadrant_id)

    pre_cti_data = layout_util.rotate_array_via_roe_corner_from(
        array=pre_cti_data, roe_corner=roe_corner
    )

    """
    The `Clocker` models the CCDPhase read-out, including CTI.

    For parallel clocking, we use 'charge injection mode' which transfers the charge of every pixel over the full CCDPhase.
    """

    """
    The CTI model used by arCTIc to add CTI to the input image in the parallel direction, which contains: 

        - 2 `TrapInstantCapture` species in the parallel direction.
        - A simple CCDPhase volume beta parametrization.
        - 3 `TrapInstantCapture` species in the serial direction.
        - A simple CCDPhase volume beta parametrization.
    """

    post_cti_data = clocker.add_cti(
        data=pre_cti_data,
        parallel_trap_list=parallel_trap_list,
        parallel_ccd=parallel_ccd,
        serial_trap_list=serial_trap_list,
        serial_ccd=serial_ccd,
    )

    return layout_util.rotate_array_via_roe_corner_from(
        array=post_cti_data, roe_corner=roe_corner
    )
