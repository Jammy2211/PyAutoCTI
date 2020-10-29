from autoarray.instruments import euclid
from autoarray.util import frame_util

from autocti.util import clocker as clk
from autocti.util import ccd
from autocti.util import traps

from autocti.charge_injection import ci_frame, ci_pattern as pattern
from autocti.charge_injection import ci_imaging

"""
Note on the rotations of frames:

The function 'non_uniform_frame_for_ou_sim' returns a frame that is rotated according to the ccd_id and quadrant id,
which given the quadrant unique define the orientation of the ndarray necessary to add paralle and serial
CTI in the correct direction. These are defined accoridng to:

    http://euclid.esac.esa.int/dm/dpdd/latest/le1dpd/dpcards/le1_visrawframe.html

This function will return a frame, which OU-Sim will then add the following effects too:

    - Add cosmic rays.
    - Bias.
    - Non-linearity.
    - Crosstalk?

The addition of CTI can then be performed using either the 'add_cti_to_frame_for_ou_sim' function, or the standard
function OU-Sim use to add CTI (I guess they ultimately both flow through arctic in an identical way, albeit our
code uses SWIG and omits the need to define arCTIc parameter files.

Due to the rotations performed above, this means all images produced will be ndarrays oriented in the same way. I 
am not clear on how standard ELViS data products are oriented, but it may be that we require rotations before writing
them to fits to oriented them in the way they are observed (VIS_CTI has tools for this, as I'm sure ELViS does too).
"""


def non_uniform_frame_for_ou_sim(
    ccd_id,
    quadrant_id,
    ci_normalization,
    parallel_size=2086,
    serial_size=2128,
    serial_overscan_size=29,
    pixel_scales=0.1,
):
    """
    Returns a charge injection line image suitable for OU-SIM to run through the ElVIS simulator.

    By default, this frame has dimensions (2086, 2128), representing a Euclid quadrant with a serial prescan of
    size 51 pixels, a serial overscan with 29 pixels and parallel overscan with 20 pixels.

    The charge injection line pattern is simulated using the VIS_CTI Processing element, and includes
    effects such as a non-uniform charge injection pattern. The charge injection is simulated in 3 distinct
    regions on the quadrant.

    This function assumes the same orientation for the charge injection line image, irrespective of the Euclid
    CCD ID and Quadrant ID. The orientation that it assumes has arctic clock the ndarray towards [0, 0]. However,
    based on an input CCD ID and Quadrant ID, the array is rotated to match Euclid clocking.

    Parameters
    ----------
    ccd_id : str
        The CCD ID of Euclid (runs 1 through 6)
    quadrant_id : str
        The quadrant id (E, F, G, H)
    ci_normalization : float
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

    shape_2d = (parallel_size, serial_size)

    """Specify the charge injection regions on the CCD, which in this case is 3 equally spaced rectangular blocks."""

    ci_regions = [
        (0, 450, 51, shape_2d[1] - serial_overscan_size),
        (650, 1100, 51, shape_2d[1] - serial_overscan_size),
        (1300, 1750, 51, shape_2d[1] - serial_overscan_size),
    ]

    """Use the charge injection normalizations and regions to create *CIPatternNonUniform* of every image we'll simulate."""
    ci_pattern = pattern.CIPatternNonUniform(
        normalization=ci_normalization,
        regions=ci_regions,
        row_slope=0.0,
        column_sigma=100.0,
        maximum_normalization=84700,
    )

    """Create every pre-cti charge injection image using each *CIPattern*"""
    ci_pre_cti = ci_pattern.ci_pre_cti_from(
        shape_2d=shape_2d, pixel_scales=pixel_scales
    )

    roe_corner = euclid.roe_corner_from(ccd_id=ccd_id, quadrant_id=quadrant_id)

    """Before passing this image to arCTIc to add CTI, we want to make it a *Frame* object, which:

        - Uses an input read-out electronics corner to perform all rotations of the image before / after adding CTI.
        - Also uses this corner to rotate images before outputting to .fits, such that *Frame* objects can be load via
         .fits with the correct orientation.
        - Includes information on different regions of the image, such as the serial prescan and overscans.
    """
    return frame_util.rotate_array_from_roe_corner(
        array=ci_pre_cti, roe_corner=roe_corner
    )


def add_cti_to_frame_for_ou_sim(frames):

    """
    The *Clocker* models the CCD read-out, including CTI.

    For parallel clocking, we use 'charge injection mode' which transfers the charge of every pixel over the full CCD.
    """
    clocker = clk.Clocker(
        parallel_express=2, parallel_charge_injection_mode=True, serial_express=2
    )

    """
    The CTI model used by arCTIc to add CTI to the input image in the parallel direction, which contains: 

        - 2 *TrapInstantCapture* species in the parallel direction.
        - A simple CCD volume beta parametrization.
        - 3 *TrapInstantCapture* species in the serial direction.
        - A simple CCD volume beta parametrization.
    """
    parallel_trap_0 = traps.TrapInstantCapture(density=0.13, release_timescale=1.25)
    parallel_trap_1 = traps.TrapInstantCapture(density=0.25, release_timescale=4.4)
    parallel_ccd = ccd.CCD(
        well_fill_power=0.8, well_notch_depth=0.0, full_well_depth=84700
    )
    serial_trap_0 = traps.TrapInstantCapture(density=0.0442, release_timescale=0.8)
    serial_trap_1 = traps.TrapInstantCapture(density=0.1326, release_timescale=4.0)
    serial_trap_2 = traps.TrapInstantCapture(density=3.9782, release_timescale=20.0)
    serial_ccd = ccd.CCD(
        well_fill_power=0.8, well_notch_depth=0.0, full_well_depth=84700
    )

    """
    To simulate charge injection image, we pass the pre-cti charge injection images above through a 
    *SimulatorCIImaging*.
    
    For use outside Euclid, the Simulator includes effects like adding read-noise to the data. We will disable these
    features given that this is handled by ELViS.
    """
    simulator = ci_imaging.SimulatorCIImaging(add_poisson_noise=False)

    ci_datasets = [
        simulator.from_ci_pre_cti(
            clocker=clocker,
            ci_pre_cti=frame.ci_pre_cti,
            ci_pattern=frame.ci_pattern,
            parallel_traps=[parallel_trap_0, parallel_trap_1],
            parallel_ccd=parallel_ccd,
            serial_traps=[serial_trap_0, serial_trap_1, serial_trap_2],
            serial_ccd=serial_ccd,
        )
        for frame in zip(frames)
    ]

    """
    Finally return the images to ELViS.
    """
    return [ci_dataset.image for ci_dataset in ci_datasets]
