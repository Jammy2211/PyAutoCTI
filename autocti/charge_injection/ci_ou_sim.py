from arctic import clock
from arctic import model
from arctic import traps

from autocti.charge_injection import ci_frame, ci_pattern as pattern
from autocti.charge_injection import ci_imaging

"""
Note on the rotations of frames:

The function 'non_uniform_frame_for_ou_sim' returns a frame that is rotated according to the ccd_id and quadrant id,
which given the quadrant unique define the orientation of the NumPy array necessary to add paralle and serial
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

Due to the rotations performed above, this means all images produced will be NumPy arrays oriented in the same way. I 
am not clear on how standard ELViS data products are oriented, but it may be that we require rotations before writing
them to fits to oriented them in the way they are observed (VIS_CTI has tools for this, as I'm sure ELViS does too).
"""


def non_uniform_frame_for_ou_sim(ccd_id, quadrant_id):

    """The 2D shape of the image"""

    shape_2d = (2068, 2119)

    """Specify the charge injection regions on the CCD, which in this case is 3 equally spaced rectangular blocks."""

    ci_regions = [
        (0, 450, 51, shape_2d[1] - 20),
        (650, 1100, 51, shape_2d[1] - 20),
        (1300, 1750, 51, shape_2d[1] - 20),
    ]

    """The normalization of every charge injection image - this list determines how many images are simulated."""
    normalizations = [100.0, 500.0, 1000.0, 5000.0, 10000.0, 25000.0, 50000.0, 84700.0]

    """Use the charge injection normalizations and regions to create *CIPatternNonUniform* of every image we'll simulate."""
    column_deviations = [100.0] * len(normalizations)
    row_slopes = [0.0] * len(normalizations)
    ci_patterns = [
        pattern.CIPatternNonUniform(
            normalization=normalization, regions=ci_regions, row_slope=row_slope
        )
        for normalization, row_slope in zip(normalizations, row_slopes)
    ]

    """Create every pre-cti charge injection image using each *CIPattern*"""
    ci_pre_ctis = [
        ci_pattern.simulate_ci_pre_cti(
            shape=shape_2d,
            column_deviation=column_deviation,
            maximum_normalization=84700,
        )
        for ci_pattern, column_deviation in zip(ci_patterns, column_deviations)
    ]

    """Before passing this image to arCTIc to add CTI, we want to make it a *Frame* object, which:

        - Uses an input read-out electronics corner to perform all rotations of the image before / after adding CTI.
        - Also uses this corner to rotate images before outputting to .fits, such that *Frame* objects can be load via
         .fits with the correct orientation.
        - Includes information on different regions of the image, such as the serial prescan and overscans.
    """
    ci_pre_ctis = [
        ci_frame.EuclidCIFrame.ccd_and_quadrant_id(
            array=ci_pre_cti,
            ci_pattern=ci_pattern,
            ccd_id=ccd_id,
            quadrant_id=quadrant_id,
        )
        for ci_pre_cti, ci_pattern in zip(ci_pre_ctis, ci_patterns)
    ]

    return ci_pre_ctis


def add_cti_to_frame_for_ou_sim(frames):

    """
    The *Clocker* models the CCD read-out, including CTI.

    For parallel clocking, we use 'charge injection mode' which transfers the charge of every pixel over the full CCD.
    """
    clocker = clock.Clocker(
        parallel_express=2, parallel_charge_injection_mode=True, serial_express=2
    )

    """
    The CTI model used by arCTIc to add CTI to the input image in the parallel direction, which contains: 

        - 2 *Trap* species in the parallel direction.
        - A simple CCD volume beta parametrization.
        - 3 *Trap* species in the serial direction.
        - A simple CCD volume beta parametrization.
    """
    parallel_trap_0 = traps.Trap(density=0.13, lifetime=1.25)
    parallel_trap_1 = traps.Trap(density=0.25, lifetime=4.4)
    parallel_ccd_volume = model.CCDVolume(
        well_fill_beta=0.8, well_notch_depth=0.0, well_max_height=84700
    )
    serial_trap_0 = traps.Trap(density=0.0442, lifetime=0.8)
    serial_trap_1 = traps.Trap(density=0.1326, lifetime=4.0)
    serial_trap_2 = traps.Trap(density=3.9782, lifetime=20.0)
    serial_ccd_volume = model.CCDVolume(
        well_fill_beta=0.8, well_notch_depth=0.0, well_max_height=84700
    )

    """
    To simulate charge injection image, we pass the pre-cti charge injection images above through a 
    *SimulatorCIImaging*.
    
    For use outside Euclid, the Simulator includes effects like adding read-noise to the data. We will disable these
    features given that this is handled by ELViS.
    """
    simulator = ci_imaging.SimulatorCIImaging(add_noise=False)

    ci_datasets = [
        simulator.from_image(
            clocker=clocker,
            ci_pre_cti=frame.ci_pre_cti,
            ci_pattern=frame.ci_pattern,
            parallel_traps=[parallel_trap_0, parallel_trap_1],
            parallel_ccd_volume=parallel_ccd_volume,
            serial_traps=[serial_trap_0, serial_trap_1, serial_trap_2],
            serial_ccd_volume=serial_ccd_volume,
        )
        for frame in zip(frames)
    ]

    """
    Finally return the images to ELViS.
    """
    return [ci_dataset.image for ci_dataset in ci_datasets]
