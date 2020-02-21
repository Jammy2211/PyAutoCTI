import numpy as np

from autocti.masked import masked_structures
from autocti.structures import mask as msk


class MaskedCIImaging(object):
    def __init__(self, ci_imaging, mask, noise_scaling_maps=None):
        """A fitting image is the collection of simulator components (e.g. the image, noise-maps, PSF, etc.) which are used \
        to generate and fit it with a model image.

        The fitting image is in 2D and masked, primarily to remove cosmic rays.

        The fitting image also includes a number of attributes which are used to performt the fit, including (y,x) \
        grids of coordinates, convolvers and other utilities.

        Parameters
        ----------
        image : im.Image
            The 2D observed image and other observed quantities (noise-map, PSF, exposure-time map, etc.)
        mask: msk.Mask | None
            The 2D mask that is applied to image simulator.

        Attributes
        ----------
        image : ScaledSquarePixelArray
            The 2D observed image simulator (not an instance of im.Image, so does not include the other simulator attributes,
            which are explicitly made as new attributes of the fitting image).
        noise_map : NoiseMap
            An arrays describing the RMS standard deviation error in each pixel, preferably in unit_label of electrons per
            second.
        mask: msk.Mask
            The 2D mask that is applied to image simulator.
        """

        self.ci_imaging = ci_imaging

        self.mask = mask

        self.image = masked_structures.MaskedCIFrame.from_ci_frame(
            ci_frame=ci_imaging.image, mask=mask
        )
        self.noise_map = masked_structures.MaskedCIFrame.from_ci_frame(
            ci_frame=ci_imaging.noise_map, mask=mask
        )
        self.ci_pre_cti = masked_structures.MaskedCIFrame.from_ci_frame(
            ci_frame=ci_imaging.ci_pre_cti, mask=mask
        )
        self.cosmic_ray_map = masked_structures.MaskedCIFrame.from_ci_frame(
            ci_frame=ci_imaging.cosmic_ray_map, mask=mask
        )
        self.noise_scaling_maps = noise_scaling_maps

    @property
    def signal_to_noise_map(self):
        """The estimated signal-to-noise_maps mappers of the image."""
        signal_to_noise_map = np.divide(self.image, self.noise_map)
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return signal_to_noise_map

    @property
    def signal_to_noise_max(self):
        """The maximum value of signal-to-noise_maps in an image pixel in the image's signal-to-noise_maps mappers"""
        return np.max(self.signal_to_noise_map)

    @classmethod
    def for_parallel_from_columns(
        cls, ci_imaging, mask, columns, noise_scaling_maps=None
    ):
        """
        Creates a MaskedCIData object for a parallel section of the CCD

        Parameters
        ----------
        noise_scaling_maps
            A list of maps that are used to scale noise
        columns
            Columns to be extracted
        mask
            A mask
        Returns
        -------
        MaskedCIImaging
        """

        return MaskedCIImaging(
            ci_imaging=ci_imaging.parallel_calibration_ci_imaging_for_columns(
                columns=columns
            ),
            mask=ci_imaging.image.parallel_calibration_mask_from_mask_and_columns(
                mask=mask, columns=columns
            ),
            noise_scaling_maps=noise_scaling_maps,
        )

    @classmethod
    def for_serial_from_rows(cls, ci_imaging, mask, rows, noise_scaling_maps=None):
        """
        Creates a MaskedCIData object for a serial section of the CCD

        Parameters
        ----------
        noise_scaling_maps
            A list of maps that are used to scale noise
        rows
            Columns to be extracted
        mask
            A mask
        Returns
        -------
        MaskedCIImaging
        """

        return MaskedCIImaging(
            ci_imaging=ci_imaging.serial_calibration_ci_imaging_for_rows(rows=rows),
            mask=ci_imaging.image.serial_calibration_mask_from_mask_and_rows(
                mask=mask, rows=rows
            ),
            noise_scaling_maps=noise_scaling_maps,
        )
