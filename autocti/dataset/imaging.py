import copy
import logging

import numpy as np
from autoarray.dataset import imaging as im
from autocti.structures import arrays

logger = logging.getLogger(__name__)


class Imaging(im.AbstractImaging):
    def __init__(self, image, noise_map, name=None):
        """A collection of 2D imaging dataset(an image, noise-map, psf, etc.)

        Parameters
        ----------
        image : aa.Array
            The array of the image data, in units of electrons per second.
        noise_map : NoiseMap | float | ndarray
            An array describing the RMS standard deviation error in each pixel, preferably in units of electrons per
            second.
        background_noise_map : NoiseMap
            An array describing the RMS standard deviation error in each pixel due to the background sky noise_map,
            preferably in units of electrons per second.
        poisson_noise_map : NoiseMap
            An array describing the RMS standard deviation error in each pixel due to the Poisson counts of the source,
            preferably in units of electrons per second.
        exposure_time_map : aa.Array
            An array describing the effective exposure time in each imaging pixel.
        background_sky_map : aa.Scaled
            An array describing the background sky.
        """

        super(Imaging, self).__init__(image=image, noise_map=noise_map, name=name)

    @classmethod
    def from_fits(
        cls,
        image_path,
        pixel_scales,
        image_hdu=0,
        noise_map_path=None,
        noise_map_hdu=0,
        name=None,
    ):
        """Factory for loading the imaging data_type from .fits files, as well as computing properties like the noise-map,
        exposure-time map, etc. from the imaging-data.

        This factory also includes a number of routines for converting the imaging-data from unit_label not supported by PyAutoLens \
        (e.g. adus, electrons) to electrons per second.

        Parameters
        ----------
        name
        image_path : str
            The path to the image .fits file containing the image (e.g. '/path/to/image.fits')
        pixel_scales : float
            The size of each pixel in arc seconds.
        image_hdu : int
            The hdu the image is contained in the .fits file specified by *image_path*.
        noise_map_path : str
            The path to the noise_map .fits file containing the noise_map (e.g. '/path/to/noise_map.fits')
        noise_map_hdu : int
            The hdu the noise_map is contained in the .fits file specified by *noise_map_path*.
        """

        image = arrays.Array.from_fits(
            file_path=image_path, hdu=image_hdu, pixel_scales=pixel_scales
        )

        noise_map = arrays.Array.from_fits(
            file_path=noise_map_path, hdu=noise_map_hdu, pixel_scales=pixel_scales
        )

        return Imaging(image=image, noise_map=noise_map, name=name)

    def output_to_fits(self, image_path, noise_map_path=None, overwrite=False):

        self.image.output_to_fits(file_path=image_path, overwrite=overwrite)

        if self.noise_map is not None and noise_map_path is not None:
            self.noise_map.output_to_fits(file_path=noise_map_path, overwrite=overwrite)


class MaskedImaging(im.MaskedImaging):
    def __init__(self, imaging, mask):
        """
        The lens dataset is the collection of data_type (image, noise-map, PSF), a mask, grid, convolver \
        and other utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image, noise-map, etc. are loaded in 2D, the lens dataset creates reduced 1D arrays of each \
        for lens calculations.

        Parameters
        ----------
        imaging: im.Imaging
            The imaging data_type all in 2D (the image, noise-map, PSF, etc.)
        mask: msk.Mask2D
            The 2D mask that is applied to the image.
        """

        super().__init__(imaging=imaging, mask=mask)

        self.image = imaging.image * np.invert(mask)
        self.noise_map = imaging.noise_map * np.invert(mask)
