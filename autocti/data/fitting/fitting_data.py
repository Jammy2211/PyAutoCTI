import numpy as np

from autolens.data.imaging import image as im, convolution
from autolens.data.array import grids


class FittingImage(im.Image):

    def __new__(cls, image, mask):
        return np.array(image).view(cls)

    def __init__(self, image, mask):
        """A fitting image is the collection of data components (e.g. the image, noise-maps, PSF, etc.) which are used \
        to generate and fit it with a model image.

        The fitting image is in 2D and masked, primarily to removoe cosmic rays.

        The fitting image also includes a number of attributes which are used to performt the fit, including (y,x) \
        grids of coordinates, convolvers and other utilities.

        Parameters
        ----------
        image : im.Image
            The 2D observed image and other observed quantities (noise-map, PSF, exposure-time map, etc.)
        mask: msk.Mask
            The 2D mask that is applied to image data.

        Attributes
        ----------
        image : ScaledSquarePixelArray
            The 2D observed image data (not an instance of im.Image, so does not include the other data attributes,
            which are explicitly made as new attributes of the fitting image).
        noise_map : NoiseMap
            An array describing the RMS standard deviation error in each pixel, preferably in units of electrons per
            second.
        mask: msk.Mask
            The 2D mask that is applied to image data.
        """
        super().__init__(array=image, pixel_scale=image.pixel_scale, noise_map=image.noise_map, psf=image.psf,
                         background_noise_map=image.background_noise_map, poisson_noise_map=image.poisson_noise_map,
                         exposure_time_map=image.exposure_time_map, background_sky_map=image.background_sky_map)

        self.image = image[:,:]
        self.noise_map = image.noise_map
        self.mask = mask

    def __array_finalize__(self, obj):
        super(FittingImage, self).__array_finalize__(obj)
        if isinstance(obj, FittingImage):
            self.image = obj.image
            self.noise_map = obj.noise_map
            self.mask = obj.mask


class FittingHyperImage(FittingImage):

    def __new__(cls, image, mask, hyper_model_image, hyper_galaxy_images, hyper_minimum_values, sub_grid_size=2,
                image_psf_shape=None):
        return np.array(mask.map_2d_array_to_masked_1d_array(image)).view(cls)

    def __init__(self, image, mask, hyper_model_image, hyper_galaxy_images, hyper_minimum_values, sub_grid_size=2,
                 image_psf_shape=None):
        """A fitting hyper image is a fitting_image (see *FittingImage) which additionally includes a set of \
        'hyper_data'. This hyper-data is the best-fit model images of the observed data from a previous analysis, \
        and it is used to scale the noise in the image, so as to avoid over-fitting localized regions of the image \
        where the model does not provide a good fit.

        Look at the *FittingImage* docstring for all parameters / attributes not specific to a hyper image.

        Parameters
        ----------
        hyper_model_images : [ndarray]
            List of the masked 1D array best-fit model image's to each observed image in a previous analysis.
        hyper_galaxy_images : [[ndarray]]
            List of the masked 1D array best-fit model image's of every galaxy to each observed image in a previous \
            analysis.

        Attributes
        ----------

        """
        super(FittingHyperImage, self).__init__(image=image, mask=mask, sub_grid_size=sub_grid_size,
                                                image_psf_shape=image_psf_shape)

        self.hyper_model_image = hyper_model_image
        self.hyper_galaxy_images = hyper_galaxy_images
        self.hyper_minimum_values = hyper_minimum_values

    def __array_finalize__(self, obj):
        super(FittingImage, self).__array_finalize__(obj)
        if isinstance(obj, FittingHyperImage):
            self.noise_map_ = obj.noise_map_
            self.background_noise_map_ = obj.background_noise_map_
            self.poisson_noise_map_ = obj.poisson_noise_map_
            self.exposure_time_map_ = obj.exposure_time_map_
            self.background_sky_map_ = obj.background_sky_map_
            self.mask = obj.mask
            self.convolver_image = obj.convolver_image
            self.grids = obj.grids
            self.border = obj.border
            self.hyper_model_image = obj.hyper_model_image
            self.hyper_galaxy_images = obj.hyper_galaxy_images
            self.hyper_minimum_values = obj.hyper_minimum_values