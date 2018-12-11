class FittingImage(object):

    def __new__(cls, image, noise_map, mask):
        fitting_image = image
        fitting_image.noise_map = noise_map
        fitting_image.mask = mask
        return fitting_image

    def __init__(self, image, noise_map, mask):
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
        self.image = image
        self.noise_map = noise_map
        self.mask = mask

    def __array_finalize__(self, obj):
        if isinstance(obj, FittingImage):
            self.image = obj.image
            self.noise_map = obj.noise_map
            self.mask = obj.mask


class FittingHyperImage(FittingImage):

    def __new__(cls, image, noise_map, mask, hyper_noises):
        fitting_image = image
        fitting_image.noise_map = noise_map
        fitting_image.mask = mask
        fitting_image.hyper_noises = hyper_noises
        return fitting_image

    def __init__(self, image, noise_map, mask, hyper_noises):
        """A fitting hyper image is a fitting_image (see *FittingImage) which additionally includes a set of \
        'hyper_data'. This hyper-data is the best-fit model images of the observed data from a previous analysis, \
        and it is used to scale the noise in the image, so as to avoid over-fitting localized regions of the image \
        where the model does not provide a good fit.

        Look at the *FittingImage* docstring for all parameters / attributes not specific to a hyper image.

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
        self.image = image
        self.noise_map = noise_map
        self.mask = mask
        self.hyper_noises = hyper_noises

    def __array_finalize__(self, obj):
        if isinstance(obj, FittingHyperImage):
            self.image = obj.image
            self.noise_map = obj.noise_map
            self.mask = obj.mask
            self.hyper_noises = obj.hyper_noises
