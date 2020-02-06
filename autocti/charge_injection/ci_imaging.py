import numpy as np

from autocti.charge_injection import ci_frame as frame
from autocti.charge_injection import ci_pattern as pattern
from autocti.structures import frame
from autocti.structures import mask as msk
from autoarray.util import array_util


class CIImaging(object):
    def __init__(self, image, noise_map, ci_pre_cti, cosmic_ray_map=None):
        self.image = image
        self.noise_map = noise_map
        self.ci_pre_cti = ci_pre_cti
        self.cosmic_ray_map = cosmic_ray_map

    @classmethod
    def simulate(
        cls,
        ci_pre_cti,
        frame_geometry,
        ci_pattern,
        cti_params,
        cti_settings,
        read_noise=None,
        cosmic_ray_map=None,
        use_parallel_poisson_densities=False,
        noise_seed=-1,
    ):
        """Simulate a charge injection image, including effects like noises.

        Parameters
        -----------
        ci_pre_cti
        cosmic_ray_map
            The dimensions of the output simulated charge injection image.
        frame_geometry : ci_frame.CIQuadGeometry
            The quadrant geometry of the simulated image, defining where the parallel / serial overscans are and \
            therefore the direction of clocking and rotations before input into the cti algorithm.
        ci_pattern : ci_pattern.CIPatternSimulate
            The charge injection ci_pattern (regions, normalization, etc.) of the charge injection image.
        cti_params : ArcticParams.ArcticParams
            The CTI model parameters (trap density, trap lifetimes etc.).
        cti_settings : ArcticSettings.ArcticSettings
            The settings that control the cti clocking algorithm (e.g. ccd well_depth express option).
        read_noise : None or float
            The FWHM of the Gaussian read-noises added to the image.
        noise_seed : int
            Seed for the read-noises added to the image.
        """

        ci_frame = frame.CIFrame(frame_geometry=frame_geometry, ci_pattern=ci_pattern)

        if cosmic_ray_map is not None:
            ci_pre_cti += cosmic_ray_map

        ci_pre_cti = cti_image.FrameArray(
            frame_geometry=frame_geometry, array=ci_pre_cti
        )

        ci_post_cti = ci_pre_cti.add_cti_to_image(
            cti_params=cti_params,
            cti_settings=cti_settings,
            use_parallel_poisson_densities=use_parallel_poisson_densities,
        )

        if read_noise is not None:
            ci_image = ci_post_cti + read_noise_map_from_shape_and_sigma(
                shape=ci_post_cti.shape, sigma=read_noise, noise_seed=noise_seed
            )
            ci_noise_map = read_noise * np.ones(ci_post_cti.shape)
        else:
            ci_image = ci_post_cti
            ci_noise_map = None

        return CIImaging(
            ci_frame=ci_frame,
            image=ci_image,
            noise_map=ci_noise_map,
            ci_pre_cti=ci_pre_cti,
            ci_pattern=ci_pattern,
            cosmic_ray_map=cosmic_ray_map,
        )

    @property
    def ci_pattern(self):
        return self.image.ci_pattern

    @property
    def shape(self):
        return self.image.shape

    def map_to_ci_data_masked(self, func, mask, noise_scaling_maps=None):
        """
        Maps an extraction function onto the structures in this object, a mask and noise scaling maps.

        Parameters
        ----------
        func
            The extraction function
        mask
            A mask
        noise_scaling_maps
            A list of noise maps used for scaling noise in poorly fit regions

        Returns
        -------
        masked_ci_data: MaskedCIImaging
        """

        return MaskedCIImaging(
            image=func(self.image),
            noise_map=func(self.noise_map),
            ci_pre_cti=func(self.ci_pre_cti),
            mask=func(mask),
            ci_pattern=self.ci_pattern,
            ci_frame=self.ci_frame,
            noise_scaling_maps=list(map(func, noise_scaling_maps))
            if noise_scaling_maps is not None
            else None,
            cosmic_ray_map=self.cosmic_ray_map,
        )

    def parallel_extractor(self, columns):
        """
        Creates a function to extract a parallel section for given columns
        """

        def extractor(obj):
            return self.ci_frame.frame_geometry.parallel_calibration_section_for_columns(
                array=obj, columns=columns
            )

        return extractor

    def serial_extractor(self, rows):
        """
        Creates a function to extract a serial section for given rows
        """

        def extractor(obj):
            return self.ci_frame.frame_geometry.serial_calibration_section_for_rows(
                array=obj, rows=rows
            )

        return extractor

    def parallel_serial_extractor(self):
        """
        Creates a function to extract a parallel and serial calibration section
        """

        def extractor(obj):
            return self.ci_frame.frame_geometry.parallel_serial_calibration_section(
                array=obj
            )

        return extractor

    def parallel_ci_data_masked_from_columns_and_mask(
        self, columns: (int,), mask: msk.Mask, noise_scaling_maps: (np.ndarray,) = None
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
        return self.map_to_ci_data_masked(
            func=self.parallel_extractor(columns),
            mask=mask,
            noise_scaling_maps=noise_scaling_maps,
        )

    def serial_ci_data_masked_from_rows_and_mask(
        self, rows: (int,), mask: msk.Mask, noise_scaling_maps: (np.ndarray,) = None
    ):
        """
        Creates a MaskedCIData object for a serial section of the CCD

        Parameters
        ----------
        noise_scaling_maps
            A list of maps that are used to scale noise
        rows
            Rows to be extracted
        mask
            A mask
        Returns
        -------
        MaskedCIImaging
        """
        return self.map_to_ci_data_masked(
            func=self.serial_extractor(rows),
            mask=mask,
            noise_scaling_maps=noise_scaling_maps,
        )

    def parallel_serial_ci_data_masked_from_mask(
        self, mask: msk.Mask, noise_scaling_maps: (np.ndarray,) = None
    ):
        """
        Creates a MaskedCIData object for a section of the CCD

        Parameters
        ----------
        noise_scaling_maps
            A list of maps that are used to scale noise
        mask
            A mask
        Returns
        -------
        MaskedCIImaging
        """
        return self.map_to_ci_data_masked(
            func=self.parallel_serial_extractor(),
            mask=mask,
            noise_scaling_maps=noise_scaling_maps,
        )

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


def ci_pre_cti_from_ci_pattern_geometry_image_and_mask(ci_pattern, image, mask=None):
    """Setup a pre-cti image from this charge injection ci_data, using the charge injection ci_pattern.

    The pre-cti image is computed depending on whether the charge injection ci_pattern is uniform, non-uniform or \
    'fast' (see ChargeInjectPattern).
    """
    if isinstance(ci_pattern, pattern.CIPatternUniform):
        return ci_pattern.ci_pre_cti_from_shape(image.shape)
    return ci_pattern.ci_pre_cti_from_ci_image_and_mask(image, mask)


def ci_data_from_fits(
    frame_geometry,
    ci_pattern,
    image_path,
    image_hdu=0,
    noise_map_path=None,
    noise_map_hdu=0,
    noise_map_from_single_value=None,
    ci_pre_cti_path=None,
    ci_pre_cti_hdu=0,
    cosmic_ray_map_path=None,
    cosmic_ray_map_hdu=0,
    mask=None,
):

    ci_frame = frame.CIFrame(frame_geometry=frame_geometry, ci_pattern=ci_pattern)

    ci_image = array_util.numpy_array_2d_from_fits(file_path=image_path, hdu=image_hdu)

    if noise_map_path is not None:
        ci_noise_map = array_util.numpy_array_2d_from_fits(
            file_path=noise_map_path, hdu=noise_map_hdu
        )
    else:
        ci_noise_map = np.ones(ci_image.shape) * noise_map_from_single_value

    if ci_pre_cti_path is not None:
        ci_pre_cti = array_util.numpy_array_2d_from_fits(
            file_path=ci_pre_cti_path, hdu=ci_pre_cti_hdu
        )
    else:
        ci_pre_cti = ci_pre_cti_from_ci_pattern_geometry_image_and_mask(
            ci_pattern, ci_image, mask=mask
        )

    if cosmic_ray_map_path is not None:
        cosmic_ray_map = array_util.numpy_array_2d_from_fits(
            file_path=cosmic_ray_map_path, hdu=cosmic_ray_map_hdu
        )
    else:
        cosmic_ray_map = None

    return CIImaging(
        image=ci_image,
        noise_map=ci_noise_map,
        ci_pre_cti=ci_pre_cti,
        ci_pattern=ci_pattern,
        ci_frame=ci_frame,
        cosmic_ray_map=cosmic_ray_map,
    )


def output_ci_data_to_fits(
    ci_data,
    image_path,
    noise_map_path=None,
    ci_pre_cti_path=None,
    cosmic_ray_map_path=None,
    overwrite=False,
):

    array_util.numpy_array_2d_to_fits(
        array_2d=ci_data.profile_image, file_path=image_path, overwrite=overwrite
    )

    if ci_data.noise_map is not None and noise_map_path is not None:
        array_util.numpy_array_2d_to_fits(
            array_2d=ci_data.noise_map, file_path=noise_map_path, overwrite=overwrite
        )

    if ci_data.ci_pre_cti is not None and ci_pre_cti_path is not None:
        array_util.numpy_array_2d_to_fits(
            array_2d=ci_data.ci_pre_cti, file_path=ci_pre_cti_path, overwrite=overwrite
        )

    if ci_data.cosmic_ray_map is not None and cosmic_ray_map_path is not None:
        array_util.numpy_array_2d_to_fits(
            array_2d=ci_data.cosmic_ray_map,
            file_path=cosmic_ray_map_path,
            overwrite=overwrite,
        )


def read_noise_map_from_shape_and_sigma(shape, sigma, noise_seed=-1):
    """Generate a two-dimensional read noises-map, generating values from a Gaussian distribution with mean 0.0.

    Params
    ----------
    shape : (int, int)
        The (x,y) image_shape of the generated Gaussian noises map.
    read_noise : float
        Standard deviation of the 1D Gaussian that each noises value is drawn from
    seed : int
        The seed of the random number generator, used for the random noises maps.
    """
    if noise_seed == -1:
        # Use one seed, so all regions have identical column non-uniformity.
        noise_seed = np.random.randint(0, int(1e9))
    np.random.seed(noise_seed)
    read_noise_map = np.random.normal(loc=0.0, scale=sigma, size=shape)
    return read_noise_map
