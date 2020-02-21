import numpy as np

from autocti.charge_injection import ci_frame
from autocti.charge_injection import ci_pattern as pattern
from autoarray.util import array_util


class CIImaging(object):
    def __init__(self, image, noise_map, ci_pre_cti, cosmic_ray_map=None):
        self.image = image
        self.noise_map = noise_map
        self.ci_pre_cti = ci_pre_cti
        self.cosmic_ray_map = cosmic_ray_map

    @property
    def ci_pattern(self):
        return self.image.ci_pattern

    @property
    def shape_2d(self):
        return self.image.shape_2d

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

    def parallel_calibration_ci_imaging_for_columns(self, columns):
        """
        Creates a function to extract a parallel section for given columns
        """

        cosmic_ray_map = (
            self.cosmic_ray_map.parallel_calibration_frame_from_columns(columns=columns)
            if self.cosmic_ray_map is not None
            else None
        )

        return CIImaging(
            image=self.image.parallel_calibration_frame_from_columns(columns=columns),
            noise_map=self.noise_map.parallel_calibration_frame_from_columns(
                columns=columns
            ),
            ci_pre_cti=self.ci_pre_cti.parallel_calibration_frame_from_columns(
                columns=columns
            ),
            cosmic_ray_map=cosmic_ray_map,
        )

    def serial_calibration_ci_imaging_for_rows(self, rows):
        """
        Creates a function to extract a serial section for given rows
        """

        cosmic_ray_map = (
            self.cosmic_ray_map.serial_calibration_frame_from_rows(rows=rows)
            if self.cosmic_ray_map is not None
            else None
        )

        return CIImaging(
            image=self.image.serial_calibration_frame_from_rows(rows=rows),
            noise_map=self.noise_map.serial_calibration_frame_from_rows(rows=rows),
            ci_pre_cti=self.ci_pre_cti.serial_calibration_frame_from_rows(rows=rows),
            cosmic_ray_map=cosmic_ray_map,
        )

    @classmethod
    def from_fits(
        cls,
        roe_corner,
        ci_pattern,
        image_path,
        pixel_scales=None,
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
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

        ci_image = ci_frame.CIFrame.from_fits(
            file_path=image_path,
            hdu=image_hdu,
            roe_corner=roe_corner,
            ci_pattern=ci_pattern,
            pixel_scales=pixel_scales,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

        if noise_map_path is not None:
            ci_noise_map = array_util.numpy_array_2d_from_fits(
                file_path=noise_map_path, hdu=noise_map_hdu
            )
        else:
            ci_noise_map = np.ones(ci_image.shape_2d) * noise_map_from_single_value

        ci_noise_map = ci_frame.CIFrame.manual(
            array=ci_noise_map,
            roe_corner=roe_corner,
            ci_pattern=ci_pattern,
            pixel_scales=pixel_scales,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

        if ci_pre_cti_path is not None:
            ci_pre_cti = array_util.numpy_array_2d_from_fits(
                file_path=ci_pre_cti_path, hdu=ci_pre_cti_hdu
            )
        else:
            ci_pre_cti = ci_pre_cti_from_ci_pattern_geometry_image_and_mask(
                ci_pattern, ci_image, mask=mask
            )

        ci_pre_cti = ci_frame.CIFrame.manual(
            array=ci_pre_cti,
            roe_corner=roe_corner,
            ci_pattern=ci_pattern,
            pixel_scales=pixel_scales,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

        if cosmic_ray_map_path is not None:

            cosmic_ray_map = ci_frame.CIFrame.from_fits(
                file_path=cosmic_ray_map_path,
                hdu=cosmic_ray_map_hdu,
                roe_corner=roe_corner,
                ci_pattern=ci_pattern,
                pixel_scales=pixel_scales,
                parallel_overscan=parallel_overscan,
                serial_prescan=serial_prescan,
                serial_overscan=serial_overscan,
            )

        else:
            cosmic_ray_map = None

        return CIImaging(
            image=ci_image,
            noise_map=ci_noise_map,
            ci_pre_cti=ci_pre_cti,
            cosmic_ray_map=cosmic_ray_map,
        )

    @classmethod
    def simulate(
        cls,
        clocker,
        ci_pre_cti,
        ci_pattern,
        parallel_traps=None,
        parallel_ccd_volume=None,
        serial_traps=None,
        serial_ccd_volume=None,
        read_noise=None,
        cosmic_ray_map=None,
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

        if cosmic_ray_map is not None:
            ci_pre_cti += cosmic_ray_map

        ci_post_cti = clocker.add_cti(
            image=ci_pre_cti,
            parallel_traps=parallel_traps,
            parallel_ccd_volume=parallel_ccd_volume,
            serial_traps=serial_traps,
            serial_ccd_volume=serial_ccd_volume,
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
            image=ci_frame.CIFrame.manual(array=ci_image, ci_pattern=ci_pattern),
            noise_map=ci_frame.CIFrame.manual(
                array=ci_noise_map, ci_pattern=ci_pattern
            ),
            ci_pre_cti=ci_frame.CIFrame.manual(array=ci_pre_cti, ci_pattern=ci_pattern),
            cosmic_ray_map=ci_frame.CIFrame.manual(
                array=cosmic_ray_map, ci_pattern=ci_pattern
            ),
        )


def ci_pre_cti_from_ci_pattern_geometry_image_and_mask(ci_pattern, image, mask=None):
    """Setup a pre-cti image from this charge injection ci_data, using the charge injection ci_pattern.

    The pre-cti image is computed depending on whether the charge injection ci_pattern is uniform, non-uniform or \
    'fast' (see ChargeInjectPattern).
    """
    if isinstance(ci_pattern, pattern.CIPatternUniform):
        return ci_pattern.ci_pre_cti_from_shape(image.shape)
    return ci_pattern.ci_pre_cti_from_ci_image_and_mask(image, mask)


def output_ci_data_to_fits(
    ci_data,
    image_path,
    noise_map_path=None,
    ci_pre_cti_path=None,
    cosmic_ray_map_path=None,
    overwrite=False,
):

    array_util.numpy_array_2d_to_fits(
        array_2d=ci_data.image, file_path=image_path, overwrite=overwrite
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
