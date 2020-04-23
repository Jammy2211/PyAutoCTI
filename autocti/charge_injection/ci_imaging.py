import numpy as np
import copy

from autocti.dataset import preprocess, imaging
from autocti.charge_injection import ci_frame
from autocti.charge_injection import ci_pattern as pattern
from autocti.util import array_util


class CIImaging(imaging.Imaging):
    def __init__(self, image, noise_map, ci_pre_cti, cosmic_ray_map=None, name=None):

        super().__init__(image=image, noise_map=noise_map, name=name)

        self.ci_pre_cti = ci_pre_cti
        self.cosmic_ray_map = cosmic_ray_map

    @property
    def ci_pattern(self):
        return self.image.ci_pattern

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

    def output_to_fits(
        self,
        image_path,
        noise_map_path=None,
        ci_pre_cti_path=None,
        cosmic_ray_map_path=None,
        overwrite=False,
    ):

        array_util.numpy_array_2d_to_fits(
            array_2d=self.image, file_path=image_path, overwrite=overwrite
        )

        if self.noise_map is not None and noise_map_path is not None:
            array_util.numpy_array_2d_to_fits(
                array_2d=self.noise_map, file_path=noise_map_path, overwrite=overwrite
            )

        if self.ci_pre_cti is not None and ci_pre_cti_path is not None:
            array_util.numpy_array_2d_to_fits(
                array_2d=self.ci_pre_cti, file_path=ci_pre_cti_path, overwrite=overwrite
            )

        if self.cosmic_ray_map is not None and cosmic_ray_map_path is not None:
            array_util.numpy_array_2d_to_fits(
                array_2d=self.cosmic_ray_map,
                file_path=cosmic_ray_map_path,
                overwrite=overwrite,
            )


def ci_pre_cti_from_ci_pattern_geometry_image_and_mask(ci_pattern, image, mask=None):
    """Setup a pre-cti image from this charge injection ci_data, using the charge injection ci_pattern.

    The pre-cti image is computed depending on whether the charge injection ci_pattern is uniform, non-uniform or \
    'fast' (see ChargeInjectPattern).
    """
    if isinstance(ci_pattern, pattern.CIPatternUniform):
        return ci_pattern.ci_pre_cti_from_shape(image.shape)
    return ci_pattern.ci_pre_cti_from_ci_image_and_mask(image, mask)


class MaskedCIImaging(imaging.MaskedImaging):
    def __init__(
        self,
        ci_imaging,
        mask,
        noise_scaling_maps=None,
        parallel_columns=None,
        serial_rows=None,
    ):
        """A data is the collection of simulator components (e.g. the image, noise maps, PSF, etc.) which are used \
        to generate and fit it with a model image.

        The data is in 2D and masked, primarily to remove cosmic rays.

        The data also includes a number of attributes which are used to performt the fit, including (y,x) \
        grids of coordinates, convolvers and other utilities.

        Parameters
        ----------
        image : im.Image
            The 2D observed image and other observed quantities (noise map, PSF, exposure-time map, etc.)
        mask: msk.Mask | None
            The 2D mask that is applied to image simulator.

        Attributes
        ----------
        image : ScaledSquarePixelArray
            The 2D observed image simulator (not an instance of im.Image, so does not include the other simulator attributes,
            which are explicitly made as new attributes of the data).
        noise_map : NoiseMap
            An arrays describing the RMS standard deviation error in each pixel, preferably in units of electrons per
            second.
        mask: msk.Mask
            The 2D mask that is applied to image simulator.
        """

        self.ci_imaging_full = copy.deepcopy(ci_imaging)
        self.mask_full = copy.deepcopy(mask)

        if parallel_columns is not None:

            ci_imaging = self.ci_imaging_full.parallel_calibration_ci_imaging_for_columns(
                columns=parallel_columns
            )

            mask = self.ci_imaging_full.image.parallel_calibration_mask_from_mask_and_columns(
                mask=mask, columns=parallel_columns
            )

            if noise_scaling_maps is not None:
                noise_scaling_maps = [
                    noise_scaling_map.parallel_calibration_frame_from_columns(
                        columns=parallel_columns
                    )
                    for noise_scaling_map in noise_scaling_maps
                ]

        if serial_rows is not None:

            ci_imaging = self.ci_imaging_full.serial_calibration_ci_imaging_for_rows(
                rows=serial_rows
            )

            mask = self.ci_imaging_full.image.serial_calibration_mask_from_mask_and_rows(
                mask=mask, rows=serial_rows
            )

            if noise_scaling_maps is not None:

                noise_scaling_maps = [
                    noise_scaling_map.serial_calibration_frame_from_rows(
                        rows=serial_rows
                    )
                    for noise_scaling_map in noise_scaling_maps
                ]

        super().__init__(imaging=ci_imaging, mask=mask)

        self.image = ci_frame.MaskedCIFrame.from_ci_frame(
            ci_frame=ci_imaging.image, mask=mask
        )
        self.noise_map = ci_frame.MaskedCIFrame.from_ci_frame(
            ci_frame=ci_imaging.noise_map, mask=mask
        )
        self.ci_pre_cti = ci_frame.MaskedCIFrame.from_ci_frame(
            ci_frame=ci_imaging.ci_pre_cti, mask=mask
        )

        if ci_imaging.cosmic_ray_map is not None:

            self.cosmic_ray_map = ci_frame.MaskedCIFrame.from_ci_frame(
                ci_frame=ci_imaging.cosmic_ray_map, mask=mask
            )

        else:

            self.cosmic_ray_map = None

        if noise_scaling_maps is not None:
            self.noise_scaling_maps = [
                ci_frame.MaskedCIFrame.from_ci_frame(
                    ci_frame=noise_scaling_map, mask=mask
                )
                for noise_scaling_map in noise_scaling_maps
            ]
        else:
            self.noise_scaling_maps = None

    @property
    def ci_imaging(self):
        return self.imaging


class SimulatorCIImaging:
    def __init__(
        self,
        read_noise=None,
        add_noise=True,
        noise_if_add_noise_false=0.1,
        noise_seed=-1,
    ):
        """A class representing a Imaging observation, using the shape of the image, the pixel scale,
        psf, exposure time, etc.

        Parameters
        ----------
        exposure_time_map : float
            The exposure time of an observation using this data_type.
        """

        self.read_noise = read_noise
        self.add_noise = add_noise
        self.noise_if_add_noise_false = noise_if_add_noise_false
        self.noise_seed = noise_seed

    def from_image(
        self,
        clocker,
        ci_pre_cti,
        ci_pattern,
        parallel_traps=None,
        parallel_ccd_volume=None,
        serial_traps=None,
        serial_ccd_volume=None,
        cosmic_ray_map=None,
        name=None,
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
        clocker : ArcticSettings.ArcticSettings
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

        if self.read_noise is not None:
            ci_image = preprocess.data_with_gaussian_noise_added(
                data=ci_post_cti, sigma=self.read_noise, seed=self.noise_seed
            )
            ci_noise_map = self.read_noise * np.ones(ci_post_cti.shape)
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
            name=name,
        )
