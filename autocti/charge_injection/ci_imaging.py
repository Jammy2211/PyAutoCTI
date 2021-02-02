import copy

import numpy as np
from autoconf import conf
from autocti.charge_injection import ci_frame, ci_pattern as pattern
from autoarray.dataset import preprocess, imaging
from autocti.mask import mask as msk
from autoarray.util import array_util
from autocti import exc


class CIImaging(imaging.AbstractImaging):
    def __init__(self, image, noise_map, ci_pre_cti, cosmic_ray_map=None, name=None):

        super().__init__(image=image, noise_map=noise_map, name=name)

        self.ci_pre_cti = ci_pre_cti
        self.cosmic_ray_map = cosmic_ray_map

    @property
    def mask(self):
        return msk.Mask2D.unmasked(
            shape_native=self.shape_native, pixel_scales=self.pixel_scales
        )

    @property
    def ci_pattern(self):
        return self.image.ci_pattern

    def parallel_calibration_ci_imaging_for_columns(self, columns):
        """
        Returnss a function to extract a parallel section for given columns
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
        Returnss a function to extract a serial section for given rows
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
        ci_pattern,
        image_path,
        pixel_scales,
        roe_corner=(1, 0),
        scans=None,
        image_hdu=0,
        noise_map_path=None,
        noise_map_hdu=0,
        noise_map_from_single_value=None,
        ci_pre_cti_path=None,
        ci_pre_cti_hdu=0,
        cosmic_ray_map_path=None,
        cosmic_ray_map_hdu=0,
    ):

        ci_image = ci_frame.CIFrame.from_fits(
            file_path=image_path,
            hdu=image_hdu,
            roe_corner=roe_corner,
            ci_pattern=ci_pattern,
            pixel_scales=pixel_scales,
            scans=scans,
        )

        if noise_map_path is not None:
            ci_noise_map = array_util.numpy_array_2d_from_fits(
                file_path=noise_map_path, hdu=noise_map_hdu
            )
        else:
            ci_noise_map = np.ones(ci_image.shape_native) * noise_map_from_single_value

        ci_noise_map = ci_frame.CIFrame.manual(
            array=ci_noise_map,
            roe_corner=roe_corner,
            ci_pattern=ci_pattern,
            pixel_scales=pixel_scales,
            scans=scans,
        )

        if ci_pre_cti_path is not None:
            ci_pre_cti = array_util.numpy_array_2d_from_fits(
                file_path=ci_pre_cti_path, hdu=ci_pre_cti_hdu
            )
        else:
            if isinstance(ci_pattern, pattern.CIPatternUniform):
                ci_pre_cti = ci_pattern.ci_pre_cti_from(
                    shape_native=ci_image.shape, pixel_scales=pixel_scales
                )
            else:
                raise exc.CIPatternException(
                    "Cannot estimate ci_pre_cti image from non-uniform charge injectiono pattern"
                )

        ci_pre_cti = ci_frame.CIFrame.manual(
            array=ci_pre_cti,
            roe_corner=roe_corner,
            ci_pattern=ci_pattern,
            pixel_scales=pixel_scales,
            scans=scans,
        )

        if cosmic_ray_map_path is not None:

            cosmic_ray_map = ci_frame.CIFrame.from_fits(
                file_path=cosmic_ray_map_path,
                hdu=cosmic_ray_map_hdu,
                roe_corner=roe_corner,
                ci_pattern=ci_pattern,
                pixel_scales=pixel_scales,
                scans=scans,
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


class SettingsMaskedCIImaging(imaging.AbstractSettingsMaskedImaging):
    def __init__(self, parallel_columns=None, serial_rows=None):

        super().__init__()

        self.parallel_columns = parallel_columns
        self.serial_rows = serial_rows

    @property
    def tag(self):
        return (
            f"{conf.instance['notation']['settings_tags']['ci_imaging']['ci_imaging']}["
            f"{self.parallel_columns_tag}"
            f"{self.serial_rows_tag}]"
        )

    @property
    def parallel_columns_tag(self):
        """Generate a columns tag, to customize phase names based on the number of columns of simulator extracted in the fit,
        which is used to speed up parallel CTI fits.

        This changes the phase settings folder as follows:

        columns = None -> settings
        columns = 10 -> settings__col_10
        columns = 60 -> settings__col_60
        """
        if self.parallel_columns == None:
            return ""
        else:
            x0 = str(self.parallel_columns[0])
            x1 = str(self.parallel_columns[1])
            return f"__{conf.instance['notation']['settings_tags']['ci_imaging']['parallel_columns']}_({x0},{x1})"

    @property
    def serial_rows_tag(self):
        """Generate a rows tag, to customize phase names based on the number of rows of simulator extracted in the fit,
        which is used to speed up serial CTI fits.

        This changes the phase settings folder as follows:

        rows = None -> settings
        rows = (0, 10) -> settings__rows_(0,10)
        rows = (20, 60) -> settings__rows_(20,60)
        """
        if self.serial_rows == None:
            return ""
        else:
            x0 = str(self.serial_rows[0])
            x1 = str(self.serial_rows[1])
            return f"__{conf.instance['notation']['settings_tags']['ci_imaging']['serial_rows']}_({x0},{x1})"

    def modify_via_fit_type(self, is_parallel_fit, is_serial_fit):
        """Modify the settings based on the type of fit being performed where:

        - If the fit is a parallel only fit (is_parallel_fit=True, is_serial_fit=False) the serial_rows are set to None
          and all other settings remain the same.

        - If the fit is a serial only fit (is_parallel_fit=False, is_serial_fit=True) the parallel_columns are set to
          None and all other settings remain the same.

        - If the fit is a parallel and serial fit (is_parallel_fit=True, is_serial_fit=True) the *parallel_columns* and
          *serial_rows* are set to None and all other settings remain the same.

         These settings reflect the appropriate way to extract the charge injection imaging data for fits which use a
         parallel only CTI model, serial only CTI model or fit both.

         Parameters
         ----------
         is_parallel_fit : bool
            If True, the CTI model that is used to fit the charge injection data includes a parallel CTI component.
         is_serial_fit : bool
            If True, the CTI model that is used to fit the charge injection data includes a serial CTI component.
        """

        settings = copy.copy(self)

        if is_parallel_fit:
            settings.serial_rows = None

        if is_serial_fit:
            settings.parallel_columns = None

        return settings


class MaskedCIImaging(imaging.AbstractMaskedImaging):
    def __init__(
        self,
        ci_imaging,
        mask,
        noise_scaling_maps=None,
        settings=SettingsMaskedCIImaging(),
    ):
        """A data is the collection of simulator components (e.g. the image, noise-maps, PSF, etc.) which are used \
        to generate and fit it with a model image.

        The data is in 2D and masked, primarily to remove cosmic rays.

        The data also includes a number of attributes which are used to performt the fit, including (y,x) \
        grids of coordinates, convolvers and other utilities.

        Parameters
        ----------
        image : im.Image
            The 2D observed image and other observed quantities (noise-map, PSF, exposure-time map, etc.)
        mask: msk.Mask2D | None
            The 2D mask that is applied to image simulator.

        Attributes
        ----------
        image : ScaledSquarePixelArray
            The 2D observed image simulator (not an instance of im.Image, so does not include the other simulator attributes,
            which are explicitly made as new attributes of the data).
        noise_map : NoiseMap
            An arrays describing the RMS standard deviation error in each pixel, preferably in units of electrons per
            second.
        mask: msk.Mask2D
            The 2D mask that is applied to image simulator.
        """

        self.ci_imaging_full = copy.deepcopy(ci_imaging)
        self.ci_imaging_full.subplot_noise_scaling_maps = noise_scaling_maps
        self.mask_full = copy.deepcopy(mask)

        if settings.parallel_columns is not None:

            ci_imaging = self.ci_imaging_full.parallel_calibration_ci_imaging_for_columns(
                columns=settings.parallel_columns
            )

            mask = self.ci_imaging_full.image.parallel_calibration_mask_from_mask_and_columns(
                mask=mask, columns=settings.parallel_columns
            )

            if noise_scaling_maps is not None:
                noise_scaling_maps = [
                    noise_scaling_map.parallel_calibration_frame_from_columns(
                        columns=settings.parallel_columns
                    )
                    for noise_scaling_map in noise_scaling_maps
                ]

        if settings.serial_rows is not None:

            ci_imaging = self.ci_imaging_full.serial_calibration_ci_imaging_for_rows(
                rows=settings.serial_rows
            )

            mask = self.ci_imaging_full.image.serial_calibration_mask_from_mask_and_rows(
                mask=mask, rows=settings.serial_rows
            )

            if noise_scaling_maps is not None:

                noise_scaling_maps = [
                    noise_scaling_map.serial_calibration_frame_from_rows(
                        rows=settings.serial_rows
                    )
                    for noise_scaling_map in noise_scaling_maps
                ]

        super().__init__(imaging=ci_imaging, mask=mask, settings=settings)

        self.image = ci_frame.CIFrame.from_ci_frame(
            ci_frame=ci_imaging.image, mask=mask
        )
        self.noise_map = ci_frame.CIFrame.from_ci_frame(
            ci_frame=ci_imaging.noise_map, mask=mask
        )
        self.ci_pre_cti = ci_imaging.ci_pre_cti

        if ci_imaging.cosmic_ray_map is not None:

            self.cosmic_ray_map = ci_frame.CIFrame.from_ci_frame(
                ci_frame=ci_imaging.cosmic_ray_map, mask=mask
            )

        else:

            self.cosmic_ray_map = None

        if noise_scaling_maps is not None:
            self.noise_scaling_maps = [
                ci_frame.CIFrame.from_ci_frame(ci_frame=noise_scaling_map, mask=mask)
                for noise_scaling_map in noise_scaling_maps
            ]
        else:
            self.noise_scaling_maps = None

    @property
    def ci_imaging(self):
        return self.imaging


class SimulatorCIImaging(imaging.AbstractSimulatorImaging):
    def __init__(
        self,
        shape_native,
        pixel_scales,
        read_noise=None,
        add_poisson_noise=False,
        scans=None,
        noise_if_add_noise_false=0.1,
        noise_seed=-1,
        ci_seed=-1,
    ):
        """A class representing a Imaging observation, using the shape of the image, the pixel scale,
        psf, exposure time, etc.

        Parameters
        ----------
        exposure_time_map : float
            The exposure time of an observation using this data_type.
        """

        super(SimulatorCIImaging, self).__init__(
            read_noise=read_noise,
            exposure_time=1.0,
            add_poisson_noise=add_poisson_noise,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

        self.shape_native = shape_native
        self.scans = scans
        self.pixel_scales = pixel_scales
        self.ci_seed = ci_seed

    def from_ci_pattern(
        self,
        ci_pattern,
        clocker,
        parallel_traps=None,
        parallel_ccd=None,
        serial_traps=None,
        serial_ccd=None,
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
            The CTI model parameters (trap density, trap release_timescales etc.).
        clocker : ArcticSettings.ArcticSettings
            The settings that control the cti clocking algorithm (e.g. ccd well_depth express option).
        read_noise : None or float
            The FWHM of the Gaussian read-noises added to the image.
        noise_seed : int
            Seed for the read-noises added to the image.
        """
        if isinstance(ci_pattern, pattern.CIPatternUniform):
            ci_pre_cti = ci_pattern.ci_pre_cti_from(
                shape_native=self.shape_native, pixel_scales=self.pixel_scales
            )
        else:
            ci_pre_cti = ci_pattern.ci_pre_cti_from(
                shape_native=self.shape_native,
                ci_seed=self.ci_seed,
                pixel_scales=self.pixel_scales,
            )

        return self.from_ci_pre_cti(
            ci_pre_cti=ci_pre_cti,
            ci_pattern=ci_pattern,
            clocker=clocker,
            parallel_traps=parallel_traps,
            parallel_ccd=parallel_ccd,
            serial_traps=serial_traps,
            serial_ccd=serial_ccd,
            cosmic_ray_map=cosmic_ray_map,
            name=name,
        )

    def from_ci_pre_cti(
        self,
        ci_pre_cti,
        ci_pattern,
        clocker,
        parallel_traps=None,
        parallel_ccd=None,
        serial_traps=None,
        serial_ccd=None,
        cosmic_ray_map=None,
        name=None,
    ):

        if cosmic_ray_map is not None:
            ci_pre_cti += cosmic_ray_map

        ci_post_cti = clocker.add_cti(
            image=ci_pre_cti,
            parallel_traps=parallel_traps,
            parallel_ccd=parallel_ccd,
            serial_traps=serial_traps,
            serial_ccd=serial_ccd,
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
            image=ci_frame.CIFrame.manual(
                array=ci_image,
                ci_pattern=ci_pattern,
                scans=self.scans,
                pixel_scales=self.pixel_scales,
            ),
            noise_map=ci_frame.CIFrame.manual(
                array=ci_noise_map,
                ci_pattern=ci_pattern,
                scans=self.scans,
                pixel_scales=self.pixel_scales,
            ),
            ci_pre_cti=ci_frame.CIFrame.manual(
                array=ci_pre_cti,
                ci_pattern=ci_pattern,
                scans=self.scans,
                pixel_scales=self.pixel_scales,
            ),
            cosmic_ray_map=ci_frame.CIFrame.manual(
                array=cosmic_ray_map,
                ci_pattern=ci_pattern,
                scans=self.scans,
                pixel_scales=self.pixel_scales,
            ),
            name=name,
        )
